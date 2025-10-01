"""FastAPI application exposing the FAQ assistant search/generation pipeline."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from .config import SettingsManager, AppSettings
from .search import (
    HybridSearchIndex,
    build_hybrid_index,
    hybrid_search,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
)
from .generation import GenerationSettings, generate_response
from .embeddings import load_faq_index, build_faq_index, save_faq_index


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="User query to search/respond to. Maximum 500 characters."
    )
    
    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v


class QueryResponse(BaseModel):
    answer: str
    semantic_confidence: str = Field(..., description="Confidence in semantic intent understanding (high/medium/low)")
    top_combined_score: float = Field(..., description="Highest combined score from hybrid search (semantic + BM25)")
    results: list
    metadata: dict


class SettingsResponse(BaseModel):
    top_k: int
    semantic_weight: float
    bm25_weight: float
    temperature: float


class SettingsUpdateRequest(BaseModel):
    top_k: Optional[int] = Field(None, ge=1, le=10)
    semantic_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    bm25_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)


app = FastAPI(title="FAQ Assistant", version="0.1.0")
SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "app_settings.json"
settings_manager = SettingsManager(SETTINGS_PATH)

# In-memory query cache (exact match with normalization)
# Future: Replace with gptcache for semantic similarity-based caching
_query_cache: dict[str, QueryResponse] = {}
CACHE_MAX_SIZE = 1000  # Limit cache size to prevent memory issues
_cache_hits = 0
_cache_misses = 0


def normalize_query(query: str) -> str:
    """Normalize query for cache lookup: lowercase, strip, collapse whitespace."""
    normalized = query.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)  # Collapse multiple spaces
    return normalized


def get_from_cache(query: str) -> Optional[QueryResponse]:
    """Get cached response for normalized query."""
    normalized = normalize_query(query)
    return _query_cache.get(normalized)


def put_in_cache(query: str, response: QueryResponse) -> None:
    """Cache response for normalized query with FIFO eviction."""
    normalized = normalize_query(query)
    if len(_query_cache) >= CACHE_MAX_SIZE:
        # Simple FIFO eviction: remove oldest entry
        _query_cache.pop(next(iter(_query_cache)))
    _query_cache[normalized] = response


@lru_cache(maxsize=1)
def get_hybrid_index() -> HybridSearchIndex:
    try:
        faq_index = load_faq_index()
    except FileNotFoundError:
        faq_index = build_faq_index()
        save_faq_index(faq_index)
    return build_hybrid_index(faq_index)


@app.get("/health")
async def health_check():
    index = get_hybrid_index()
    return {
        "status": "ok",
        "documents": len(index.faq_index.documents),
        "model": index.faq_index.model,
        "thresholds": {
            "high": HIGH_CONFIDENCE_THRESHOLD,
            "medium": MEDIUM_CONFIDENCE_THRESHOLD,
        },
        "cache": {
            "size": len(_query_cache),
            "max_size": CACHE_MAX_SIZE,
        },
    }


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    total_requests = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total_requests * 100) if total_requests > 0 else 0.0
    return {
        "size": len(_query_cache),
        "max_size": CACHE_MAX_SIZE,
        "hits": _cache_hits,
        "misses": _cache_misses,
        "hit_rate_percent": round(hit_rate, 2),
    }


@app.post("/cache/clear")
async def clear_cache():
    """Clear the query cache and reset statistics."""
    global _cache_hits, _cache_misses
    _query_cache.clear()
    _cache_hits = 0
    _cache_misses = 0
    return {"status": "cleared", "size": 0, "stats_reset": True}


@app.get("/settings", response_model=SettingsResponse)
async def get_settings():
    return SettingsResponse(**settings_manager.settings.to_dict())


@app.put("/settings", response_model=SettingsResponse)
async def update_settings(request: SettingsUpdateRequest):
    updated = settings_manager.update({k: v for k, v in request.model_dump(exclude_none=True).items()})
    return SettingsResponse(**updated.to_dict())


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a user query and return an answer.
    
    Raises:
        HTTPException 400: Invalid query format
        HTTPException 500: Server configuration error (missing API key)
        HTTPException 502: External service error (OpenAI API failure)
    """
    global _cache_hits, _cache_misses
    
    try:
        # Check cache first
        cached_response = get_from_cache(request.query)
        if cached_response is not None:
            _cache_hits += 1
            # Return copy with updated metadata to avoid mutating cached object
            response_dict = cached_response.model_dump()
            response_dict["metadata"]["cached"] = True
            return QueryResponse(**response_dict)
        
        _cache_misses += 1
        
        # Cache miss - execute full pipeline
        try:
            hybrid_index = get_hybrid_index()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"FAQ index not found: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load FAQ index: {str(e)}"
            )
        
        app_settings = settings_manager.settings
        
        # Perform search
        try:
            search_response = hybrid_search(
                request.query,
                hybrid_index,
                top_k=app_settings.top_k,
                semantic_weight=app_settings.semantic_weight,
                bm25_weight=app_settings.bm25_weight,
            )
        except RuntimeError as e:
            # Embedding API key error
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Configuration error: {str(e)}"
            )
        except Exception as e:
            # OpenAI API errors (rate limit, timeout, etc.)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Search service error: {str(e)}"
            )

        # Generate response
        try:
            gen_settings = GenerationSettings(temperature=app_settings.temperature)
            generation = generate_response(search_response, settings=gen_settings)
        except RuntimeError as e:
            # API key error
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Configuration error: {str(e)}"
            )
        except Exception as e:
            # OpenAI API errors
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Generation service error: {str(e)}"
            )

        results_payload = [
            {
                "question": scored.document.question,
                "answer": scored.document.answer,
                "intent": scored.document.intent,
                "combined_score": scored.score,
                "semantic_score": scored.semantic_score,
                "bm25_score": scored.bm25_score,
                "semantic_confidence": scored.confidence,
            }
            for scored in search_response.results
        ]

        semantic_confidence = "low"
        top_combined_score = 0.0
        if search_response.results:
            semantic_confidence = search_response.results[0].confidence
            top_combined_score = search_response.results[0].score

        response = QueryResponse(
            answer=generation["answer"],
            semantic_confidence=semantic_confidence,
            top_combined_score=top_combined_score,
            results=results_payload,
            metadata={
                "generation": generation["metadata"],
                "settings": app_settings.to_dict(),
                "search": {
                    "weights": search_response.weights,
                    "thresholds": search_response.thresholds,
                    "total_documents": search_response.total_documents,
                },
                "cached": False,
            },
        )
        
        # Store in cache
        put_in_cache(request.query, response)
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Catch-all for unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


