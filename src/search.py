"""Search utilities (semantic + BM25 hybrid) for the FAQ assistant."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    FAQDocument,
    FAQDocumentIndex,
    build_faq_index,
    embed_texts,
)

HIGH_CONFIDENCE_THRESHOLD = 0.85
MEDIUM_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_TOP_K = 5
SEMANTIC_WEIGHT = 0.7
BM25_WEIGHT = 0.3

_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


@dataclass
class ScoredFAQ:
    """FAQ document paired with its semantic similarity score."""

    document: FAQDocument
    score: float
    rank: int
    semantic_score: float
    bm25_score: float

    @property
    def confidence(self) -> str:
        """Semantic intent understanding confidence (diagnostic metadata).
        
        Indicates how well the query semantically matches the FAQ question:
        - high (≥0.85): Strong semantic match, clear user intent
        - medium (≥0.65): Moderate match, some ambiguity
        - low (<0.65): Weak semantic match, keyword query or unclear intent
        
        Note: This is diagnostic metadata about semantic similarity, independent
        of the combined_score (which includes BM25 keyword matching). The LLM
        generation layer makes final response decisions based on full context,
        not by checking this field.
        
        Example: Query "password"
        - combined_score: 0.92 (high) - BM25 found keyword matches
        - semantic_score: 0.38 (low) - Intent unclear
        - confidence: "low" - Reflects semantic ambiguity
        - LLM response: Clarification list (decided from context)
        """
        if self.semantic_score >= HIGH_CONFIDENCE_THRESHOLD:
            return "high"
        if self.semantic_score >= MEDIUM_CONFIDENCE_THRESHOLD:
            return "medium"
        return "low"


@dataclass
class SearchResponse:
    """Results returned from a semantic FAQ lookup."""

    query: str
    query_embedding: np.ndarray
    results: List[ScoredFAQ]
    total_documents: int
    model: str
    thresholds: dict
    weights: Optional[dict] = None


@dataclass
class HybridSearchIndex:
    faq_index: FAQDocumentIndex
    bm25: BM25Okapi
    tokenized_questions: List[List[str]]


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _similarities(query_vector: np.ndarray, normalized_matrix: np.ndarray) -> np.ndarray:
    if normalized_matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return normalized_matrix @ query_vector


def _tokenize(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _semantic_scores(
    query: str,
    index: FAQDocumentIndex,
    *,
    embedding_model: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    model_to_use = embedding_model or index.model or DEFAULT_EMBEDDING_MODEL
    embedding_result = embed_texts([query], model=model_to_use)
    query_embedding = embedding_result.embeddings.reshape(-1)
    normalized_query = _normalize(query_embedding)
    similarity_scores = _similarities(normalized_query, index.normalized_embeddings)
    return query_embedding, similarity_scores, embedding_result.model


def semantic_search(
    query: str,
    index: FAQDocumentIndex,
    *,
    top_k: int = DEFAULT_TOP_K,
    embedding_model: str | None = None,
) -> SearchResponse:
    """Embed the query and score against the provided FAQ index."""

    if not query.strip():
        return SearchResponse(
            query=query,
            query_embedding=np.zeros((0,), dtype=np.float32),
            results=[],
            total_documents=len(index.documents),
            model=index.model,
            thresholds={
                "high": HIGH_CONFIDENCE_THRESHOLD,
                "medium": MEDIUM_CONFIDENCE_THRESHOLD,
            },
            weights=None,
        )

    query_embedding, similarity_scores, model_used = _semantic_scores(
        query, index, embedding_model=embedding_model
    )
    ranked_indices = np.argsort(similarity_scores)[::-1]

    top_indices = ranked_indices[: max(0, top_k)] if top_k else ranked_indices

    results: List[ScoredFAQ] = []
    for rank, doc_index in enumerate(top_indices, start=1):
        score = float(similarity_scores[doc_index])
        document = index.documents[doc_index]
        results.append(
            ScoredFAQ(
                document=document,
                score=score,
                rank=rank,
                semantic_score=score,
                bm25_score=0.0,
            )
        )

    return SearchResponse(
        query=query,
        query_embedding=query_embedding,
        results=results,
        total_documents=len(index.documents),
        model=model_used,
        thresholds={
            "high": HIGH_CONFIDENCE_THRESHOLD,
            "medium": MEDIUM_CONFIDENCE_THRESHOLD,
        },
        weights=None,
    )


def ensure_index(
    index: FAQDocumentIndex | None,
    *,
    cleaned_path=None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> FAQDocumentIndex:
    """Return an FAQ index, building one if necessary."""

    if index is not None:
        return index
    return build_faq_index(cleaned_path=cleaned_path, embedding_model=embedding_model)


def build_hybrid_index(
    faq_index: FAQDocumentIndex | None = None,
    *,
    cleaned_path=None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> HybridSearchIndex:
    base_index = faq_index or build_faq_index(cleaned_path=cleaned_path, embedding_model=embedding_model)
    tokenized = [_tokenize(doc.question) for doc in base_index.documents]
    bm25 = BM25Okapi(tokenized)
    return HybridSearchIndex(faq_index=base_index, bm25=bm25, tokenized_questions=tokenized)


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    max_val = values.max()
    if max_val <= 0:
        return np.zeros_like(values)
    return values / max_val


def hybrid_search(
    query: str,
    hybrid_index: HybridSearchIndex,
    *,
    top_k: int = DEFAULT_TOP_K,
    semantic_weight: float = SEMANTIC_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    embedding_model: str | None = None,
) -> SearchResponse:
    if not query.strip():
        return SearchResponse(
            query=query,
            query_embedding=np.zeros((0,), dtype=np.float32),
            results=[],
            total_documents=len(hybrid_index.faq_index.documents),
            model=hybrid_index.faq_index.model,
            thresholds={
                "high": HIGH_CONFIDENCE_THRESHOLD,
                "medium": MEDIUM_CONFIDENCE_THRESHOLD,
            },
            weights={"semantic": semantic_weight, "bm25": bm25_weight},
        )

    query_embedding, semantic_scores, model_used = _semantic_scores(
        query, hybrid_index.faq_index, embedding_model=embedding_model
    )

    semantic_scores = np.clip(semantic_scores, 0.0, None)
    semantic_norm = _normalize_scores(semantic_scores.copy())

    query_tokens = _tokenize(query)
    if query_tokens:
        bm25_scores = np.array(hybrid_index.bm25.get_scores(query_tokens), dtype=np.float32)
    else:
        bm25_scores = np.zeros(len(hybrid_index.tokenized_questions), dtype=np.float32)
    bm25_norm = _normalize_scores(bm25_scores.copy())

    combined = semantic_weight * semantic_norm + bm25_weight * bm25_norm
    ranked_indices = np.argsort(combined)[::-1]
    top_indices = ranked_indices[: max(0, top_k)] if top_k else ranked_indices

    results: List[ScoredFAQ] = []
    for rank, doc_index in enumerate(top_indices, start=1):
        results.append(
            ScoredFAQ(
                document=hybrid_index.faq_index.documents[doc_index],
                score=float(combined[doc_index]),
                rank=rank,
                semantic_score=float(semantic_scores[doc_index]),
                bm25_score=float(bm25_norm[doc_index]),  # Use normalized score for consistency
            )
        )

    return SearchResponse(
        query=query,
        query_embedding=query_embedding,
        results=results,
        total_documents=len(hybrid_index.faq_index.documents),
        model=model_used,
        thresholds={
            "high": HIGH_CONFIDENCE_THRESHOLD,
            "medium": MEDIUM_CONFIDENCE_THRESHOLD,
        },
        weights={"semantic": semantic_weight, "bm25": bm25_weight},
    )


__all__ = [
    "HIGH_CONFIDENCE_THRESHOLD",
    "MEDIUM_CONFIDENCE_THRESHOLD",
    "DEFAULT_TOP_K",
    "SEMANTIC_WEIGHT",
    "BM25_WEIGHT",
    "ScoredFAQ",
    "SearchResponse",
    "semantic_search",
    "ensure_index",
    "HybridSearchIndex",
    "build_hybrid_index",
    "hybrid_search",
]


