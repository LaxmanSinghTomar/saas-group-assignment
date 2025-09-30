"""Embedding utilities and FAQ index helpers for the FAQ assistant."""

from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:  # Optional dependency for local development convenience
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional
    load_dotenv = None

if load_dotenv:  # pragma: no cover - side effect
    load_dotenv()

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_NAME_PATTERN = re.compile(r"cleaned_faq_(v\d+)\.json")

_embedding_client = None


def _get_client():  # pragma: no cover - network helper
    global _embedding_client

    if _embedding_client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is required for embedding generation") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set; cannot generate embeddings")

        _embedding_client = OpenAI(api_key=api_key)

    return _embedding_client


@dataclass
class FAQDocument:
    question: str
    answer: str
    intent: str
    canonical_id: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FAQDocumentIndex:
    """Container holding FAQ documents and their embeddings."""

    documents: List[FAQDocument]
    embeddings: np.ndarray
    model: str
    metadata: Dict[str, Any]
    source_path: Path

    def __post_init__(self) -> None:
        self._normalized_embeddings = self._compute_normalized(self.embeddings)

    @staticmethod
    def _compute_normalized(matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-12, None)

    @property
    def normalized_embeddings(self) -> np.ndarray:
        return self._normalized_embeddings


@dataclass
class EmbeddingResult:
    embeddings: np.ndarray
    model: str


def embed_texts(texts: Iterable[str], model: str = DEFAULT_EMBEDDING_MODEL) -> EmbeddingResult:
    """Generate embeddings for the provided texts."""

    texts_list: List[str] = [text if text else "" for text in texts]
    if not texts_list:
        return EmbeddingResult(embeddings=np.zeros((0, 0)), model=model)

    client = _get_client()
    response = client.embeddings.create(model=model, input=texts_list)

    vectors = [item.embedding for item in response.data]
    return EmbeddingResult(embeddings=np.array(vectors, dtype=np.float32), model=model)


def load_cleaned_faq(path: Path | None = None) -> Tuple[List[FAQDocument], Dict[str, Any]]:
    """Load canonical FAQ entries from the cleaned dataset."""

    target = path or (DATA_DIR / "cleaned_faq_v2.json")
    with target.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    documents: List[FAQDocument] = []
    for item in payload.get("items", []):
        meta: Dict[str, Any] = {
            "quality_score": item.get("quality_score"),
            "issues": item.get("issues", []),
            "duplicates": item.get("duplicates", []),
            "reasoning": item.get("reasoning"),
        }
        documents.append(
            FAQDocument(
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                intent=item.get("intent", "unknown"),
                canonical_id=item.get("canonical_id"),
                metadata=meta,
            )
        )

    metadata = payload.get("metadata", {})
    return documents, metadata


def build_faq_index(
    cleaned_path: Path | None = None,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> FAQDocumentIndex:
    """Load cleaned FAQs and embed their questions to build a searchable index."""

    documents, metadata = load_cleaned_faq(cleaned_path)
    embedding_result = embed_texts((doc.question for doc in documents), model=embedding_model)

    source = cleaned_path or (DATA_DIR / "cleaned_faq_v2.json")
    return FAQDocumentIndex(
        documents=documents,
        embeddings=embedding_result.embeddings,
        model=embedding_result.model,
        metadata=metadata,
        source_path=Path(source).resolve(),
    )


def default_index_path(cleaned_path: Path | None = None) -> Path:
    target = Path(cleaned_path) if cleaned_path else (DATA_DIR / "cleaned_faq_v2.json")
    match = INDEX_NAME_PATTERN.match(target.name)
    suffix = match.group(1) if match else "latest"
    return target.with_name(f"faq_index_{suffix}.pkl")


def save_faq_index(index: FAQDocumentIndex, path: Path | None = None) -> Path:
    target = Path(path) if path else default_index_path(index.source_path)
    payload = {
        "documents": [
            {
                "question": doc.question,
                "answer": doc.answer,
                "intent": doc.intent,
                "canonical_id": doc.canonical_id,
                "metadata": doc.metadata,
            }
            for doc in index.documents
        ],
        "embeddings": index.embeddings.astype(np.float32),
        "model": index.model,
        "metadata": index.metadata,
        "source_path": str(index.source_path),
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        pickle.dump(payload, f)
    return target.resolve()


def load_faq_index(path: Path | None = None) -> FAQDocumentIndex:
    target = Path(path) if path else default_index_path()
    with target.open("rb") as f:
        payload = pickle.load(f)

    documents = [
        FAQDocument(
            question=item.get("question", ""),
            answer=item.get("answer", ""),
            intent=item.get("intent", "unknown"),
            canonical_id=item.get("canonical_id"),
            metadata=item.get("metadata", {}),
        )
        for item in payload["documents"]
    ]

    embeddings = payload["embeddings"]
    model = payload["model"]
    metadata = payload.get("metadata", {})
    source_path = Path(payload.get("source_path", DATA_DIR / "cleaned_faq_v2.json"))

    return FAQDocumentIndex(
        documents=documents,
        embeddings=embeddings,
        model=model,
        metadata=metadata,
        source_path=source_path,
    )


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "FAQDocument",
    "FAQDocumentIndex",
    "EmbeddingResult",
    "embed_texts",
    "load_cleaned_faq",
    "build_faq_index",
    "default_index_path",
    "save_faq_index",
    "load_faq_index",
]


