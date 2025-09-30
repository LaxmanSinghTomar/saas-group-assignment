"""Semantic deduplication for classified FAQs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIED_PATH = PROJECT_ROOT / "data" / "processed" / "classified_faq_v1.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_faq_v1.json"

SIMILARITY_THRESHOLD = 0.85


@dataclass
class FAQItem:
    question: str
    answer: str
    intent: str
    quality_score: float
    issues: List[str] = field(default_factory=list)
    reasoning: str | None = None
    canonical_id: str | None = None
    duplicates: List[str] = field(default_factory=list)


def load_classified(path: Path = CLASSIFIED_PATH) -> Tuple[List[FAQItem], Dict[str, str]]:
    import json

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    metadata = payload.get("metadata", {})
    items = [FAQItem(**item) for item in payload["items"]]
    return items, metadata


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros((0, 0))

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.clip(norms, 1e-12, None)
    return normalized @ normalized.T


def cluster_duplicates(embeddings: np.ndarray, threshold: float = SIMILARITY_THRESHOLD) -> List[List[int]]:
    sim_matrix = cosine_similarity_matrix(embeddings)
    n = sim_matrix.shape[0]
    visited = [False] * n
    clusters: List[List[int]] = []

    for idx in range(n):
        if visited[idx]:
            continue
        cluster = [idx]
        visited[idx] = True
        for jdx in range(idx + 1, n):
            if sim_matrix[idx, jdx] >= threshold:
                visited[jdx] = True
                cluster.append(jdx)
        clusters.append(cluster)

    return clusters


def select_representative(cluster: List[FAQItem]) -> FAQItem:
    sorted_items = sorted(
        cluster,
        key=lambda item: (
            -item.quality_score,
            -len(item.question),
        ),
    )
    return sorted_items[0]


def deduplicate_faq(
    items: List[FAQItem],
    embeddings: np.ndarray,
    threshold: float = SIMILARITY_THRESHOLD,
) -> Tuple[List[FAQItem], List[Dict[str, str]]]:
    clusters_idx = cluster_duplicates(embeddings, threshold=threshold)
    cleaned: List[FAQItem] = []
    audit: List[Dict[str, str]] = []

    for cluster in clusters_idx:
        grouped = [items[i] for i in cluster]
        canonical = select_representative(grouped)

        duplicates = [items[i] for i in cluster if items[i] is not canonical]
        canonical.canonical_id = canonical.canonical_id or canonical.question.lower().replace(" ", "_")
        canonical_duplicate_questions = [dup.question for dup in duplicates]

        cleaned.append(
            FAQItem(
                question=canonical.question,
                answer=canonical.answer,
                intent=canonical.intent,
                quality_score=canonical.quality_score,
                issues=canonical.issues,
                reasoning=canonical.reasoning,
                canonical_id=canonical.canonical_id,
                duplicates=canonical_duplicate_questions,
            )
        )

        audit.append(
            {
                "canonical": canonical.question,
                "duplicates": canonical_duplicate_questions,
            }
        )

    return cleaned, audit


def top_similar_pairs(
    items: List[FAQItem],
    embeddings: np.ndarray,
    top_k: int = 10,
) -> List[Tuple[float, str, str]]:
    if embeddings.size == 0:
        return []

    sim_matrix = cosine_similarity_matrix(embeddings)
    n = sim_matrix.shape[0]
    pairs: List[Tuple[float, str, str]] = []

    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((sim_matrix[i, j], items[i].question, items[j].question))

    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs[:top_k]


def save_cleaned(
    items: List[FAQItem],
    metadata: Dict[str, str],
    audit: List[Dict[str, str]],
    path: Path = OUTPUT_PATH,
) -> None:
    import json
    from datetime import datetime, timezone

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "items": [item.__dict__ for item in items],
        "audit": audit,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


__all__ = [
    "CLASSIFIED_PATH",
    "OUTPUT_PATH",
    "SIMILARITY_THRESHOLD",
    "FAQItem",
    "load_classified",
    "deduplicate_faq",
    "save_cleaned",
    "cluster_duplicates",
]


