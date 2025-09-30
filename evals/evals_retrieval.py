"""Evaluation utilities for FAQ retrieval quality."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from datetime import datetime, UTC

from src.search import HybridSearchIndex, build_hybrid_index, hybrid_search
from src.embeddings import load_faq_index, build_faq_index, save_faq_index, default_index_path

import argparse


@dataclass
class EvaluationResult:
    query_id: str
    query: str
    expected_question: str
    retrieved_questions: List[str]
    scores: List[float]
    hit_rank: int | None


def load_gold_queries(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_hybrid_index() -> HybridSearchIndex:
    try:
        faq_index = load_faq_index()
    except FileNotFoundError:
        faq_index = build_faq_index()
        save_faq_index(faq_index, default_index_path())
    return build_hybrid_index(faq_index)


def evaluate_queries(
    queries: Iterable[dict],
    hybrid_index: HybridSearchIndex,
    *,
    top_k: int = 5,
):
    results: List[EvaluationResult] = []
    for item in queries:
        query = item["query"]
        expected = item["expected_question"].lower()
        response = hybrid_search(query, hybrid_index, top_k=top_k)
        scores = [result.score for result in response.results]
        retrieved = [result.document.question for result in response.results]
        hit_rank = None
        for idx, question in enumerate(retrieved, start=1):
            if question.lower() == expected:
                hit_rank = idx
                break
        results.append(
            EvaluationResult(
                query_id=item.get("id", query),
                query=query,
                expected_question=item["expected_question"],
                retrieved_questions=retrieved,
                scores=scores,
                hit_rank=hit_rank,
            )
        )
    return results


def mean_reciprocal_rank(results: Iterable[EvaluationResult]) -> float:
    rr = [1.0 / res.hit_rank for res in results if res.hit_rank]
    return float(np.mean(rr)) if rr else 0.0


def ndcg_at_k(result: EvaluationResult, k: int = 5) -> float:
    if not result.hit_rank or result.hit_rank > k:
        return 0.0
    dcg = 1.0 / np.log2(result.hit_rank + 1)
    idcg = 1.0 / np.log2(2)
    return dcg / idcg


def average_ndcg(results: Iterable[EvaluationResult], k: int = 5) -> float:
    ndcgs = [ndcg_at_k(res, k=k) for res in results]
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def export_results(results: Iterable[EvaluationResult], path: Path) -> None:
    payload = [
        {
            "query_id": res.query_id,
            "query": res.query,
            "expected_question": res.expected_question,
            "retrieved_questions": res.retrieved_questions,
            "scores": res.scores,
            "hit_rank": res.hit_rank,
        }
        for res in results
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"created_at": datetime.now(UTC).isoformat(), "results": payload}, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FAQ retrieval performance.")
    parser.add_argument("--gold", type=Path, default=Path("data/evals/gold_queries.json"), help="Path to the gold query set.")
    parser.add_argument("--output", type=Path, default=Path("data/evals/eval_results.json"), help="Path to write detailed evaluation results.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to consider per query.")
    parser.add_argument("--print-metrics", action="store_true", help="Print summary metrics to stdout.")
    args = parser.parse_args()

    queries = load_gold_queries(args.gold)
    hybrid_index = ensure_hybrid_index()
    results = evaluate_queries(queries, hybrid_index, top_k=args.top_k)
    export_results(results, args.output)

    if args.print_metrics:
        total = len(results)
        top1_hits = sum(1 for res in results if res.hit_rank == 1)
        mrr_at_1 = top1_hits / total if total else 0.0
        print(f"Queries evaluated: {total}")
        print(f"MRR@1 (top-1 accuracy): {mrr_at_1:.3f}")


if __name__ == "__main__":
    main()


