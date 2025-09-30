import json
from pathlib import Path

from evals.evals_retrieval import (
    ensure_hybrid_index,
    evaluate_queries,
    mean_reciprocal_rank,
    average_ndcg,
)


def load_gold():
    path = Path("data/evals/gold_queries.json")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_retrieval_metrics():
    hybrid_index = ensure_hybrid_index()
    queries = load_gold()
    results = evaluate_queries(queries, hybrid_index, top_k=5)

    assert results
    mrr = mean_reciprocal_rank(results)
    ndcg = average_ndcg(results, k=5)

    assert 0.0 <= mrr <= 1.0
    assert 0.0 <= ndcg <= 1.0
