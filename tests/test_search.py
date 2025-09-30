from src.search import semantic_search, hybrid_search


def test_semantic_search_basic_structure(hybrid_index):
    response = semantic_search("How do I change my password?", hybrid_index.faq_index, top_k=2)
    assert response.query_embedding.size > 0
    assert response.results
    assert len(response.results) <= 2


def test_hybrid_empty_query_returns_nothing(hybrid_index):
    response = hybrid_search("", hybrid_index, top_k=3)
    assert not response.results
    assert response.query_embedding.size == 0


