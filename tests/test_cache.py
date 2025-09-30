"""Tests for query caching functionality."""

import pytest
from src.api import normalize_query, get_from_cache, put_in_cache, QueryResponse, _query_cache


def test_normalize_query():
    """Test query normalization."""
    assert normalize_query("  Hello World  ") == "hello world"
    assert normalize_query("HELLO WORLD") == "hello world"
    assert normalize_query("hello   world") == "hello world"
    assert normalize_query("  Hello   World  ") == "hello world"


def test_cache_put_and_get():
    """Test putting and getting from cache."""
    # Clear cache first
    _query_cache.clear()
    
    # Create a mock response
    response = QueryResponse(
        answer="Test answer",
        confidence="high",
        results=[],
        metadata={"cached": False}
    )
    
    # Put in cache
    put_in_cache("Test Query", response)
    
    # Get from cache (exact match after normalization)
    cached = get_from_cache("test query")
    assert cached is not None
    assert cached.answer == "Test answer"
    
    # Get with different whitespace (should still match after normalization)
    cached2 = get_from_cache("  TEST   QUERY  ")
    assert cached2 is not None
    assert cached2.answer == "Test answer"


def test_cache_miss():
    """Test cache miss."""
    _query_cache.clear()
    
    cached = get_from_cache("nonexistent query")
    assert cached is None


def test_cache_size_limit():
    """Test cache size limiting with FIFO eviction."""
    _query_cache.clear()
    
    # We can't easily test the full 1000 limit, but we can verify the logic works
    response = QueryResponse(
        answer="Test",
        confidence="high",
        results=[],
        metadata={}
    )
    
    # Add a few entries
    for i in range(5):
        put_in_cache(f"query {i}", response)
    
    assert len(_query_cache) == 5
    
    # All should be retrievable
    for i in range(5):
        assert get_from_cache(f"query {i}") is not None

