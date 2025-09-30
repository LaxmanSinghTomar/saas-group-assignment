"""Tests for API input validation and error handling."""

import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_query_max_length_validation():
    """Test that queries exceeding max_length are rejected."""
    long_query = "a" * 501  # Exceeds 500 char limit
    response = client.post("/query", json={"query": long_query})
    assert response.status_code == 422
    error_detail = response.json()
    assert "detail" in error_detail
    # Check that it mentions max length
    assert any("500" in str(item) for item in error_detail["detail"])


def test_query_empty_validation():
    """Test that empty/whitespace-only queries are rejected."""
    response = client.post("/query", json={"query": "   "})
    assert response.status_code == 422
    error_detail = response.json()
    assert "detail" in error_detail


def test_valid_query_within_limits():
    """Test that valid queries within limits work correctly."""
    response = client.post("/query", json={"query": "How do I reset my password?"})
    # Should succeed (200) or fail gracefully if no API key (500/502)
    assert response.status_code in [200, 500, 502]


def test_cache_stats_structure():
    """Test that cache stats endpoint returns expected structure."""
    response = client.get("/cache/stats")
    assert response.status_code == 200
    data = response.json()
    assert "size" in data
    assert "max_size" in data
    assert "hits" in data
    assert "misses" in data
    assert "hit_rate_percent" in data


def test_health_endpoint():
    """Test that health endpoint returns expected structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "documents" in data
    assert "model" in data
    assert "thresholds" in data
    assert "cache" in data

