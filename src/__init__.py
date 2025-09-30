"""Core exports for the FAQ assistant project."""

from .pipelines.data_cleaning import (
    RawFAQ,
    ClassifiedFAQ,
    ClassificationIntent,
    FAQClassifier,
    OpenAILLMClassifier,
    classify_faqs,
)

from .embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    FAQDocument,
    FAQDocumentIndex,
    EmbeddingResult,
    embed_texts,
    load_cleaned_faq,
    build_faq_index,
    default_index_path,
    save_faq_index,
    load_faq_index,
)

from .search import (
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    DEFAULT_TOP_K,
    SEMANTIC_WEIGHT,
    BM25_WEIGHT,
    ScoredFAQ,
    SearchResponse,
    semantic_search,
    ensure_index,
    HybridSearchIndex,
    build_hybrid_index,
    hybrid_search,
)

from .generation import (
    DEFAULT_COMPLETION_MODEL,
    GenerationSettings,
    generate_response,
)

from .api import app

__all__ = [
    "RawFAQ",
    "ClassifiedFAQ",
    "ClassificationIntent",
    "FAQClassifier",
    "OpenAILLMClassifier",
    "classify_faqs",
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
    "DEFAULT_COMPLETION_MODEL",
    "GenerationSettings",
    "generate_response",
    "app",
]
