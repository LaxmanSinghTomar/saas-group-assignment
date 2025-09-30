from pathlib import Path
from typing import Generator

import pytest

from evals.evals_retrieval import build_faq_index, save_faq_index, default_index_path, build_hybrid_index, HybridSearchIndex


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session")
def cleaned_faq_path(data_dir: Path) -> Path:
    return data_dir / "processed" / "cleaned_faq_v2.json"


@pytest.fixture(scope="session")
def faq_index(cleaned_faq_path: Path) -> Generator:
    index = build_faq_index(cleaned_path=cleaned_faq_path)
    save_faq_index(index, default_index_path(cleaned_faq_path))
    yield index


@pytest.fixture(scope="session")
def hybrid_index(faq_index) -> HybridSearchIndex:
    return build_hybrid_index(faq_index)


