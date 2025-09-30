"""Run deduplication over classified FAQs and produce cleaned dataset."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.pipelines import data_cleaning
from src import embeddings
from src.pipelines import deduplication


DATA_DIR = PROJECT_ROOT / "data"
CLASSIFIED_BASENAME = "classified_faq"
CLEANED_BASENAME = "cleaned_faq"
INTENT_ENV_VAR = "INTENT_CONFIG_PATH"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional explicit path to classified_faq JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit path for cleaned FAQ output",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=deduplication.SIMILARITY_THRESHOLD,
        help="Cosine similarity threshold for duplicates",
    )
    parser.add_argument(
        "--model",
        default=embeddings.DEFAULT_EMBEDDING_MODEL,
        help="Embedding model to use",
    )
    parser.add_argument(
        "--intents",
        nargs="*",
        default=None,
        help="Optional list of intents to include (others will be excluded)",
    )
    parser.add_argument(
        "--exclude-intents",
        nargs="*",
        default=[data_cleaning.ClassificationIntent.OFF_TOPIC],
        help="Intents to drop before deduplication",
    )
    parser.add_argument(
        "--report-top",
        type=int,
        default=10,
        help="Number of top similar question pairs to log for debugging",
    )
    parser.add_argument(
        "--intent-config",
        type=Path,
        default=None,
        help="Optional explicit path to intent config (overrides auto-detected)",
    )
    return parser.parse_args()


def filter_by_intents(items, include=None, exclude=None):
    include_set = set(include) if include else None
    exclude_set = set(exclude) if exclude else set()

    filtered = []
    for item in items:
        if include_set and item.intent not in include_set:
            continue
        if item.intent in exclude_set:
            continue
        filtered.append(item)
    return filtered


def latest_version_path(base_name: str, directory: Path) -> Path:
    pattern = re.compile(rf"{re.escape(base_name)}_v(\\d+)\\.json")
    latest = None
    max_version = -1
    for path in directory.glob(f"{base_name}_v*.json"):
        match = pattern.match(path.name)
        if match:
            version = int(match.group(1))
            if version > max_version:
                max_version = version
                latest = path
    if latest is None:
        raise FileNotFoundError(f"No versioned file found for {base_name} in {directory}")
    return latest


def next_version_path(base_name: str, directory: Path) -> Path:
    pattern = re.compile(rf"{re.escape(base_name)}_v(\\d+)\\.json")
    max_version = 0
    for path in directory.glob(f"{base_name}_v*.json"):
        match = pattern.match(path.name)
        if match:
            max_version = max(max_version, int(match.group(1)))
    return directory / f"{base_name}_v{max_version + 1}.json"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_args()
    if args.intent_config:
        os.environ[INTENT_ENV_VAR] = str(args.intent_config)

    input_path = args.input or latest_version_path(CLASSIFIED_BASENAME, DATA_DIR)
    output_path = args.output or next_version_path(CLEANED_BASENAME, DATA_DIR)

    items, metadata = deduplication.load_classified(input_path)
    items = filter_by_intents(items, include=args.intents, exclude=args.exclude_intents)

    logging.info("Embedding %s FAQ items", len(items))
    embed_result = embeddings.embed_texts([item.question for item in items], model=args.model)

    metadata = dict(metadata)
    metadata.update(
        {
            "embedding_model": embed_result.model,
            "embedding_dimension": embed_result.embeddings.shape[1] if embed_result.embeddings.size else 0,
            "dedup_similarity_threshold": args.threshold,
            "filtered_count": len(items),
            "total_raw": metadata.get("total_raw"),
        }
    )

    cleaned, audit = deduplication.deduplicate_faq(items, embed_result.embeddings, threshold=args.threshold)

    top_pairs = deduplication.top_similar_pairs(items, embed_result.embeddings, top_k=args.report_top)
    for score, q1, q2 in top_pairs:
        logging.info("Similarity %.4f | %s | %s", score, q1, q2)

    logging.info("Saving cleaned dataset with %s items", len(cleaned))
    deduplication.save_cleaned(cleaned, metadata=metadata, audit=audit, path=output_path)


if __name__ == "__main__":
    main()


