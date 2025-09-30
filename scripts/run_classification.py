"""Run the LLM-based classification over the raw FAQ dataset."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import re
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.pipelines.data_cleaning import (
    DEFAULT_CLASSIFICATION_MODEL,
    OpenAILLMClassifier,
    classify_faqs,
    export_classifications,
    load_raw_faq,
    reload_intent_config,
)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
BASE_NAME = "classified_faq"
INTENT_ENV_VAR = "INTENT_CONFIG_PATH"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_CLASSIFICATION_MODEL,
        help="OpenAI model name to use for classification",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output path (otherwise auto-versioned)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit of FAQ entries to classify (for quick smoke tests)",
    )
    parser.add_argument(
        "--intent-config",
        type=Path,
        default=None,
        help="Optional explicit path to intent config (overrides auto-detected)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for the classification model",
    )
    parser.add_argument(
        "--reuse-from",
        type=Path,
        default=None,
        help="Path to an existing classified_faq file to reuse instead of re-running classification",
    )
    return parser.parse_args()


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
        reload_intent_config(args.intent_config)

    if args.reuse_from:
        output_path = args.output or next_version_path(BASE_NAME, DATA_DIR)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(args.reuse_from.read_bytes())
        logging.info("Reused existing classifications from %s -> %s", args.reuse_from, output_path)
        return

    classifier = OpenAILLMClassifier(model=args.model, temperature=args.temperature)
    raw_faqs = load_raw_faq()

    if args.limit is not None:
        raw_faqs = raw_faqs[: args.limit]

    classified = classify_faqs(raw_faqs, classifier)

    output_path = args.output or next_version_path(BASE_NAME, DATA_DIR)
    export_classifications(
        classified,
        path=output_path,
        metadata={"model": args.model, "total_raw": len(raw_faqs)},
    )

    logging.info("Saved classified FAQs to %s", output_path)


if __name__ == "__main__":
    main()


