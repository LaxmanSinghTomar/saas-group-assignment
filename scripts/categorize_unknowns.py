"""Cluster FAQs labeled as unknown/off_topic into proposed categories."""

from __future__ import annotations

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.pipelines import data_cleaning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional path to classified_faq JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output path (auto-versioned otherwise)",
    )
    parser.add_argument(
        "--model",
        default=data_cleaning.DEFAULT_CLASSIFICATION_MODEL,
        help="Model to use for clustering"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature for LLM call"
    )
    return parser.parse_args()


def load_unknown_entries(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    items = payload.get("items", [])
    targets = []
    for item in items:
        if item.get("intent") == data_cleaning.ClassificationIntent.UNKNOWN:
            targets.append({
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "intent": item.get("intent"),
                "quality_score": item.get("quality_score"),
            })
    return targets


def build_prompt(unknown_items: List[Dict[str, str]]) -> str:
    known_intents = [
        f"- {intent['id']}: {intent['description']}"
        for intent in data_cleaning.INTENT_CONFIG["intents"]
    ]
    lines = [
        "You are helping cluster FAQ entries that were labeled as 'unknown' because they did not fit existing intents.",
        "Define sensible new categories for product/support coverage when possible, and flag any true off-topic noise separately.",
        "Return JSON with schema: {\"categories\": [{\"id\": str, \"name\": str, \"description\": str, \"faqs\": [question strings]}], \"off_topic\": [question strings]}.",
        "Guidelines:",
        "- Group FAQs that clearly belong together (e.g., integrations, legal/policies, platform availability).",
        "- Ensure new category IDs are lowercase_with_underscores and distinct from the existing ones listed below.",
        "- Provide concise but descriptive category descriptions.",
        "Existing core intents:",
    ]
    lines.extend(known_intents)
    lines.extend([
        "- Only categorize the provided unknown items; assume off-topic noise has already been filtered out.",
        "- Do not hallucinate answers; just categorize questions.",
        "Items:",
    ])
    for idx, item in enumerate(unknown_items, 1):
        lines.append(f"{idx}. Q: {item['question']} | Intent: {item['intent']} | Quality: {item.get('quality_score')}")
    return "\n".join(lines)


def categorize_unknowns(unknown_items: List[Dict[str, str]], model: str, temperature: float) -> Dict[str, object]:
    if not unknown_items:
        return {"categories": [], "off_topic": []}

    prompt = build_prompt(unknown_items)

    client = data_cleaning.OpenAILLMClassifier(model=model, temperature=temperature)
    base_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert technical writer helping organize FAQs."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "faq_unknown_cluster",
                "schema": {
                    "type": "object",
                    "properties": {
                        "categories": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "faqs": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                },
                                "required": ["name", "description", "faqs"],
                            },
                        },
                        "off_topic": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["categories", "off_topic"],
                    "additionalProperties": False,
                },
            },
        },
    }

    from openai import OpenAI
    api_key = client._api_key  # type: ignore[attr-defined]
    response = OpenAI(api_key=api_key).chat.completions.create(**base_payload)
    message = response.choices[0].message
    if not message.content:
        raise RuntimeError("No content returned for unknown categorization")

    return json.loads(message.content)


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


DATA_DIR = PROJECT_ROOT / "data"
UNKNOWN_BASENAME = "unknown_categorization"


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

    input_path = args.input or (DATA_DIR / "classified_faq_v1.json")
    output_path = args.output or next_version_path(UNKNOWN_BASENAME, DATA_DIR)

    unknown_items = load_unknown_entries(input_path)
    logging.info("Found %s unknown/off_topic items", len(unknown_items))

    result = categorize_unknowns(unknown_items, model=args.model, temperature=args.temperature)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        relative_source = input_path.relative_to(PROJECT_ROOT)
    except ValueError:
        relative_source = input_path
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "source": str(relative_source),
            "model": args.model,
            "temperature": args.temperature,
            "items": unknown_items,
            "clusters": result,
        }, f, ensure_ascii=False, indent=2)

    logging.info("Saved categorization to %s", output_path)


if __name__ == "__main__":
    main()


