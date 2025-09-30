"""Data cleaning pipeline for the FAQ assistant.

This module handles:
- Loading the raw FAQ dataset from `data/raw_faq.json`.
- Preparing prompts/payloads for LLM-based classification (intent + quality).
- Filtering malformed/off-topic entries based on classification metadata.
- Semantic deduplication placeholder (actual embedding logic lives in `embeddings.py`).

The current focus is step 1 of the plan: define the data structures and attach a
classification interface that we can back with the OpenAI API (or a mock for testing).
"""

from __future__ import annotations

import json
import os
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set
import json

# Optional runtime dependency for loading environment variables.
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv:  # pragma: no cover - side effect with external file
    load_dotenv()

# Lazy import; only load OpenAI client if needed to avoid hard dependency
_openai_client = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "raw_faq.json"
# Default model chosen for the provided project key; smaller GPT-5 variant
# balances quality and cost for classification.
DEFAULT_CLASSIFICATION_MODEL = "gpt-4.1-mini"
CLASSIFIED_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "classified_faq_v1.json"


class ClassificationIntent:
    """Enum-like container for supported intents."""

    ACCOUNT_MANAGEMENT = "account_management"
    SECURITY = "security"
    SETTINGS = "settings"
    DATA_PRIVACY = "data_privacy"
    UNKNOWN = "unknown"
    OFF_TOPIC = "off_topic"  # Represents malformed / junk entries


CONFIG_DIR = PROJECT_ROOT / "data" / "configs"
CONFIG_ENV_VAR = "INTENT_CONFIG_PATH"


def _latest_intent_config(path: Path = CONFIG_DIR) -> Path:
    configs = sorted(path.glob("intent_config_v*.json"))
    if not configs:
        raise FileNotFoundError("No intent_config_v*.json found in data directory")
    return configs[-1]


CONFIG_PATH = Path(os.getenv(CONFIG_ENV_VAR)).resolve() if os.getenv(CONFIG_ENV_VAR) else _latest_intent_config()


def load_intent_config(path: Path | None = None) -> Dict[str, Any]:
    target = path or CONFIG_PATH
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


INTENT_CONFIG: Dict[str, Any]
INTENT_CHOICES: Set[str]
INTENT_ORDER: List[str]
INTENT_DESCRIPTIONS: Dict[str, str]


def _apply_intent_config(config: Dict[str, Any], path: Path) -> None:
    global CONFIG_PATH, INTENT_CONFIG, INTENT_CHOICES, INTENT_ORDER, INTENT_DESCRIPTIONS

    CONFIG_PATH = path
    INTENT_CONFIG = config

    base_intents = [intent["id"] for intent in config["intents"]]
    unknown_id = config["special_intents"]["unknown"]["id"]
    off_topic_id = config["special_intents"]["off_topic"]["id"]

    INTENT_ORDER = base_intents + [unknown_id, off_topic_id]

    INTENT_DESCRIPTIONS = {
        intent["id"]: intent["description"] for intent in config["intents"]
    }
    INTENT_DESCRIPTIONS[unknown_id] = config["special_intents"]["unknown"]["description"]
    INTENT_DESCRIPTIONS[off_topic_id] = config["special_intents"]["off_topic"]["description"]

    intent_choices = set(INTENT_ORDER)
    intent_choices.update(
        {
            ClassificationIntent.ACCOUNT_MANAGEMENT,
            ClassificationIntent.SECURITY,
            ClassificationIntent.SETTINGS,
            ClassificationIntent.DATA_PRIVACY,
            ClassificationIntent.UNKNOWN,
            ClassificationIntent.OFF_TOPIC,
        }
    )
    INTENT_CHOICES = intent_choices


def reload_intent_config(path: Path) -> None:
    resolved = path.resolve()
    config = load_intent_config(resolved)
    _apply_intent_config(config, resolved)


_apply_intent_config(load_intent_config(CONFIG_PATH), CONFIG_PATH)


@dataclass
class RawFAQ:
    question: str
    answer: str

    def stripped_question(self) -> str:
        return self.question.strip()


@dataclass
class ClassifiedFAQ(RawFAQ):
    intent: str
    quality_score: float
    issues: List[str] = field(default_factory=list)
    reasoning: Optional[str] = None

    def is_relevant(self) -> bool:
        return self.intent != ClassificationIntent.OFF_TOPIC

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "intent": self.intent,
            "quality_score": self.quality_score,
            "issues": self.issues,
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# Data loading


def load_raw_faq(path: Path = RAW_DATA_PATH) -> List[RawFAQ]:
    """Load raw FAQ entries from disk."""

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [RawFAQ(**entry) for entry in data]


# ---------------------------------------------------------------------------
# Classification interface


class FAQClassifier:
    """Abstraction for classifying FAQ entries using an LLM."""

    def classify(self, faq: RawFAQ) -> ClassifiedFAQ:
        raise NotImplementedError


class OpenAILLMClassifier(FAQClassifier):
    """Implementation skeleton that will call OpenAI's API.

    The actual API call is intentionally deferred; this skeleton focuses on
    building the prompt payload so we can test the pipeline without network
    dependency.
    """

    def __init__(
        self,
        model: str = DEFAULT_CLASSIFICATION_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature

        if self._api_key is None:
            raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY in environment or .env")

    def build_prompt(self, faq: RawFAQ) -> Dict[str, Any]:
        """Return the messages payload for the chat completion request."""

        system = (
            "You are a helpful assistant that labels FAQ entries with intent and "
            "quality information."
        )
        intent_lines = [
            f"- {intent}: {INTENT_DESCRIPTIONS[intent]}"
            for intent in INTENT_ORDER
        ]
        intent_guidance = "\n".join(intent_lines)
        user = (
            "Classify the FAQ entry into exactly one of the intents: "
            f"{', '.join(INTENT_ORDER)}.\n"
            "Intent descriptions:\n"
            f"{intent_guidance}\n\n"
            "Prioritize the QUESTION text when determining the intent; use the "
            "answer only to validate whether the Q/A pair makes sense.\n"
            "Return JSON with fields intent, quality_score (0-1), issues (list), "
            "reasoning. If the question and answer clearly do not match, set the "
            "intent to 'malformed' and describe the mismatch in issues.\n" \
            f"Question: {faq.question!r}\nAnswer: {faq.answer!r}"
        )
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "faq_classification",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "intent": {"type": "string"},
                            "quality_score": {"type": "number"},
                            "issues": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "reasoning": {"type": "string"},
                        },
                        "required": ["intent", "quality_score", "issues"],
                        "additionalProperties": False,
                    },
                },
            },
        }

    def classify(self, faq: RawFAQ) -> ClassifiedFAQ:  # pragma: no cover - network call
        global _openai_client  # avoid repeated imports to keep things fast in loops

        if _openai_client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("openai package is required for OpenAILLMClassifier") from exc

            _openai_client = OpenAI(api_key=self._api_key)

        payload = self.build_prompt(faq)
        payload["temperature"] = self.temperature
        try:
            response = _openai_client.chat.completions.create(**payload)
            message = response.choices[0].message
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.exception("OpenAI classification failed: %s", exc)
            raise

        if not message.content:
            raise RuntimeError("OpenAI response had no content")

        result = json.loads(message.content)

        intent = result["intent"].strip().lower()
        if intent not in INTENT_CHOICES:
            intent = ClassificationIntent.UNKNOWN

        return ClassifiedFAQ(
            question=faq.question,
            answer=faq.answer,
            intent=intent,
            quality_score=float(result.get("quality_score", 0.0)),
            issues=list(result.get("issues", [])),
            reasoning=result.get("reasoning"),
        )


def classify_faqs(
    faqs: Iterable[RawFAQ],
    classifier: FAQClassifier,
) -> List[ClassifiedFAQ]:
    """Apply a classifier to a collection of FAQs."""

    results: List[ClassifiedFAQ] = []
    start_time = time.perf_counter()

    for idx, entry in enumerate(faqs, start=1):
        if not entry.stripped_question():
            logger.info("Skipping FAQ with empty question")
            continue

        classified = classifier.classify(entry)
        results.append(classified)

        if idx % 5 == 0:
            logger.info("Classified %s FAQ entries", idx)

    elapsed = time.perf_counter() - start_time
    logger.info("Classification finished: %s entries in %.2fs", len(results), elapsed)
    return results


def export_classifications(
    faqs: List[ClassifiedFAQ],
    path: Path = CLASSIFIED_OUTPUT_PATH,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist classified FAQs and metadata for downstream steps."""

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "items": [faq.to_dict() for faq in faqs],
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)



# ---------------------------------------------------------------------------
# Utility helpers for caching/invalidation (stubs for now)


def build_cache_metadata(model_name: str) -> Dict[str, Any]:
    """Return metadata to persist alongside embeddings for invalidation."""

    return {
        "embedding_model": model_name,
        # Potential extension point: add checksum of raw data file, timestamp, etc.
    }


__all__ = [
    "RawFAQ",
    "ClassifiedFAQ",
    "ClassificationIntent",
    "FAQClassifier",
    "OpenAILLMClassifier",
    "load_raw_faq",
    "classify_faqs",
    "build_cache_metadata",
]


