"""LLM-judge evaluation for generation responses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
from datetime import datetime, UTC

from openai import OpenAI

from src.generation import generate_response, GenerationSettings
from src.search import hybrid_search
from evals.evals_retrieval import ensure_hybrid_index

import argparse


JUDGE_PROMPT = """
 You are evaluating responses from an FAQ assistant. Use the following rubric:
 
 - Category: direct or paraphrase
   - Expected behavior: Provide only the FAQ answer (and its in-scope steps) that matches the user's question. Do not include extra tips, personal opinions, or follow-up questions.
 - Category: ambiguous
  - Expected behavior: Present relevant options (bulleted or short phrases are fine) and ask for clarification. No step-by-step instructions yet.
 - Category: off_topic
   - Expected behavior: Respond with: "I'm an FAQ assistant and can't help with that request."
 - Category: unknown
  - Expected behavior: Use the explicit fallback message ("Sorry, I don't have information on <topic>. Please reach out to support@example.com.") because these test cases intentionally lack FAQ coverage—even if the topic sounds relevant. Do NOT penalize the assistant for answering with that fallback, and do NOT expect the off-topic refusal here.
 - Category: greeting
  - Expected behavior: Reply with a brief friendly acknowledgment, mention you're here for FAQ help, and invite the user to ask when ready.
 - Category: farewell
  - Expected behavior: Offer a warm goodbye, optionally remind the user they can return for help later.
 - Category: capability
  - Expected behavior: When the user asks about the assistant's overall abilities (e.g., "What can you do?"), clearly list the supported areas—account management, security & privacy, billing & subscriptions, integrations, troubleshooting—without bullet lists.
 
 Scoring:
 - good: Response fully meets the expectation for the category.
 - mixed: Response mostly meets the expectation but includes minor issues (extra detail, slightly off phrasing).
 - bad: Response misses the expectation (wrong content, missing clarification, hallucination, etc.).

Return JSON with fields: score (good/mixed/bad), reasoning (short explanation).
"""


@dataclass
class GenerationEvaluation:
    case_id: str
    category: str
    query: str
    response: str
    score: str
    reasoning: str


def load_eval_cases(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_model_response(query: str, hybrid_index, top_k: int = 3) -> str:
    search_response = hybrid_search(query, hybrid_index, top_k=top_k)
    generation = generate_response(search_response, settings=GenerationSettings())
    return generation["answer"], generation


def judge_response(query: str, category: str, response: str, client: OpenAI) -> dict:
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "category": category,
                    "user_query": query,
                    "assistant_response": response,
                },
                ensure_ascii=False,
            ),
        },
    ]
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    payload = completion.choices[0].message.content
    return json.loads(payload)


def evaluate_generation_cases(
    cases: Iterable[dict],
    *,
    output_path: Path,
    top_k: int = 3,
) -> List[GenerationEvaluation]:
    hybrid_index = ensure_hybrid_index()
    client = OpenAI()
    results: List[GenerationEvaluation] = []

    for case in cases:
        answer, metadata = generate_model_response(case["query"], hybrid_index, top_k=top_k)
        judgment = judge_response(case["query"], case["category"], answer, client)
        results.append(
            GenerationEvaluation(
                case_id=case["id"],
                category=case["category"],
                query=case["query"],
                response=answer,
                score=judgment.get("score", "unknown"),
                reasoning=judgment.get("reasoning", ""),
            )
        )

    payload = [result.__dict__ for result in results]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"created_at": datetime.now(UTC).isoformat(), "results": payload}, f, ensure_ascii=False, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate FAQ generation quality using an LLM judge.")
    parser.add_argument("--cases", type=Path, default=Path("data/evals/generation_eval_cases.json"), help="Path to the evaluation cases JSON.")
    parser.add_argument("--output", type=Path, default=Path("data/evals/generation_judge_results.json"), help="Where to save judged responses.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of search results to provide as context.")
    parser.add_argument("--print-summary", action="store_true", help="Print aggregate judge scores after evaluation.")
    args = parser.parse_args()

    cases = load_eval_cases(args.cases)
    results = evaluate_generation_cases(cases, output_path=args.output, top_k=args.top_k)

    if args.print_summary:
        total = len(results)
        good = sum(1 for r in results if r.score == "good")
        mixed = sum(1 for r in results if r.score == "mixed")
        bad = sum(1 for r in results if r.score == "bad")
        print(f"Cases evaluated: {total}")
        print(f"good: {good}")
        print(f"mixed: {mixed}")
        print(f"bad: {bad}")


if __name__ == "__main__":
    main()
