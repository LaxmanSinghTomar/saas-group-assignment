"""LLM-based response generation for FAQ search results."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence

from .search import SearchResponse

DEFAULT_COMPLETION_MODEL = "gpt-4.1-mini"


@dataclass
class GenerationSettings:
    model: str = DEFAULT_COMPLETION_MODEL
    temperature: float = 0.2
    max_context: int = 3


def _get_client():  # pragma: no cover - network helper
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set; cannot generate responses")
    return OpenAI(api_key=api_key)


def _build_context(search_response: SearchResponse, max_context: int) -> List[str]:
    context_lines: List[str] = []
    for scored in search_response.results[:max_context]:
        context_lines.append(
            f"Intent: {scored.document.intent}\nQuestion: {scored.document.question}\nAnswer: {scored.document.answer}"
        )
    return context_lines


def _build_messages(search_response: SearchResponse, context: Sequence[str]) -> List[dict]:
    system_prompt = (
         "You are a friendly, confident customer support assistant with a helpful personality. Answer using the FAQ context provided.\n"
         "Handle the scenarios precisely:\n"
        "1. Clear match (high confidence): Provide only the answer drawn from the matched FAQ. Include concise steps if (and only if) they appear in that FAQ. Do not add extra tips, reminders, or follow-up questions.\n"
        "2. Ambiguous query (medium confidence, generic wording, or multiple relevant FAQs): Start with \"Hmm, I'm not sure what you mean by that. As an FAQ assistant, I can help you with:\" followed by a '-' bullet list that groups related FAQs into warm, action-oriented summaries (e.g., '- manage account issues like suspensions or username changes'). If the query is very broad but still about capabilities (e.g., 'what can you help me with', 'what do you do'), skip the bullets and instead respond with: \"I can help with account management, security & privacy, billing & subscriptions, integrations, and troubleshooting. Let me know which area you'd like to explore.\" Keep each bullet under 12 words. After any list, end with a short, upbeat question inviting the user to pick a topic. Do NOT provide step-by-step instructions yet.\n"
        "3. Off-topic: Reply with 'I'm an FAQ assistant and can't help with that request.'\n"
        "4. Unknown but potentially relevant (no strong FAQ match in the provided context): Say 'Sorry, I don't have information on <topic>. Please reach out to support@example.com.'\n"
        "5. Simple greetings or thanks (e.g., 'hi', 'hello', 'thanks'): Offer a quick friendly acknowledgment, mention you're here for FAQ help, and invite them to ask when they're ready—avoid listing specific topics.\n"
        "6. Farewells (e.g., 'goodbye', 'take care'): Respond with a warm goodbye and let them know you're available if they need more help later.\n"
        "Rules:\n- Treat very short or generic single-word queries (like 'password', 'account', 'billing') as ambiguous unless the FAQ context clearly shows a single, high-confidence match.\n- Keep the tone warm, upbeat, and conversational while staying concise.\n- Never fabricate information—only use details that appear in the FAQ context.\n- Be concise (around 2 sentences unless sharing a full FAQ answer).\n"
    )

    context_block = "\n\n".join(context) if context else "No relevant FAQ context was retrieved."

    user_prompt = (
        f"User query: {search_response.query}\n\n"
        f"FAQ context:\n{context_block}\n\n"
        "Respond according to the rules above."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_response(
    search_response: SearchResponse,
    settings: GenerationSettings | None = None,
):
    """Use an LLM to craft the final user-facing response."""

    settings = settings or GenerationSettings()
    client = _get_client()
    context = _build_context(search_response, settings.max_context)
    messages = _build_messages(search_response, context)

    response = client.chat.completions.create(
        model=settings.model,
        messages=messages,
        temperature=settings.temperature,
    )

    message = response.choices[0].message
    return {
        "answer": message.content if message.content else "",
        "metadata": {
            "model": settings.model,
            "temperature": settings.temperature,
            "context_used": len(context),
            "thresholds": search_response.thresholds,
        },
    }


__all__ = ["GenerationSettings", "generate_response", "DEFAULT_COMPLETION_MODEL"]