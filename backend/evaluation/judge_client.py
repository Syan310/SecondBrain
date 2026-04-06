"""Judge LLM client — evaluation judge via Kimi K2.5 (OpenAI-compatible API).

Used for LLM-as-Judge evaluation (Faithfulness, Relevancy, Quiz Quality, etc.)
Separated from main LLM client to avoid self-evaluation bias (Qwen judges Qwen).

Kimi K2.5 constraints (per official docs):
- temperature, top_p, frequency_penalty, presence_penalty, n are NOT modifiable
- Use max_completion_tokens (not max_tokens) for output length control
- thinking: {"type": "disabled"} to skip reasoning for judge tasks
"""

from __future__ import annotations

import logging
from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


def build_judge_request_kwargs(
    *,
    model: str,
    messages: list[dict],
    max_completion_tokens: int | None = None,
) -> dict:
    """Build Kimi K2.5 Chat Completions kwargs for the judge path.

    Kimi K2.5 does not accept temperature, top_p, frequency_penalty,
    presence_penalty, or n — these are fixed by the model and must be omitted.
    """
    create_kwargs: dict = {
        "model": model,
        "messages": messages,
        "extra_body": {"thinking": {"type": "disabled"}},
    }

    if max_completion_tokens is not None:
        create_kwargs["max_completion_tokens"] = max_completion_tokens

    logger.info(
        "Judge request: model='%s', thinking=disabled, max_completion_tokens=%s",
        model,
        max_completion_tokens,
    )
    return create_kwargs


class JudgeClient:
    """Async wrapper for evaluation judging via Kimi K2.5."""

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=settings.judge_api_key,
                base_url=settings.judge_base_url,
            )
        return self._client

    async def judge(
        self,
        messages: list[dict],
        max_completion_tokens: int | None = None,
    ) -> str:
        """Single judge call, returns content string."""
        client = self._get_client()
        create_kwargs = build_judge_request_kwargs(
            model=settings.judge_model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
        )
        response = await client.chat.completions.create(**create_kwargs)
        return response.choices[0].message.content or ""


judge_client = JudgeClient()
