"""Judge LLM client — evaluation judge via OpenAI-compatible API.

Used for LLM-as-Judge evaluation (Faithfulness, Relevancy, Quiz Quality, etc.)
Separated from main LLM client to avoid self-evaluation bias (Qwen judges Qwen).
"""

from __future__ import annotations

import logging
from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

GPT5_JUDGE_MODEL_PREFIX = "gpt-5.4"


def build_judge_request_kwargs(
    *,
    model: str,
    messages: list[dict],
    max_completion_tokens: int,
) -> dict:
    """Build OpenAI Chat Completions kwargs for the GPT-5.4 judge path.

    GPT-5.4 follows the reasoning-model parameter contract: use
    ``max_completion_tokens`` and keep temperature fixed at ``1``.
    """
    normalized_model = (model or "").strip().lower()
    if normalized_model and not normalized_model.startswith(GPT5_JUDGE_MODEL_PREFIX):
        logger.warning(
            "Judge model '%s' is outside the validated GPT-5.4 path; continuing with GPT-5.4-compatible params.",
            model,
        )

    return {
        "model": model,
        "messages": messages,
        "temperature": 1,
        "max_completion_tokens": max_completion_tokens,
    }


class JudgeClient:
    """Async wrapper for evaluation judging."""

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
        max_completion_tokens: int = 2048,
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
