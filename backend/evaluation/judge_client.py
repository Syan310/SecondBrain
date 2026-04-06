"""Judge LLM client — Kimi-K2.5 via OpenAI-compatible Moonshot API.

Used for LLM-as-Judge evaluation (Faithfulness, Relevancy, Quiz Quality, etc.)
Separated from main LLM client to avoid self-evaluation bias (Qwen judges Qwen).
"""

from __future__ import annotations

import logging
from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class JudgeClient:
    """Async wrapper around Kimi-K2.5 for evaluation judging."""

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
        # Moonshot kimi-k2.5 rejects temperature=0; API requires 1 for this model.
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> str:
        """Single judge call, returns content string."""
        client = self._get_client()
        response = await client.chat.completions.create(
            model=settings.judge_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


judge_client = JudgeClient()
