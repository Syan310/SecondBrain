"""Judge LLM client — evaluation judge via an OpenAI-compatible API.

Used for LLM-as-Judge evaluation (Faithfulness, Relevancy, Quiz Quality, etc.)
Separated from the main LLM client to avoid self-evaluation bias: the judge
(Claude Sonnet via OpenRouter) comes from a different vendor than the app model.

Claude constraints:
- temperature, top_p, top_k are rejected (HTTP 400), so they are never sent
- reasoning parameters are left at the OpenRouter default
"""

from __future__ import annotations

import logging
from openai import AsyncOpenAI

from app.config import settings
from app.services.llm_client import build_routing_extra_body, make_async_openai

logger = logging.getLogger(__name__)


def build_judge_request_kwargs(
    *,
    model: str,
    messages: list[dict],
    max_completion_tokens: int | None = None,
    extra_body: dict | None = None,
) -> dict:
    """Build Chat Completions kwargs for the judge path.

    No sampling parameters are sent: Claude rejects temperature/top_p/top_k.
    ``extra_body`` carries OpenRouter provider routing.
    """
    create_kwargs: dict = {
        "model": model,
        "messages": messages,
    }

    if extra_body:
        create_kwargs["extra_body"] = extra_body
    if max_completion_tokens is not None:
        create_kwargs["max_completion_tokens"] = max_completion_tokens

    logger.info(
        "Judge request: model='%s', max_completion_tokens=%s, extra_body=%s",
        model,
        max_completion_tokens,
        extra_body,
    )
    return create_kwargs


class JudgeClient:
    """Async wrapper for evaluation judging."""

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = make_async_openai(
                api_key=settings.judge_api_key,
                base_url=settings.judge_base_url,
                role="judge",
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
            extra_body=build_routing_extra_body(
                settings.judge_base_url, settings.judge_provider_order
            ),
        )
        response = await client.chat.completions.create(**create_kwargs)
        return response.choices[0].message.content or ""


judge_client = JudgeClient()
