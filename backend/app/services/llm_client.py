from __future__ import annotations

"""LLM client for Qwen3.6-Plus via OpenAI-compatible DashScope API.

Provides:
- chat()        : non-streaming, for agent internal reasoning / tool-calling
- stream_chat() : streaming (async generator), for user-facing responses
"""

import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Async wrapper around Qwen3.6-Plus via OpenAI-compatible API."""

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
            )
        return self._client

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
    ) -> dict:
        """Non-streaming chat completion.

        Used by agent internal reasoning and tool-calling.

        Args:
            messages: OpenAI-format message list.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            tools: Tool definitions for function calling.
            tool_choice: Tool choice strategy ("auto", "none", or specific tool).

        Returns:
            The response message dict (role, content, tool_calls, etc.).
        """
        client = self._get_client()

        kwargs = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        response = await client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        return {
            "role": message.role,
            "content": message.content,
            "tool_calls": (
                [tc.model_dump() for tc in message.tool_calls]
                if message.tool_calls
                else None
            ),
        }

    async def stream_chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Streaming chat completion (async generator).

        Used for user-facing responses. Yields content chunks as they arrive.

        Args:
            messages: OpenAI-format message list.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.

        Yields:
            Content string chunks.
        """
        client = self._get_client()

        stream = await client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


# Module-level singleton
llm_client = LLMClient()
