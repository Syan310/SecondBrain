from __future__ import annotations

"""LLM client for the application model via an OpenAI-compatible API.

Provides:
- chat()        : non-streaming, for agent internal reasoning / tool-calling
- stream_chat() : streaming (async generator), for user-facing responses

Also provides the shared plumbing used by the judge clients:
- build_routing_extra_body() : per-endpoint request extras (OpenRouter provider
  pinning, DeepSeek thinking switch)
- make_async_openai()        : AsyncOpenAI whose non-streaming responses are
  recorded in usage_tracker (tokens, cost, actual provider)
"""

import json
import logging
import threading
from collections import Counter
from typing import Any, AsyncGenerator

import httpx
from openai import AsyncOpenAI, APIConnectionError, APIStatusError, RateLimitError

from app.config import settings

logger = logging.getLogger(__name__)


def build_routing_extra_body(base_url: str, provider_order: list[str]) -> dict[str, Any]:
    """Build the ``extra_body`` for one endpoint.

    - OpenRouter: ask for cost in ``usage``; pin providers in ``provider_order``
      and disable fallbacks, so every request is served by the same provider.
    - DeepSeek official API: thinking is on by default, so turn it off to get
      the non-reasoning chat behaviour.
    """
    extra_body: dict[str, Any] = {}
    if "openrouter.ai" in base_url:
        extra_body["usage"] = {"include": True}  # adds usage.cost to the response
        if provider_order:
            extra_body["provider"] = {"order": list(provider_order), "allow_fallbacks": False}
    if "api.deepseek.com" in base_url:
        extra_body["thinking"] = {"type": "disabled"}
    return extra_body


def describe_provider(base_url: str, payload: dict[str, Any]) -> str:
    """Name the provider that served a response.

    OpenRouter returns it in the ``provider`` field; a direct API is named by host.
    """
    if payload.get("provider"):
        return str(payload["provider"])
    return httpx.URL(base_url).host or base_url


class UsageTracker:
    """Thread-safe per-role counters for requests, tokens, cost, and providers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._roles: dict[str, dict[str, Any]] = {}

    def reset(self) -> None:
        with self._lock:
            self._roles = {}

    def record(self, role: str, base_url: str, payload: dict[str, Any]) -> None:
        usage = payload.get("usage") or {}
        with self._lock:
            entry = self._roles.setdefault(
                role,
                {
                    "requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "prompt_cache_hit_tokens": 0,
                    "prompt_cache_miss_tokens": 0,
                    "cost_usd": 0.0,
                    "providers": Counter(),
                    "models": Counter(),
                },
            )
            entry["requests"] += 1
            entry["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            entry["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            # DeepSeek only: prompt tokens split by prefix-cache hit / miss
            entry["prompt_cache_hit_tokens"] += int(usage.get("prompt_cache_hit_tokens") or 0)
            entry["prompt_cache_miss_tokens"] += int(usage.get("prompt_cache_miss_tokens") or 0)
            entry["cost_usd"] += float(usage.get("cost") or 0.0)
            entry["providers"][describe_provider(base_url, payload)] += 1
            entry["models"][str(payload.get("model", ""))] += 1

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                role: {
                    **{k: v for k, v in entry.items() if k not in ("providers", "models")},
                    "providers": dict(entry["providers"]),
                    "models": dict(entry["models"]),
                }
                for role, entry in self._roles.items()
            }


usage_tracker = UsageTracker()


def make_async_openai(
    *,
    api_key: str,
    base_url: str,
    role: str,
    client_cls: type[AsyncOpenAI] = AsyncOpenAI,
) -> AsyncOpenAI:
    """Create an AsyncOpenAI client that records every JSON response under ``role``.

    Streaming responses are skipped: reading their body here would consume the stream.
    """

    async def _record_response(response: httpx.Response) -> None:
        if "application/json" not in response.headers.get("content-type", ""):
            return
        await response.aread()
        try:
            payload = json.loads(response.content)
        except ValueError:
            return
        if isinstance(payload, dict) and "choices" in payload:
            usage_tracker.record(role, base_url, payload)

    return client_cls(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=10.0),
            event_hooks={"response": [_record_response]},
        ),
    )


class LLMClient:
    """Async wrapper around the application model via OpenAI-compatible API."""

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            client_cls: type[AsyncOpenAI] = AsyncOpenAI
            if settings.langfuse_public_key and settings.langfuse_secret_key:
                from langfuse.openai import AsyncOpenAI as TracedAsyncOpenAI

                client_cls = TracedAsyncOpenAI
            self._client = make_async_openai(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                role="llm",
                client_cls=client_cls,
            )
        return self._client

    @staticmethod
    def _extra_body() -> dict[str, Any]:
        return build_routing_extra_body(settings.llm_base_url, settings.llm_provider_order)

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
            The response message dict (role, content, tool_calls), plus ``usage``:
            the response's token usage as a dict, or None.
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
        extra_body = self._extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body

        try:
            response = await client.chat.completions.create(**kwargs)
        except (APIConnectionError, APIStatusError, RateLimitError) as e:
            logger.error("LLM API call failed: %s", e)
            raise

        message = response.choices[0].message

        return {
            "role": message.role,
            "content": message.content,
            "tool_calls": (
                [tc.model_dump() for tc in message.tool_calls]
                if message.tool_calls
                else None
            ),
            "usage": response.usage.model_dump() if response.usage else None,
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

        try:
            stream = await client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                extra_body=self._extra_body() or None,
            )
        except (APIConnectionError, APIStatusError, RateLimitError) as e:
            logger.error("LLM stream API call failed: %s", e)
            raise

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


# Module-level singleton
llm_client = LLMClient()
