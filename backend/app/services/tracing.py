"""Langfuse observability integration (optional, no-op when not configured).

When LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set in .env,
all @observe()-decorated functions produce traces/spans and every
OpenAI call made via llm_client is auto-instrumented.

When keys are absent everything degrades to zero-cost no-ops —
no network calls, no side-effects, no import errors.

Supports langfuse v4+ API (get_client / propagate_attributes).
"""

from __future__ import annotations

import logging
import os

from app.config import settings

logger = logging.getLogger(__name__)

enabled: bool = bool(settings.langfuse_public_key and settings.langfuse_secret_key)

if enabled:
    # Langfuse SDK reads credentials from os.environ, but pydantic-settings
    # loads .env into its own model without exporting.  Bridge the gap here.
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)

    from langfuse import observe, get_client, propagate_attributes  # noqa: F401

    langfuse_client = get_client()
    logger.info("Langfuse tracing enabled (host=%s)", settings.langfuse_host)
else:

    def observe(func=None, *, name=None, as_type=None, **_kw):  # type: ignore[misc]
        """No-op decorator that handles @observe / @observe() / @observe(name=...)."""

        def decorator(fn):
            return fn

        if func is not None:
            return func
        return decorator

    def propagate_attributes(**_kw):  # type: ignore[misc]
        """No-op context manager mimicking langfuse.propagate_attributes."""
        from contextlib import nullcontext
        return nullcontext()

    class _NoOpClient:
        """Silently absorb any attribute access or method call."""

        def __getattr__(self, _name: str):
            return lambda *_a, **_kw: None

    langfuse_client = _NoOpClient()  # type: ignore[assignment]

    logger.debug("Langfuse tracing disabled (no keys configured)")


def get_callback_handler():
    """Return a Langfuse LangGraph/LangChain CallbackHandler when tracing is enabled."""
    if not enabled:
        return None
    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except ImportError:
        logger.warning("Langfuse CallbackHandler not available — LangGraph spans won't be traced")
        return None


def flush() -> None:
    """Flush pending Langfuse events. Call on app shutdown."""
    if enabled:
        langfuse_client.flush()
