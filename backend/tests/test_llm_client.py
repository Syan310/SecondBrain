import httpx
import pytest

from app.services.llm_client import (
    UsageTracker,
    build_routing_extra_body,
    describe_provider,
    make_async_openai,
    usage_tracker,
)


def test_openrouter_pins_provider_and_disables_fallbacks():
    assert build_routing_extra_body("https://openrouter.ai/api/v1", ["anthropic"]) == {
        "usage": {"include": True},
        "provider": {"order": ["anthropic"], "allow_fallbacks": False},
    }


def test_openrouter_without_order_sends_no_provider_block():
    assert build_routing_extra_body("https://openrouter.ai/api/v1", []) == {
        "usage": {"include": True},
    }


def test_deepseek_direct_disables_thinking_and_ignores_provider_order():
    assert build_routing_extra_body("https://api.deepseek.com", ["deepseek"]) == {
        "thinking": {"type": "disabled"},
    }


def test_other_endpoints_get_no_extra_body():
    assert build_routing_extra_body("https://api.example.com/v1", ["x"]) == {}


def test_describe_provider_prefers_openrouter_field_then_host():
    assert describe_provider("https://openrouter.ai/api/v1", {"provider": "Anthropic"}) == "Anthropic"
    assert describe_provider("https://api.deepseek.com", {}) == "api.deepseek.com"


def test_usage_tracker_accumulates_per_role():
    tracker = UsageTracker()
    payload = {
        "model": "anthropic/claude-sonnet-5.5",
        "provider": "Anthropic",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001},
    }
    tracker.record("judge", "https://openrouter.ai/api/v1", payload)
    tracker.record("judge", "https://openrouter.ai/api/v1", payload)

    snap = tracker.snapshot()["judge"]
    assert snap["requests"] == 2
    assert snap["prompt_tokens"] == 20
    assert snap["completion_tokens"] == 10
    assert snap["cost_usd"] == pytest.approx(0.002)
    assert snap["providers"] == {"Anthropic": 2}
    assert snap["models"] == {"anthropic/claude-sonnet-5.5": 2}


def test_usage_tracker_sums_deepseek_cache_tokens():
    tracker = UsageTracker()
    payload = {"model": "deepseek-flash", "usage": {
        "prompt_tokens": 10, "completion_tokens": 1,
        "prompt_cache_hit_tokens": 7, "prompt_cache_miss_tokens": 3,
    }}
    tracker.record("llm", "https://api.deepseek.com", payload)
    tracker.record("llm", "https://api.deepseek.com", payload)
    snap = tracker.snapshot()["llm"]
    assert (snap["prompt_cache_hit_tokens"], snap["prompt_cache_miss_tokens"]) == (14, 6)


@pytest.mark.asyncio
async def test_chat_returns_usage_with_deepseek_cache_fields(monkeypatch):
    from app.services import llm_client as mod

    body = {
        "id": "x", "object": "chat.completion", "created": 0, "model": "deepseek-flash",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4,
                  "prompt_cache_hit_tokens": 2, "prompt_cache_miss_tokens": 1},
    }
    client = make_async_openai(api_key="k", base_url="https://api.deepseek.com", role="llm")
    client._client._transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    wrapper = mod.LLMClient()
    monkeypatch.setattr(wrapper, "_get_client", lambda: client)

    resp = await wrapper.chat(messages=[{"role": "user", "content": "hi"}])

    assert resp["content"] == "ok"
    assert resp["usage"]["prompt_tokens"] == 3
    assert resp["usage"]["prompt_cache_hit_tokens"] == 2
    usage_tracker.reset()


@pytest.mark.asyncio
async def test_make_async_openai_records_json_responses():
    usage_tracker.reset()
    body = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-flash",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    client = make_async_openai(api_key="k", base_url="https://api.deepseek.com", role="llm")
    # Swap the transport for a mock while keeping the recording hook.
    client._client._transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))

    resp = await client.chat.completions.create(model="deepseek-flash", messages=[])

    assert resp.choices[0].message.content == "ok"
    snap = usage_tracker.snapshot()["llm"]
    assert snap["requests"] == 1
    assert snap["prompt_tokens"] == 3
    assert snap["providers"] == {"api.deepseek.com": 1}
    usage_tracker.reset()
