"""Tests for citations in the QA SubAgent (ids, prompts, post-processing).

LLM, rag_search and Milvus are mocked; no external services are needed.
"""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Import qa_agent without heavy service modules ────────────
# Mock the service chain in sys.modules, import, then restore so other test
# files in the same session see the real modules. The two packages are
# replaced by empty stubs so their __init__ does not import sibling agents.

_APP_DIR = Path(__file__).resolve().parent.parent / "app"
_STUB_PACKAGES = {
    "app.agent.tools": _APP_DIR / "agent" / "tools",
    "app.agent.subagents": _APP_DIR / "agent" / "subagents",
}
_MOCKED_MODULE_NAMES = [
    "app.agent.tools.rag_search",
    "app.services.llm_client", "app.services.milvus_client",
]
_TOUCHED = [*_STUB_PACKAGES, *_MOCKED_MODULE_NAMES, "app.agent.subagents.qa_agent"]
_originals = {k: sys.modules[k] for k in _TOUCHED if k in sys.modules}

for _name, _path in _STUB_PACKAGES.items():
    _pkg = ModuleType(_name)
    _pkg.__path__ = [str(_path)]
    sys.modules[_name] = _pkg
for _name in _MOCKED_MODULE_NAMES:
    sys.modules[_name] = MagicMock()
sys.modules.pop("app.agent.subagents.qa_agent", None)

import importlib  # noqa: E402

qa = importlib.import_module("app.agent.subagents.qa_agent")  # noqa: E402

for _name in _TOUCHED:
    if _name in _originals:
        sys.modules[_name] = _originals[_name]
    else:
        sys.modules.pop(_name, None)

from app.services.citations import CITATION_RULES, REFUSAL_SENTENCE  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────


def _chunk(i: int) -> SimpleNamespace:
    return SimpleNamespace(
        text=f"chunk text {i}", source_doc=f"doc_{i}.md",
        heading_path=f"H/{i}", rerank_score=0.5,
    )


def _tool_call_msg(call_id: str, query: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": "rag_search", "arguments": json.dumps({"query": query})},
        }],
    }


def _answer_msg(text: str) -> dict:
    return {"role": "assistant", "content": text, "tool_calls": None}


def _reflection(passed: bool) -> dict:
    return {"role": "assistant", "content": json.dumps({"passed": passed, "feedback": "补充"})}


def _tool_ids(chat_mock: AsyncMock) -> list[list[int]]:
    """Chunk ids in every tool message the LLM saw, in call order."""
    ids = []
    for call in chat_mock.call_args_list:
        for m in call.kwargs["messages"]:
            if m.get("role") == "tool":
                ids.append([c["id"] for c in json.loads(m["content"])])
    return ids


@pytest.fixture
def agent():
    """Patch services; yield (chat mock, set_mode)."""
    chat = AsyncMock()
    search = MagicMock(side_effect=lambda query, top_k=5: [_chunk(i) for i in range(2)])
    llm = SimpleNamespace(chat=chat)
    milvus = SimpleNamespace(count=lambda: 10)
    with patch.object(qa, "llm_client", llm), \
         patch.object(qa, "_rag_search_sync", search), \
         patch.object(qa, "milvus_service", milvus), \
         patch.object(qa.settings, "citation_mode", "filter"):
        yield chat


# ── Prompts ──────────────────────────────────────────────────


def test_citation_prompt_keeps_existing_rules_and_adds_citation_rules():
    p = qa.REACT_CITATION_SYSTEM_PROMPT
    assert "篇幅控制在 200 字以内" in p
    assert CITATION_RULES in p
    assert "`id` 字段就是它的片段编号" in p
    assert CITATION_RULES not in qa.REACT_SYSTEM_PROMPT


def test_reflection_citation_prompt_requires_citations():
    assert "每句话都有引用" in qa.REFLECTION_CITATION_SYSTEM_PROMPT
    assert "每句话都有引用" not in qa.REFLECTION_SYSTEM_PROMPT


# ── Chunk ids ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ids_continue_across_rag_search_calls_and_retry(agent):
    agent.side_effect = [
        _tool_call_msg("c1", "q1"),       # ReAct round 1: search
        _answer_msg("A 是 B[1]。"),       # ReAct round 2: answer
        _reflection(False),               # reflection fails → retry
        _tool_call_msg("c2", "q2"),       # retry round 1: search again
        _answer_msg("C 是 D[3]。E[9]。"), # retry answer cites the new chunk
    ]
    result = await qa.run_qa_agent("问题")

    assert _tool_ids(agent)[-1] == [3, 4]
    assert [1, 2] in _tool_ids(agent)
    assert result["answer"] == "C 是 D。"
    assert result["confidence"] == "low"


@pytest.mark.asyncio
async def test_ids_restart_for_each_run(agent):
    agent.side_effect = [
        _tool_call_msg("c1", "q"), _answer_msg("A[1]。"), _reflection(True),
        _tool_call_msg("c2", "q"), _answer_msg("B[2]。"), _reflection(True),
    ]
    await qa.run_qa_agent("第一个问题")
    second = await qa.run_qa_agent("第二个问题")

    assert _tool_ids(agent)[-1] == [1, 2]
    assert second["answer"] == "B。"


# ── Post-processing by citation_mode ─────────────────────────


@pytest.mark.asyncio
async def test_filter_mode_drops_uncited_sentences(agent):
    agent.side_effect = [
        _tool_call_msg("c1", "q"),
        _answer_msg("A 是 B[1]。没有依据的话。"),
        _reflection(True),
    ]
    result = await qa.run_qa_agent("问题")

    assert result["answer"] == "A 是 B。"
    assert result["confidence"] == "high"
    first_call = agent.call_args_list[0].kwargs["messages"]
    assert first_call[0]["content"] == qa.REACT_CITATION_SYSTEM_PROMPT
    reflect_call = agent.call_args_list[2].kwargs["messages"]
    assert reflect_call[0]["content"] == qa.REFLECTION_CITATION_SYSTEM_PROMPT
    # Reflection reviews the raw answer, with markers
    assert "[1]" in reflect_call[1]["content"]


@pytest.mark.asyncio
async def test_filter_mode_without_retrieval_returns_refusal(agent):
    agent.side_effect = [_answer_msg("A 是 B[1]。"), _reflection(True)]
    result = await qa.run_qa_agent("问题")

    assert result["answer"] == REFUSAL_SENTENCE


@pytest.mark.asyncio
async def test_prompt_mode_strips_markers_only(agent):
    agent.side_effect = [
        _tool_call_msg("c1", "q"),
        _answer_msg("A 是 B[1]。没有依据的话。"),
        _reflection(True),
    ]
    with patch.object(qa.settings, "citation_mode", "prompt"):
        result = await qa.run_qa_agent("问题")

    assert result["answer"] == "A 是 B。没有依据的话。"


@pytest.mark.asyncio
async def test_off_mode_keeps_answer_and_old_prompts(agent):
    agent.side_effect = [
        _tool_call_msg("c1", "q"),
        _answer_msg("A 是 B。没有引用。"),
        _reflection(True),
    ]
    with patch.object(qa.settings, "citation_mode", "off"):
        result = await qa.run_qa_agent("问题")

    assert result["answer"] == "A 是 B。没有引用。"
    assert agent.call_args_list[0].kwargs["messages"][0]["content"] == qa.REACT_SYSTEM_PROMPT
    assert agent.call_args_list[2].kwargs["messages"][0]["content"] == qa.REFLECTION_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_fallback_mode_uses_fallback_rules_and_keeps_uncited_sentences(agent):
    from app.services.citations import FALLBACK_RULES

    agent.side_effect = [
        _tool_call_msg("c1", "q"),
        _answer_msg("笔记里没有找到相关内容。根据我的知识，X 是 Y。"),
        _reflection(True),
    ]
    with patch.object(qa.settings, "citation_mode", "fallback"):
        result = await qa.run_qa_agent("问题")

    assert result["answer"] == "笔记里没有找到相关内容。根据我的知识，X 是 Y。"
    system = agent.call_args_list[0].kwargs["messages"][0]["content"]
    assert system == qa.REACT_FALLBACK_SYSTEM_PROMPT
    assert FALLBACK_RULES in system and CITATION_RULES not in system


@pytest.mark.asyncio
async def test_two_rag_search_calls_in_one_round_keep_the_id_map(agent):
    """Regression: _react() used to overwrite the chunks id map with the tool result list,
    so a second rag_search in the same round crashed."""
    two_calls = _tool_call_msg("c1", "q1")
    two_calls["tool_calls"].append({
        "id": "c2", "type": "function",
        "function": {"name": "rag_search", "arguments": json.dumps({"query": "q2"})},
    })
    agent.side_effect = [two_calls, _answer_msg("A[1]。C[3]。"), _reflection(True)]
    result = await qa.run_qa_agent("问题")

    assert _tool_ids(agent)[:2] == [[1, 2], [3, 4]]
    assert result["answer"] == "A。C。"
    assert set(result["sources"]) == {"doc_0.md", "doc_1.md"}


@pytest.mark.asyncio
async def test_tool_error_result_does_not_crash_source_tracking(agent):
    with patch.object(qa, "_rag_search_sync", MagicMock(side_effect=RuntimeError("boom"))):
        agent.side_effect = [_tool_call_msg("c1", "q"), _answer_msg("A[1]。"), _reflection(True)]
        result = await qa.run_qa_agent("问题")

    assert result["sources"] == []
