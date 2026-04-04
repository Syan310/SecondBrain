"""Tests for Orchestrator: plan_node, execute_node, synthesize_node, graph routing.

These tests mock the LLM and SubAgent calls to validate orchestration logic
without requiring external services (FlagEmbedding, Milvus, etc.).
"""

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Pre-mock the entire subagent + service chain ────────────
# The project targets Python 3.10+ / Docker, but the local system has 3.9.
# Instead of importing through the full chain (which hits heavy deps like
# FlagEmbedding, pymilvus, torch etc.), we mock the subagent modules that
# orchestrator.py imports, then import orchestrator directly.

# Create mock subagent modules with the functions orchestrator needs
_mock_qa = MagicMock()
_mock_qa.run_qa_agent = AsyncMock()
_mock_gap = MagicMock()
_mock_gap.run_gap_agent = AsyncMock()

# Block the entire import chains
for mod_name in [
    # Third-party heavy deps
    "FlagEmbedding", "pymilvus", "transformers", "docx", "torch",
    "magic_pdf", "magic_pdf.data", "magic_pdf.data.data_reader_writer",
    "magic_pdf.pipe", "magic_pdf.pipe.UNIPipe",
    "aiosqlite",
    # App service layer
    "app.services.embedding", "app.services.reranker",
    "app.services.milvus_client", "app.services.doc_processor",
    # App DB layer
    "app.db", "app.db.models", "app.db.crud", "app.db.database",
    # App tools layer
    "app.agent.tools", "app.agent.tools.rag_search",
    "app.agent.tools.query_decompose", "app.agent.tools.read_mastery",
    "app.agent.tools.update_mastery", "app.agent.tools.search_concepts",
    # App subagents (except we set our mocks)
    "app.agent.subagents", "app.agent.subagents.quiz_agent",
]:
    sys.modules[mod_name] = MagicMock()

sys.modules["app.agent.subagents.qa_agent"] = _mock_qa
sys.modules["app.agent.subagents.gap_agent"] = _mock_gap

from app.agent.state import OverallState  # noqa: E402
from app.agent.nodes.orchestrator import (  # noqa: E402
    plan_node,
    execute_node,
    synthesize_node,
    passthrough_node,
    route_after_plan,
    route_after_execute,
    _parse_json,
)


# ── Helper ──────────────────────────────────────────────────


def _make_llm_response(content: str) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": None}


# ── _parse_json tests ───────────────────────────────────────


class TestParseJson:
    def test_plain_json(self):
        assert _parse_json('{"key": "value"}') == {"key": "value"}

    def test_json_in_markdown_fence(self):
        text = '```json\n{"key": "value"}\n```'
        assert _parse_json(text) == {"key": "value"}

    def test_json_in_generic_fence(self):
        text = '```\n{"key": "value"}\n```'
        assert _parse_json(text) == {"key": "value"}

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json("not json at all")


# ── plan_node tests ──────────────────────────────────────────


class TestPlanNode:
    @pytest.mark.asyncio
    async def test_single_qa_intent(self):
        plan_response = json.dumps({
            "steps": [
                {"step": 1, "intent": "qa", "params": {"query": "什么是Transformer?"}, "depends_on": []}
            ],
            "suggested_followup": None,
        })

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(plan_response))

            state: OverallState = {"user_query": "什么是Transformer?", "user_id": "test"}
            result = await plan_node(state)

            assert len(result["plan"]) == 1
            assert result["plan"][0]["intent"] == "qa"
            assert result["suggested_followup"] is None

    @pytest.mark.asyncio
    async def test_gap_analysis_intent(self):
        plan_response = json.dumps({
            "steps": [
                {"step": 1, "intent": "gap_analysis", "params": {"scope": None, "keyword": "NLP"}, "depends_on": []}
            ],
            "suggested_followup": None,
        })

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(plan_response))

            state: OverallState = {"user_query": "帮我看看NLP哪里薄弱", "user_id": "test"}
            result = await plan_node(state)

            assert len(result["plan"]) == 1
            assert result["plan"][0]["intent"] == "gap_analysis"

    @pytest.mark.asyncio
    async def test_multi_intent_gap_then_qa(self):
        plan_response = json.dumps({
            "steps": [
                {"step": 1, "intent": "gap_analysis", "params": {"scope": None, "keyword": "NLP"}, "depends_on": []},
                {"step": 2, "intent": "qa", "params": {"query": "讲解最弱概念"}, "depends_on": [1]},
            ],
            "suggested_followup": None,
        })

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(plan_response))

            state: OverallState = {"user_query": "分析NLP盲区然后讲讲最弱的", "user_id": "test"}
            result = await plan_node(state)

            assert len(result["plan"]) == 2
            assert result["plan"][1]["depends_on"] == [1]

    @pytest.mark.asyncio
    async def test_quiz_only_goes_to_followup(self):
        plan_response = json.dumps({
            "steps": [],
            "suggested_followup": {"intent": "quiz", "params": {"topics": ["NLP"]}},
        })

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(plan_response))

            state: OverallState = {"user_query": "来几道NLP的题", "user_id": "test"}
            result = await plan_node(state)

            assert len(result["plan"]) == 0
            assert result["suggested_followup"]["intent"] == "quiz"

    @pytest.mark.asyncio
    async def test_fallback_on_parse_error(self):
        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response("这不是JSON"))

            state: OverallState = {"user_query": "什么是Transformer?", "user_id": "test"}
            result = await plan_node(state)

            # Should fallback to single QA step
            assert len(result["plan"]) == 1
            assert result["plan"][0]["intent"] == "qa"
            assert result["plan"][0]["params"]["query"] == "什么是Transformer?"

    @pytest.mark.asyncio
    async def test_invalid_intent_filtered_out(self):
        plan_response = json.dumps({
            "steps": [
                {"step": 1, "intent": "qa", "params": {"query": "test"}, "depends_on": []},
                {"step": 2, "intent": "quiz", "params": {}, "depends_on": []},
            ],
            "suggested_followup": None,
        })

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(plan_response))

            state: OverallState = {"user_query": "test", "user_id": "test"}
            result = await plan_node(state)

            # quiz intent in steps should be filtered out
            intents = [s["intent"] for s in result["plan"]]
            assert "quiz" not in intents


# ── execute_node tests ───────────────────────────────────────


class TestExecuteNode:
    @pytest.mark.asyncio
    async def test_single_qa_step(self):
        mock_session = AsyncMock()

        with patch("app.agent.nodes.orchestrator.run_qa_agent", new_callable=AsyncMock) as mock_qa:
            mock_qa.return_value = {"answer": "Transformer是一种模型架构", "confidence": "high", "plan": {}, "sources": []}

            state: OverallState = {
                "plan": [{"step": 1, "intent": "qa", "params": {"query": "什么是Transformer?"}, "depends_on": []}],
                "user_query": "什么是Transformer?",
                "user_id": "test",
            }

            result = await execute_node(state, mock_session)

            assert len(result["step_results"]) == 1
            assert result["step_results"][0]["intent"] == "qa"
            assert "Transformer" in result["step_results"][0]["result"]
            mock_qa.assert_called_once_with(query="什么是Transformer?", user_id="test")

    @pytest.mark.asyncio
    async def test_single_gap_step(self):
        mock_session = AsyncMock()

        with patch("app.agent.nodes.orchestrator.run_gap_agent", new_callable=AsyncMock) as mock_gap:
            mock_gap.return_value = {"report": "NLP诊断报告：你在Attention方面较弱"}

            state: OverallState = {
                "plan": [{"step": 1, "intent": "gap_analysis", "params": {"scope": None, "keyword": "NLP"}, "depends_on": []}],
                "user_query": "分析NLP盲区",
                "user_id": "test",
            }

            result = await execute_node(state, mock_session)

            assert len(result["step_results"]) == 1
            assert result["step_results"][0]["intent"] == "gap_analysis"
            mock_gap.assert_called_once()

    @pytest.mark.asyncio
    async def test_serial_gap_then_qa_with_rewrite(self):
        mock_session = AsyncMock()

        rewrite_response = json.dumps({"query": "讲解Attention机制"})

        with patch("app.agent.nodes.orchestrator.run_gap_agent", new_callable=AsyncMock) as mock_gap, \
             patch("app.agent.nodes.orchestrator.run_qa_agent", new_callable=AsyncMock) as mock_qa, \
             patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:

            mock_gap.return_value = {"report": "最弱主题: Attention"}
            mock_qa.return_value = {"answer": "Attention是...", "confidence": "high", "plan": {}, "sources": []}
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(rewrite_response))

            state: OverallState = {
                "plan": [
                    {"step": 1, "intent": "gap_analysis", "params": {"scope": None, "keyword": "NLP"}, "depends_on": []},
                    {"step": 2, "intent": "qa", "params": {"query": "讲解最弱概念"}, "depends_on": [1]},
                ],
                "user_query": "分析NLP盲区然后讲讲最弱的",
                "user_id": "test",
            }

            result = await execute_node(state, mock_session)

            assert len(result["step_results"]) == 2
            # QA should have been called with rewritten query
            mock_qa.assert_called_once_with(query="讲解Attention机制", user_id="test")

    @pytest.mark.asyncio
    async def test_empty_plan(self):
        mock_session = AsyncMock()

        state: OverallState = {
            "plan": [],
            "user_query": "来几道题",
            "user_id": "test",
        }

        result = await execute_node(state, mock_session)
        assert result["step_results"] == []


# ── synthesize_node tests ────────────────────────────────────


class TestSynthesizeNode:
    @pytest.mark.asyncio
    async def test_synthesize_merges_results(self):
        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(
                return_value=_make_llm_response("综合结果：你的NLP盲区在Attention，以下是讲解...")
            )

            state: OverallState = {
                "step_results": [
                    {"step": 1, "intent": "gap_analysis", "result": "盲区: Attention"},
                    {"step": 2, "intent": "qa", "result": "Attention是注意力机制..."},
                ],
                "user_query": "分析盲区然后讲讲",
            }

            result = await synthesize_node(state)
            assert "final_answer" in result
            assert len(result["final_answer"]) > 0


# ── passthrough_node tests ───────────────────────────────────


class TestPassthroughNode:
    @pytest.mark.asyncio
    async def test_single_result_passthrough(self):
        state: OverallState = {
            "step_results": [{"step": 1, "intent": "qa", "result": "Transformer是..."}],
        }
        result = await passthrough_node(state)
        assert result["final_answer"] == "Transformer是..."

    @pytest.mark.asyncio
    async def test_empty_results(self):
        state: OverallState = {"step_results": []}
        result = await passthrough_node(state)
        assert "没有" in result["final_answer"]


# ── Routing function tests ───────────────────────────────────


class TestRouting:
    def test_route_after_plan_has_steps(self):
        state: OverallState = {"plan": [{"step": 1}]}
        assert route_after_plan(state) == "has_steps"

    def test_route_after_plan_no_steps(self):
        state: OverallState = {"plan": []}
        assert route_after_plan(state) == "no_steps"

    def test_route_after_execute_single(self):
        state: OverallState = {"step_results": [{"step": 1}]}
        assert route_after_execute(state) == "single"

    def test_route_after_execute_multi(self):
        state: OverallState = {"step_results": [{"step": 1}, {"step": 2}]}
        assert route_after_execute(state) == "multi"


# ── Integration: graph build + invoke ────────────────────────


class TestGraphIntegration:
    @pytest.mark.asyncio
    async def test_single_qa_flow(self):
        """Test full graph: single QA intent -> plan -> execute -> passthrough -> END."""
        mock_session = AsyncMock()

        plan_response = json.dumps({
            "steps": [{"step": 1, "intent": "qa", "params": {"query": "什么是Transformer?"}, "depends_on": []}],
            "suggested_followup": None,
        })

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm, \
             patch("app.agent.nodes.orchestrator.run_qa_agent", new_callable=AsyncMock) as mock_qa:

            mock_llm.chat = AsyncMock(return_value=_make_llm_response(plan_response))
            mock_qa.return_value = {"answer": "Transformer是一种基于自注意力的模型", "confidence": "high", "plan": {}, "sources": []}

            from app.agent.graph import run_orchestrator

            result = await run_orchestrator(
                user_query="什么是Transformer?",
                user_id="test",
                session=mock_session,
            )

            assert "Transformer" in result["final_answer"]
            assert result["suggested_followup"] is None

    @pytest.mark.asyncio
    async def test_quiz_only_flow(self):
        """Test full graph: quiz-only -> plan (empty steps) -> quiz_passthrough -> END."""
        mock_session = AsyncMock()

        plan_response = json.dumps({
            "steps": [],
            "suggested_followup": {"intent": "quiz", "params": {"topics": ["NLP"]}},
        })

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(plan_response))

            from app.agent.graph import run_orchestrator

            result = await run_orchestrator(
                user_query="来几道NLP的题",
                user_id="test",
                session=mock_session,
            )

            assert result["suggested_followup"]["intent"] == "quiz"

    @pytest.mark.asyncio
    async def test_multi_intent_flow(self):
        """Test full graph: gap + qa -> plan -> execute -> synthesize -> END."""
        mock_session = AsyncMock()

        plan_response = json.dumps({
            "steps": [
                {"step": 1, "intent": "gap_analysis", "params": {"scope": None, "keyword": "NLP"}, "depends_on": []},
                {"step": 2, "intent": "qa", "params": {"query": "讲解最弱概念"}, "depends_on": [1]},
            ],
            "suggested_followup": None,
        })

        rewrite_response = json.dumps({"query": "什么是Attention机制"})

        # LLM will be called 3 times: plan, rewrite, synthesize
        call_count = {"n": 0}

        async def mock_chat(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_llm_response(plan_response)
            elif call_count["n"] == 2:
                return _make_llm_response(rewrite_response)
            else:
                return _make_llm_response("综合分析：你的NLP盲区在Attention。Attention是一种注意力机制...")

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm, \
             patch("app.agent.nodes.orchestrator.run_gap_agent", new_callable=AsyncMock) as mock_gap, \
             patch("app.agent.nodes.orchestrator.run_qa_agent", new_callable=AsyncMock) as mock_qa:

            mock_llm.chat = mock_chat
            mock_gap.return_value = {"report": "盲区: Attention (effective_mastery=0.1)"}
            mock_qa.return_value = {"answer": "Attention是注意力机制...", "confidence": "high", "plan": {}, "sources": []}

            from app.agent.graph import run_orchestrator

            result = await run_orchestrator(
                user_query="分析NLP盲区然后讲讲最弱的",
                user_id="test",
                session=mock_session,
            )

            assert len(result["step_results"]) == 2
            assert len(result["final_answer"]) > 0
            mock_gap.assert_called_once()
            mock_qa.assert_called_once()
