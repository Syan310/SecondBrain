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
_mock_quiz = MagicMock()
_mock_quiz.run_quiz_agent = AsyncMock()

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
    "app.agent.subagents",
]:
    sys.modules[mod_name] = MagicMock()

sys.modules["app.agent.subagents.qa_agent"] = _mock_qa
sys.modules["app.agent.subagents.gap_agent"] = _mock_gap
sys.modules["app.agent.subagents.quiz_agent"] = _mock_quiz

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
        })

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(plan_response))

            state: OverallState = {"user_query": "什么是Transformer?", "user_id": "test"}
            result = await plan_node(state)

            assert len(result["plan"]) == 1
            assert result["plan"][0]["intent"] == "qa"

    @pytest.mark.asyncio
    async def test_gap_analysis_intent(self):
        plan_response = json.dumps({
            "steps": [
                {"step": 1, "intent": "gap_analysis", "params": {"scope": None, "keyword": "NLP"}, "depends_on": []}
            ],
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
        })

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(plan_response))

            state: OverallState = {"user_query": "分析NLP盲区然后讲讲最弱的", "user_id": "test"}
            result = await plan_node(state)

            assert len(result["plan"]) == 2
            assert result["plan"][1]["depends_on"] == [1]

    @pytest.mark.asyncio
    async def test_quiz_in_steps(self):
        """Quiz should be allowed in steps as the last step."""
        plan_response = json.dumps({
            "steps": [
                {"step": 1, "intent": "qa", "params": {"query": "Attention机制"}, "depends_on": []},
                {"step": 2, "intent": "quiz", "params": {"topics": ["Attention"]}, "depends_on": [1]},
            ],
        })

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(plan_response))

            state: OverallState = {"user_query": "讲讲Attention然后出题考我", "user_id": "test"}
            result = await plan_node(state)

            assert len(result["plan"]) == 2
            assert result["plan"][1]["intent"] == "quiz"
            assert result["plan"][1]["depends_on"] == [1]

    @pytest.mark.asyncio
    async def test_pure_quiz_intent(self):
        """Pure quiz request should have quiz as the only step."""
        plan_response = json.dumps({
            "steps": [
                {"step": 1, "intent": "quiz", "params": {"topics": ["NLP"]}, "depends_on": []}
            ],
        })

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(plan_response))

            state: OverallState = {"user_query": "来几道NLP的题", "user_id": "test"}
            result = await plan_node(state)

            assert len(result["plan"]) == 1
            assert result["plan"][0]["intent"] == "quiz"

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
                {"step": 2, "intent": "unknown_intent", "params": {}, "depends_on": []},
            ],
        })

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(plan_response))

            state: OverallState = {"user_query": "test", "user_id": "test"}
            result = await plan_node(state)

            # unknown intent should be filtered out
            intents = [s["intent"] for s in result["plan"]]
            assert "unknown_intent" not in intents
            assert "qa" in intents


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
    async def test_quiz_step_dispatches_to_quiz_agent(self):
        mock_session = AsyncMock()

        quiz_result = {
            "questions": [
                {"topic": "NLP", "question": "什么是分词?", "choices": ["A. x", "B. y", "C. z", "D. w"], "correct_answer": "A", "explanation": "..."},
            ],
            "topics": ["NLP"],
        }

        with patch("app.agent.nodes.orchestrator.run_quiz_agent", new_callable=AsyncMock) as mock_quiz:
            mock_quiz.return_value = quiz_result

            state: OverallState = {
                "plan": [{"step": 1, "intent": "quiz", "params": {"topics": ["NLP"]}, "depends_on": []}],
                "user_query": "来几道NLP的题",
                "user_id": "test",
            }

            result = await execute_node(state, mock_session)

            assert len(result["step_results"]) == 1
            assert result["step_results"][0]["intent"] == "quiz"
            mock_quiz.assert_called_once_with(
                session=mock_session,
                user_id="test",
                topics=["NLP"],
                prior_context=None,
            )

    @pytest.mark.asyncio
    async def test_quiz_step_with_prior_qa_context(self):
        """Quiz step depending on QA should receive prior context."""
        mock_session = AsyncMock()

        quiz_result = {
            "questions": [{"topic": "Attention", "question": "Q?", "choices": ["A", "B", "C", "D"], "correct_answer": "A", "explanation": "E"}],
            "topics": ["Attention"],
        }

        rewrite_response = json.dumps({"topics": ["Attention"]})

        with patch("app.agent.nodes.orchestrator.run_qa_agent", new_callable=AsyncMock) as mock_qa, \
             patch("app.agent.nodes.orchestrator.run_quiz_agent", new_callable=AsyncMock) as mock_quiz, \
             patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:

            mock_qa.return_value = {"answer": "Attention是注意力机制...", "confidence": "high", "plan": {}, "sources": []}
            mock_quiz.return_value = quiz_result
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(rewrite_response))

            state: OverallState = {
                "plan": [
                    {"step": 1, "intent": "qa", "params": {"query": "什么是Attention"}, "depends_on": []},
                    {"step": 2, "intent": "quiz", "params": {"topics": ["Attention"]}, "depends_on": [1]},
                ],
                "user_query": "讲讲Attention然后出题",
                "user_id": "test",
            }

            result = await execute_node(state, mock_session)

            assert len(result["step_results"]) == 2
            # Quiz should have received prior context
            mock_quiz.assert_called_once()
            call_kwargs = mock_quiz.call_args[1]
            assert call_kwargs["prior_context"] is not None
            assert "Attention" in call_kwargs["prior_context"]

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

    @pytest.mark.asyncio
    async def test_pure_quiz_flow(self):
        """Test full graph: pure quiz -> plan -> execute -> passthrough -> END."""
        mock_session = AsyncMock()

        plan_response = json.dumps({
            "steps": [
                {"step": 1, "intent": "quiz", "params": {"topics": ["NLP"]}, "depends_on": []}
            ],
        })

        quiz_result = {
            "questions": [
                {"topic": "NLP", "question": "Q1?", "choices": ["A", "B", "C", "D"], "correct_answer": "A", "explanation": "E1"},
                {"topic": "NLP", "question": "Q2?", "choices": ["A", "B", "C", "D"], "correct_answer": "B", "explanation": "E2"},
            ],
            "topics": ["NLP"],
        }

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm, \
             patch("app.agent.nodes.orchestrator.run_quiz_agent", new_callable=AsyncMock) as mock_quiz:

            mock_llm.chat = AsyncMock(return_value=_make_llm_response(plan_response))
            mock_quiz.return_value = quiz_result

            from app.agent.graph import run_orchestrator

            result = await run_orchestrator(
                user_query="来几道NLP的题",
                user_id="test",
                session=mock_session,
            )

            # final_answer should contain the quiz questions JSON
            assert "questions" in result["final_answer"]

    @pytest.mark.asyncio
    async def test_multi_intent_flow(self):
        """Test full graph: gap + qa -> plan -> execute -> synthesize -> END."""
        mock_session = AsyncMock()

        plan_response = json.dumps({
            "steps": [
                {"step": 1, "intent": "gap_analysis", "params": {"scope": None, "keyword": "NLP"}, "depends_on": []},
                {"step": 2, "intent": "qa", "params": {"query": "讲解最弱概念"}, "depends_on": [1]},
            ],
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

    @pytest.mark.asyncio
    async def test_gap_then_quiz_flow(self):
        """Test full graph: gap + quiz -> plan -> execute -> synthesize -> END."""
        mock_session = AsyncMock()

        plan_response = json.dumps({
            "steps": [
                {"step": 1, "intent": "gap_analysis", "params": {"scope": None, "keyword": "NLP"}, "depends_on": []},
                {"step": 2, "intent": "quiz", "params": {"topics": []}, "depends_on": [1]},
            ],
        })

        quiz_result = {
            "questions": [
                {"topic": "Attention", "question": "Q?", "choices": ["A", "B", "C", "D"], "correct_answer": "A", "explanation": "E"},
            ],
            "topics": ["Attention"],
        }

        call_count = {"n": 0}

        async def mock_chat(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_llm_response(plan_response)
            else:
                return _make_llm_response("你的NLP盲区在Attention，以下是测验题目...")

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm, \
             patch("app.agent.nodes.orchestrator.run_gap_agent", new_callable=AsyncMock) as mock_gap, \
             patch("app.agent.nodes.orchestrator.run_quiz_agent", new_callable=AsyncMock) as mock_quiz:

            mock_llm.chat = mock_chat
            mock_gap.return_value = {"report": "盲区: Attention"}
            mock_quiz.return_value = quiz_result

            from app.agent.graph import run_orchestrator

            result = await run_orchestrator(
                user_query="分析NLP盲区然后出题考我",
                user_id="test",
                session=mock_session,
            )

            assert len(result["step_results"]) == 2
            mock_gap.assert_called_once()
            mock_quiz.assert_called_once()
            # Quiz should have received prior context from gap
            quiz_call_kwargs = mock_quiz.call_args[1]
            assert quiz_call_kwargs["prior_context"] is not None
