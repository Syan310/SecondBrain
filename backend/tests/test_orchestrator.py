"""Tests for Orchestrator: plan_node, execute_step, replan_node, synthesize_node, graph routing.

These tests mock the LLM and SubAgent calls to validate orchestration logic
without requiring external services (FlagEmbedding, Milvus, etc.).
"""

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Pre-mock the entire subagent + service chain ────────────
# We inject mocks into sys.modules so that importing orchestrator.py succeeds
# without heavy ML/DB dependencies. After importing, we CLEAN UP to avoid
# polluting other test files in the same pytest session.

_mock_qa = MagicMock()
_mock_qa.run_qa_agent = AsyncMock()
_mock_gap = MagicMock()
_mock_gap.run_gap_agent = AsyncMock()
_mock_quiz = MagicMock()
_mock_quiz.run_quiz_agent = AsyncMock()

_mock_memory = MagicMock()
_mock_memory.load_memory_context = AsyncMock(return_value="")
_mock_memory.save_conversation = AsyncMock()

_MOCKED_MODULE_NAMES = [
    "FlagEmbedding", "pymilvus", "transformers", "docx", "torch",
    "magic_pdf", "magic_pdf.data", "magic_pdf.data.data_reader_writer",
    "magic_pdf.pipe", "magic_pdf.pipe.UNIPipe",
    "aiosqlite",
    "app.services.embedding", "app.services.reranker",
    "app.services.milvus_client", "app.services.doc_processor",
    "app.db", "app.db.models", "app.db.crud", "app.db.database",
    "app.agent.tools", "app.agent.tools.rag_search",
    "app.agent.tools.query_decompose", "app.agent.tools.read_mastery",
    "app.agent.tools.update_mastery", "app.agent.tools.search_concepts",
    "app.agent.subagents",
    "app.agent.subagents.qa_agent",
    "app.agent.subagents.gap_agent",
    "app.agent.subagents.quiz_agent",
    "app.agent.nodes.memory_manager",
]

# Save original state so we can restore after importing
_originals = {k: sys.modules[k] for k in _MOCKED_MODULE_NAMES if k in sys.modules}

for mod_name in _MOCKED_MODULE_NAMES:
    sys.modules[mod_name] = MagicMock()

# Override specific subagent mocks with controlled instances
sys.modules["app.agent.subagents.qa_agent"] = _mock_qa
sys.modules["app.agent.subagents.gap_agent"] = _mock_gap
sys.modules["app.agent.subagents.quiz_agent"] = _mock_quiz
sys.modules["app.agent.nodes.memory_manager"] = _mock_memory

from app.agent.state import OverallState  # noqa: E402
from app.agent.nodes.orchestrator import (  # noqa: E402
    plan_node,
    execute_step,
    replan_node,
    synthesize_node,
    passthrough_node,
    route_after_plan,
    route_after_replan,
    route_to_output,
    MAX_REPLAN_COUNT,
    _parse_json,
)

# ── Cleanup: restore sys.modules so other test files see real modules ──
for mod_name in _MOCKED_MODULE_NAMES:
    if mod_name in _originals:
        sys.modules[mod_name] = _originals[mod_name]
    else:
        sys.modules.pop(mod_name, None)


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
            assert result["current_step_index"] == 0
            assert result["step_results"] == []
            assert result["replan_count"] == 0

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

    @pytest.mark.asyncio
    async def test_pure_quiz_intent(self):
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

            assert len(result["plan"]) == 1
            assert result["plan"][0]["intent"] == "qa"

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

            intents = [s["intent"] for s in result["plan"]]
            assert "unknown_intent" not in intents

    @pytest.mark.asyncio
    async def test_memory_context_injected(self):
        """Test that memory_context is appended to the system prompt."""
        plan_response = json.dumps({
            "steps": [
                {"step": 1, "intent": "qa", "params": {"query": "继续讲讲"}, "depends_on": []}
            ],
        })

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(plan_response))

            state: OverallState = {
                "user_query": "继续讲讲",
                "user_id": "test",
                "memory_context": "【短期记忆】用户之前问了Transformer的原理",
            }
            result = await plan_node(state)

            # Verify memory was included in the system prompt
            call_args = mock_llm.chat.call_args
            messages = call_args.kwargs.get("messages", call_args.args[0] if call_args.args else [])
            system_msg = messages[0]["content"]
            assert "用户记忆上下文" in system_msg
            assert "Transformer" in system_msg


# ── execute_step tests ──────────────────────────────────────


class TestExecuteStep:
    @pytest.mark.asyncio
    async def test_single_qa_step(self):
        mock_session = AsyncMock()

        with patch("app.agent.nodes.orchestrator.run_qa_agent", new_callable=AsyncMock) as mock_qa:
            mock_qa.return_value = {"answer": "Transformer是一种模型架构", "confidence": "high", "plan": {}, "sources": []}

            state: OverallState = {
                "plan": [{"step": 1, "intent": "qa", "params": {"query": "什么是Transformer?"}, "depends_on": []}],
                "current_step_index": 0,
                "step_results": [],
                "user_query": "什么是Transformer?",
                "user_id": "test",
            }

            result = await execute_step(state, mock_session)

            assert len(result["step_results"]) == 1
            assert result["step_results"][0]["intent"] == "qa"
            assert "Transformer" in result["step_results"][0]["result"]
            assert result["current_step_index"] == 1

    @pytest.mark.asyncio
    async def test_quiz_step_dispatches_to_quiz_agent(self):
        mock_session = AsyncMock()
        quiz_result = {"questions": [{"topic": "NLP", "question": "Q?"}], "topics": ["NLP"]}

        with patch("app.agent.nodes.orchestrator.run_quiz_agent", new_callable=AsyncMock) as mock_quiz:
            mock_quiz.return_value = quiz_result

            state: OverallState = {
                "plan": [{"step": 1, "intent": "quiz", "params": {"topics": ["NLP"]}, "depends_on": []}],
                "current_step_index": 0,
                "step_results": [],
                "user_query": "来几道NLP的题",
                "user_id": "test",
            }

            result = await execute_step(state, mock_session)

            assert len(result["step_results"]) == 1
            assert result["step_results"][0]["intent"] == "quiz"
            mock_quiz.assert_called_once_with(
                session=mock_session, user_id="test", topics=["NLP"], prior_context=None,
            )

    @pytest.mark.asyncio
    async def test_second_step_with_dependency_rewrite(self):
        mock_session = AsyncMock()
        rewrite_response = json.dumps({"query": "讲解Attention机制"})

        with patch("app.agent.nodes.orchestrator.run_qa_agent", new_callable=AsyncMock) as mock_qa, \
             patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:

            mock_qa.return_value = {"answer": "Attention是...", "confidence": "high", "plan": {}, "sources": []}
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(rewrite_response))

            state: OverallState = {
                "plan": [
                    {"step": 1, "intent": "gap_analysis", "params": {"scope": None}, "depends_on": []},
                    {"step": 2, "intent": "qa", "params": {"query": "讲解最弱概念"}, "depends_on": [1]},
                ],
                "current_step_index": 1,
                "step_results": [{"step": 1, "intent": "gap_analysis", "result": "最弱主题: Attention"}],
                "user_query": "分析盲区然后讲讲",
                "user_id": "test",
            }

            result = await execute_step(state, mock_session)

            assert len(result["step_results"]) == 2
            assert result["current_step_index"] == 2
            mock_qa.assert_called_once_with(query="讲解Attention机制", user_id="test")

    @pytest.mark.asyncio
    async def test_index_beyond_plan_is_noop(self):
        mock_session = AsyncMock()

        state: OverallState = {
            "plan": [{"step": 1, "intent": "qa", "params": {"query": "test"}, "depends_on": []}],
            "current_step_index": 1,
            "step_results": [{"step": 1, "intent": "qa", "result": "done"}],
            "user_query": "test",
            "user_id": "test",
        }

        result = await execute_step(state, mock_session)
        assert len(result["step_results"]) == 1
        assert result["current_step_index"] == 1


# ── replan_node tests ────────────────────────────────────────


class TestReplanNode:
    @pytest.mark.asyncio
    async def test_all_steps_done_llm_decides(self):
        replan_response = json.dumps({"action": "finish", "revised_steps": [], "reason": "完成"})

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(replan_response))

            state: OverallState = {
                "plan": [{"step": 1, "intent": "qa", "params": {"query": "test"}, "depends_on": []}],
                "current_step_index": 1,
                "step_results": [{"step": 1, "intent": "qa", "result": "answer"}],
                "user_query": "test",
                "replan_count": 0,
            }

            result = await replan_node(state)
            assert result["replan_action"] == "finish"
            mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_max_replan_count_forces_finish(self):
        state: OverallState = {
            "plan": [
                {"step": 1, "intent": "qa", "params": {"query": "test"}, "depends_on": []},
                {"step": 2, "intent": "qa", "params": {"query": "test2"}, "depends_on": []},
            ],
            "current_step_index": 1,
            "step_results": [{"step": 1, "intent": "qa", "result": "answer"}],
            "user_query": "test",
            "replan_count": MAX_REPLAN_COUNT,
        }

        result = await replan_node(state)
        assert result["replan_action"] == "finish"

    @pytest.mark.asyncio
    async def test_continue_action(self):
        replan_response = json.dumps({"action": "continue", "revised_steps": [], "reason": "继续"})

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(replan_response))

            state: OverallState = {
                "plan": [
                    {"step": 1, "intent": "gap_analysis", "params": {"scope": None}, "depends_on": []},
                    {"step": 2, "intent": "qa", "params": {"query": "讲解"}, "depends_on": [1]},
                ],
                "current_step_index": 1,
                "step_results": [{"step": 1, "intent": "gap_analysis", "result": "盲区: Attention"}],
                "user_query": "分析盲区然后讲讲",
                "replan_count": 0,
            }

            result = await replan_node(state)
            assert result["replan_action"] == "continue"
            assert result["replan_count"] == 1
            assert "plan" not in result

    @pytest.mark.asyncio
    async def test_revise_action_replaces_remaining_plan(self):
        replan_response = json.dumps({
            "action": "revise",
            "revised_steps": [
                {"step": 2, "intent": "qa", "params": {"query": "讲解Attention"}, "depends_on": [1]},
                {"step": 3, "intent": "qa", "params": {"query": "讲解LSTM"}, "depends_on": [1]},
            ],
            "reason": "发现2个盲区",
        })

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(replan_response))

            state: OverallState = {
                "plan": [
                    {"step": 1, "intent": "gap_analysis", "params": {"scope": None}, "depends_on": []},
                    {"step": 2, "intent": "qa", "params": {"query": "讲解最弱概念"}, "depends_on": [1]},
                ],
                "current_step_index": 1,
                "step_results": [{"step": 1, "intent": "gap_analysis", "result": "盲区: Attention, LSTM"}],
                "user_query": "分析盲区然后讲讲",
                "replan_count": 0,
            }

            result = await replan_node(state)
            assert result["replan_action"] == "revise"
            assert len(result["plan"]) == 3
            assert result["plan"][0]["intent"] == "gap_analysis"  # preserved
            assert result["plan"][1]["params"]["query"] == "讲解Attention"
            assert result["plan"][2]["params"]["query"] == "讲解LSTM"

    @pytest.mark.asyncio
    async def test_finish_action(self):
        replan_response = json.dumps({"action": "finish", "revised_steps": [], "reason": "已充分"})

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(replan_response))

            state: OverallState = {
                "plan": [
                    {"step": 1, "intent": "qa", "params": {"query": "test"}, "depends_on": []},
                    {"step": 2, "intent": "gap_analysis", "params": {"scope": None}, "depends_on": []},
                ],
                "current_step_index": 1,
                "step_results": [{"step": 1, "intent": "qa", "result": "完整答案"}],
                "user_query": "test",
                "replan_count": 0,
            }

            result = await replan_node(state)
            assert result["replan_action"] == "finish"

    @pytest.mark.asyncio
    async def test_invalid_action_defaults_to_continue(self):
        replan_response = json.dumps({"action": "invalid", "revised_steps": [], "reason": ""})

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(replan_response))

            state: OverallState = {
                "plan": [
                    {"step": 1, "intent": "qa", "params": {"query": "test"}, "depends_on": []},
                    {"step": 2, "intent": "qa", "params": {"query": "test2"}, "depends_on": []},
                ],
                "current_step_index": 1,
                "step_results": [{"step": 1, "intent": "qa", "result": "done"}],
                "user_query": "test",
                "replan_count": 0,
            }

            result = await replan_node(state)
            assert result["replan_action"] == "continue"

    @pytest.mark.asyncio
    async def test_parse_failure_defaults_to_continue(self):
        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response("不是JSON"))

            state: OverallState = {
                "plan": [
                    {"step": 1, "intent": "qa", "params": {"query": "test"}, "depends_on": []},
                    {"step": 2, "intent": "qa", "params": {"query": "test2"}, "depends_on": []},
                ],
                "current_step_index": 1,
                "step_results": [{"step": 1, "intent": "qa", "result": "done"}],
                "user_query": "test",
                "replan_count": 0,
            }

            result = await replan_node(state)
            assert result["replan_action"] == "continue"
            assert result["replan_count"] == 1

    @pytest.mark.asyncio
    async def test_revise_with_invalid_steps_falls_back_to_finish(self):
        replan_response = json.dumps({
            "action": "revise",
            "revised_steps": [
                {"step": 2, "intent": "unknown", "params": {}, "depends_on": []},
            ],
            "reason": "bad",
        })

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(replan_response))

            state: OverallState = {
                "plan": [{"step": 1, "intent": "qa", "params": {"query": "test"}, "depends_on": []}],
                "current_step_index": 1,
                "step_results": [{"step": 1, "intent": "qa", "result": "done"}],
                "user_query": "test",
                "replan_count": 0,
            }

            result = await replan_node(state)
            assert result["replan_action"] == "finish"


# ── synthesize_node / passthrough_node tests ─────────────────


class TestSynthesizeNode:
    @pytest.mark.asyncio
    async def test_synthesize_merges_results(self):
        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(
                return_value=_make_llm_response("综合结果：盲区在Attention，以下是讲解...")
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

    def test_route_after_replan_continue(self):
        state: OverallState = {"replan_action": "continue"}
        assert route_after_replan(state) == "execute"

    def test_route_after_replan_revise(self):
        state: OverallState = {"replan_action": "revise"}
        assert route_after_replan(state) == "execute"

    def test_route_after_replan_finish(self):
        state: OverallState = {"replan_action": "finish"}
        assert route_after_replan(state) == "finish"

    def test_route_to_output_single(self):
        state: OverallState = {"step_results": [{"step": 1}]}
        assert route_to_output(state) == "single"

    def test_route_to_output_multi(self):
        state: OverallState = {"step_results": [{"step": 1}, {"step": 2}]}
        assert route_to_output(state) == "multi"


# ── Integration: graph build + invoke ────────────────────────


class TestGraphIntegration:
    @pytest.mark.asyncio
    async def test_single_qa_flow(self):
        """Single QA: plan → execute_step → replan(finish) → passthrough → END."""
        mock_session = AsyncMock()

        plan_response = json.dumps({
            "steps": [{"step": 1, "intent": "qa", "params": {"query": "什么是Transformer?"}, "depends_on": []}],
        })
        replan_finish = json.dumps({"action": "finish", "revised_steps": [], "reason": "已完成"})

        call_count = {"n": 0}

        async def mock_chat(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_llm_response(plan_response)
            else:
                return _make_llm_response(replan_finish)

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm, \
             patch("app.agent.nodes.orchestrator.run_qa_agent", new_callable=AsyncMock) as mock_qa, \
             patch("app.agent.graph.load_memory_context", new_callable=AsyncMock, return_value="") as _, \
             patch("app.agent.graph.save_conversation", new_callable=AsyncMock) as _:

            mock_llm.chat = mock_chat
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
        """Pure quiz: plan → execute_step → replan(finish) → passthrough → END."""
        mock_session = AsyncMock()

        plan_response = json.dumps({
            "steps": [{"step": 1, "intent": "quiz", "params": {"topics": ["NLP"]}, "depends_on": []}],
        })
        replan_finish = json.dumps({"action": "finish", "revised_steps": [], "reason": "已完成"})

        quiz_result = {
            "questions": [{"topic": "NLP", "question": "Q1?", "choices": ["A", "B", "C", "D"], "correct_answer": "A", "explanation": "E1"}],
            "topics": ["NLP"],
        }

        call_count = {"n": 0}

        async def mock_chat(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_llm_response(plan_response)
            else:
                return _make_llm_response(replan_finish)

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm, \
             patch("app.agent.nodes.orchestrator.run_quiz_agent", new_callable=AsyncMock) as mock_quiz, \
             patch("app.agent.graph.load_memory_context", new_callable=AsyncMock, return_value="") as _, \
             patch("app.agent.graph.save_conversation", new_callable=AsyncMock) as _:

            mock_llm.chat = mock_chat
            mock_quiz.return_value = quiz_result

            from app.agent.graph import run_orchestrator

            result = await run_orchestrator(
                user_query="来几道NLP的题",
                user_id="test",
                session=mock_session,
            )

            assert "questions" in result["final_answer"]

    @pytest.mark.asyncio
    async def test_multi_intent_flow(self):
        """Gap + QA: plan → exec1 → replan(continue) → exec2 → replan(finish) → synthesize."""
        mock_session = AsyncMock()

        plan_response = json.dumps({
            "steps": [
                {"step": 1, "intent": "gap_analysis", "params": {"scope": None, "keyword": "NLP"}, "depends_on": []},
                {"step": 2, "intent": "qa", "params": {"query": "讲解最弱概念"}, "depends_on": [1]},
            ],
        })

        rewrite_response = json.dumps({"query": "什么是Attention机制"})
        replan_continue = json.dumps({"action": "continue", "revised_steps": [], "reason": "继续"})
        replan_finish = json.dumps({"action": "finish", "revised_steps": [], "reason": "完成"})

        # LLM calls: 1) plan, 2) replan-continue, 3) rewrite, 4) replan-finish, 5) synthesize
        call_count = {"n": 0}

        async def mock_chat(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_llm_response(plan_response)
            elif call_count["n"] == 2:
                return _make_llm_response(replan_continue)
            elif call_count["n"] == 3:
                return _make_llm_response(rewrite_response)
            elif call_count["n"] == 4:
                return _make_llm_response(replan_finish)
            else:
                return _make_llm_response("综合：盲区在Attention。Attention是注意力机制...")

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm, \
             patch("app.agent.nodes.orchestrator.run_gap_agent", new_callable=AsyncMock) as mock_gap, \
             patch("app.agent.nodes.orchestrator.run_qa_agent", new_callable=AsyncMock) as mock_qa, \
             patch("app.agent.graph.load_memory_context", new_callable=AsyncMock, return_value="") as _, \
             patch("app.agent.graph.save_conversation", new_callable=AsyncMock) as _:

            mock_llm.chat = mock_chat
            mock_gap.return_value = {"report": "盲区: Attention"}
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
        """Gap + Quiz: plan → exec gap → replan(continue) → exec quiz → replan(finish) → synthesize."""
        mock_session = AsyncMock()

        plan_response = json.dumps({
            "steps": [
                {"step": 1, "intent": "gap_analysis", "params": {"scope": None, "keyword": "NLP"}, "depends_on": []},
                {"step": 2, "intent": "quiz", "params": {"topics": []}, "depends_on": [1]},
            ],
        })

        quiz_result = {"questions": [{"topic": "Attention", "question": "Q?"}], "topics": ["Attention"]}
        replan_continue = json.dumps({"action": "continue", "revised_steps": [], "reason": "继续"})
        replan_finish = json.dumps({"action": "finish", "revised_steps": [], "reason": "完成"})

        call_count = {"n": 0}

        async def mock_chat(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_llm_response(plan_response)
            elif call_count["n"] == 2:
                return _make_llm_response(replan_continue)
            elif call_count["n"] == 3:
                return _make_llm_response(replan_finish)
            else:
                return _make_llm_response("盲区在Attention，以下是测验...")

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm, \
             patch("app.agent.nodes.orchestrator.run_gap_agent", new_callable=AsyncMock) as mock_gap, \
             patch("app.agent.nodes.orchestrator.run_quiz_agent", new_callable=AsyncMock) as mock_quiz, \
             patch("app.agent.graph.load_memory_context", new_callable=AsyncMock, return_value="") as _, \
             patch("app.agent.graph.save_conversation", new_callable=AsyncMock) as _:

            mock_llm.chat = mock_chat
            mock_gap.return_value = {"report": "盲区: Attention"}
            mock_quiz.return_value = quiz_result

            from app.agent.graph import run_orchestrator

            result = await run_orchestrator(
                user_query="分析NLP盲区然后出题",
                user_id="test",
                session=mock_session,
            )

            assert len(result["step_results"]) == 2
            mock_gap.assert_called_once()
            mock_quiz.assert_called_once()
            quiz_call_kwargs = mock_quiz.call_args[1]
            assert quiz_call_kwargs["prior_context"] is not None

    @pytest.mark.asyncio
    async def test_replan_revise_flow(self):
        """Replan revise: plan(1 gap) → exec → replan(revise +2 qa) → exec qa1 → replan(continue) → exec qa2 → replan(finish) → synthesize."""
        mock_session = AsyncMock()

        plan_response = json.dumps({
            "steps": [
                {"step": 1, "intent": "gap_analysis", "params": {"scope": None, "keyword": "DL"}, "depends_on": []},
            ],
        })

        replan_revise = json.dumps({
            "action": "revise",
            "revised_steps": [
                {"step": 2, "intent": "qa", "params": {"query": "讲解Attention"}, "depends_on": [1]},
                {"step": 3, "intent": "qa", "params": {"query": "讲解LSTM"}, "depends_on": [1]},
            ],
            "reason": "发现2个盲区",
        })

        rewrite_1 = json.dumps({"query": "讲解Attention"})
        rewrite_2 = json.dumps({"query": "讲解LSTM"})
        replan_continue = json.dumps({"action": "continue", "revised_steps": [], "reason": "继续"})
        replan_finish = json.dumps({"action": "finish", "revised_steps": [], "reason": "完成"})

        # LLM: 1) plan, 2) replan-revise, 3) rewrite-qa1, 4) replan-continue, 5) rewrite-qa2, 6) replan-finish, 7) synthesize
        call_count = {"n": 0}

        async def mock_chat(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_llm_response(plan_response)
            elif call_count["n"] == 2:
                return _make_llm_response(replan_revise)
            elif call_count["n"] == 3:
                return _make_llm_response(rewrite_1)
            elif call_count["n"] == 4:
                return _make_llm_response(replan_continue)
            elif call_count["n"] == 5:
                return _make_llm_response(rewrite_2)
            elif call_count["n"] == 6:
                return _make_llm_response(replan_finish)
            else:
                return _make_llm_response("综合：Attention和LSTM讲解...")

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm, \
             patch("app.agent.nodes.orchestrator.run_gap_agent", new_callable=AsyncMock) as mock_gap, \
             patch("app.agent.nodes.orchestrator.run_qa_agent", new_callable=AsyncMock) as mock_qa, \
             patch("app.agent.graph.load_memory_context", new_callable=AsyncMock, return_value="") as _, \
             patch("app.agent.graph.save_conversation", new_callable=AsyncMock) as _:

            mock_llm.chat = mock_chat
            mock_gap.return_value = {"report": "盲区: Attention, LSTM"}
            mock_qa.return_value = {"answer": "讲解内容...", "confidence": "high", "plan": {}, "sources": []}

            from app.agent.graph import run_orchestrator

            result = await run_orchestrator(
                user_query="分析DL盲区然后讲讲",
                user_id="test",
                session=mock_session,
            )

            assert len(result["step_results"]) == 3  # gap + 2 qa
            assert len(result["final_answer"]) > 0
            mock_gap.assert_called_once()
            assert mock_qa.call_count == 2

    @pytest.mark.asyncio
    async def test_replan_finish_early(self):
        """Replan finish early: plan(2 steps) → exec step1 → replan(finish) → passthrough. Step 2 skipped."""
        mock_session = AsyncMock()

        plan_response = json.dumps({
            "steps": [
                {"step": 1, "intent": "qa", "params": {"query": "什么是Transformer?"}, "depends_on": []},
                {"step": 2, "intent": "gap_analysis", "params": {"scope": None}, "depends_on": []},
            ],
        })

        replan_finish = json.dumps({"action": "finish", "revised_steps": [], "reason": "已充分"})

        call_count = {"n": 0}

        async def mock_chat(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_llm_response(plan_response)
            else:
                return _make_llm_response(replan_finish)

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm, \
             patch("app.agent.nodes.orchestrator.run_qa_agent", new_callable=AsyncMock) as mock_qa, \
             patch("app.agent.nodes.orchestrator.run_gap_agent", new_callable=AsyncMock) as mock_gap, \
             patch("app.agent.graph.load_memory_context", new_callable=AsyncMock, return_value="") as _, \
             patch("app.agent.graph.save_conversation", new_callable=AsyncMock) as _:

            mock_llm.chat = mock_chat
            mock_qa.return_value = {"answer": "Transformer是...", "confidence": "high", "plan": {}, "sources": []}

            from app.agent.graph import run_orchestrator

            result = await run_orchestrator(
                user_query="什么是Transformer？顺便看看掌握度",
                user_id="test",
                session=mock_session,
            )

            assert result["final_answer"] == "Transformer是..."
            assert len(result["step_results"]) == 1
            mock_qa.assert_called_once()
            mock_gap.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_single_qa_flow(self):
        """Single QA stream: emits status events, token chunks, and final payload."""
        mock_session = AsyncMock()

        plan_response = json.dumps({
            "steps": [{"step": 1, "intent": "qa", "params": {"query": "什么是Transformer?"}, "depends_on": []}],
        })
        replan_finish = json.dumps({"action": "finish", "revised_steps": [], "reason": "已完成"})

        call_count = {"n": 0}

        async def mock_chat(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_llm_response(plan_response)
            return _make_llm_response(replan_finish)

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm, \
             patch("app.agent.nodes.orchestrator.run_qa_agent", new_callable=AsyncMock) as mock_qa, \
             patch("app.agent.graph.load_memory_context", new_callable=AsyncMock, return_value="") as _, \
             patch("app.agent.graph.save_conversation", new_callable=AsyncMock) as _:

            mock_llm.chat = mock_chat
            mock_qa.return_value = {"answer": "Transformer是一种模型架构", "confidence": "high", "plan": {}, "sources": []}

            from app.agent.graph import run_orchestrator_stream

            events = [event async for event in run_orchestrator_stream(
                user_query="什么是Transformer?",
                user_id="test",
                session=mock_session,
            )]

            status_messages = [e["message"] for e in events if e["type"] == "status"]
            token_text = "".join(e["text"] for e in events if e["type"] == "token")
            final_event = next(e for e in events if e["type"] == "final")

            assert any("规划" in msg for msg in status_messages)
            assert any("问答结果已生成" in msg for msg in status_messages)
            assert "Transformer" in token_text
            assert "Transformer" in final_event["answer"]

    @pytest.mark.asyncio
    async def test_stream_multi_step_uses_streaming_synthesis(self):
        """Multi-step stream: synthesis tokens should come from llm_client.stream_chat."""
        mock_session = AsyncMock()

        plan_response = json.dumps({
            "steps": [
                {"step": 1, "intent": "gap_analysis", "params": {"scope": None, "keyword": "NLP"}, "depends_on": []},
                {"step": 2, "intent": "qa", "params": {"query": "讲解最弱概念"}, "depends_on": [1]},
            ],
        })
        rewrite_response = json.dumps({"query": "讲解Attention机制"})
        replan_continue = json.dumps({"action": "continue", "revised_steps": [], "reason": "继续"})
        replan_finish = json.dumps({"action": "finish", "revised_steps": [], "reason": "完成"})

        call_count = {"n": 0}

        async def mock_chat(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_llm_response(plan_response)
            if call_count["n"] == 2:
                return _make_llm_response(replan_continue)
            if call_count["n"] == 3:
                return _make_llm_response(rewrite_response)
            return _make_llm_response(replan_finish)

        async def mock_stream_chat(**kwargs):
            for chunk in ["综合：", "盲区在Attention。"]:
                yield chunk

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm, \
             patch("app.agent.nodes.orchestrator.run_gap_agent", new_callable=AsyncMock) as mock_gap, \
             patch("app.agent.nodes.orchestrator.run_qa_agent", new_callable=AsyncMock) as mock_qa, \
             patch("app.agent.graph.llm_client.stream_chat", mock_stream_chat), \
             patch("app.agent.graph.load_memory_context", new_callable=AsyncMock, return_value="") as _, \
             patch("app.agent.graph.save_conversation", new_callable=AsyncMock) as _:

            mock_llm.chat = mock_chat
            mock_gap.return_value = {"report": "盲区: Attention"}
            mock_qa.return_value = {"answer": "Attention是注意力机制...", "confidence": "high", "plan": {}, "sources": []}

            from app.agent.graph import run_orchestrator_stream

            events = [event async for event in run_orchestrator_stream(
                user_query="分析NLP盲区然后讲讲最弱的",
                user_id="test",
                session=mock_session,
            )]

            status_messages = [e["message"] for e in events if e["type"] == "status"]
            token_text = "".join(e["text"] for e in events if e["type"] == "token")
            final_event = next(e for e in events if e["type"] == "final")

            assert any("汇总多个子任务结果" in msg for msg in status_messages)
            assert token_text == "综合：盲区在Attention。"
            assert final_event["answer"] == "综合：盲区在Attention。"
