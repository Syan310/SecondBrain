"""Tests for Orchestrator: plan_node, execute_step, replan_node, synthesize_node, graph routing.

These tests mock the LLM and SubAgent calls to validate orchestration logic
without requiring external services (FlagEmbedding, Milvus, etc.).

Architecture (Replan loop):
  START → memory → plan → route_after_plan
    ├── no_steps → empty_response → END
    └── has_steps → execute_step → replan → route
                        ▲                    ├── execute → ──┘
                        └────────────────────
                                             └── finish → route_to_output
                                                            ├── single → passthrough → END
                                                            └── multi  → synthesize → END
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

# Create mock for memory_manager (graph.py now imports it)
_mock_memory = MagicMock()
_mock_memory.load_memory_context = AsyncMock(return_value="")
_mock_memory.save_conversation = AsyncMock()

# Block the entire import chains
for mod_name in [
    # Third-party heavy deps (do NOT mock aiosqlite — it corrupts real DB tests)
    "FlagEmbedding", "pymilvus", "transformers", "docx", "torch",
    "magic_pdf", "magic_pdf.data", "magic_pdf.data.data_reader_writer",
    "magic_pdf.pipe", "magic_pdf.pipe.UNIPipe",
    # App service layer
    "app.services.embedding", "app.services.reranker",
    "app.services.milvus_client", "app.services.doc_processor",
    # App DB layer
    "app.db", "app.db.models", "app.db.crud", "app.db.database",
    # App tools layer
    "app.agent.tools", "app.agent.tools.rag_search",
    "app.agent.tools.query_decompose", "app.agent.tools.read_mastery",
    "app.agent.tools.update_mastery", "app.agent.tools.search_concepts",
    # App subagents
    "app.agent.subagents",
]:
    sys.modules[mod_name] = MagicMock()

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
            assert result["current_step_index"] == 0
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
    async def test_quiz_as_last_step(self):
        plan_response = json.dumps({
            "steps": [
                {"step": 1, "intent": "qa", "params": {"query": "什么是NLP?"}, "depends_on": []},
                {"step": 2, "intent": "quiz", "params": {"topics": ["NLP"]}, "depends_on": [1]},
            ],
        })

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(plan_response))

            state: OverallState = {"user_query": "讲讲NLP然后出题", "user_id": "test"}
            result = await plan_node(state)

            assert len(result["plan"]) == 2
            assert result["plan"][-1]["intent"] == "quiz"

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


# ── execute_step tests ───────────────────────────────────────


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
            mock_qa.assert_called_once_with(query="什么是Transformer?", user_id="test")

    @pytest.mark.asyncio
    async def test_single_gap_step(self):
        mock_session = AsyncMock()

        with patch("app.agent.nodes.orchestrator.run_gap_agent", new_callable=AsyncMock) as mock_gap:
            mock_gap.return_value = {"report": "NLP诊断报告：你在Attention方面较弱"}

            state: OverallState = {
                "plan": [{"step": 1, "intent": "gap_analysis", "params": {"scope": None, "keyword": "NLP"}, "depends_on": []}],
                "current_step_index": 0,
                "step_results": [],
                "user_query": "分析NLP盲区",
                "user_id": "test",
            }

            result = await execute_step(state, mock_session)

            assert len(result["step_results"]) == 1
            assert result["step_results"][0]["intent"] == "gap_analysis"
            assert result["current_step_index"] == 1

    @pytest.mark.asyncio
    async def test_step_with_dependency_rewrites_params(self):
        """Test that execute_step rewrites params using prior step results."""
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
                "current_step_index": 1,  # executing step 2
                "step_results": [
                    {"step": 1, "intent": "gap_analysis", "result": "最弱主题: Attention"},
                ],
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
            "current_step_index": 1,  # beyond plan length
            "step_results": [{"step": 1, "intent": "qa", "result": "done"}],
            "user_query": "test",
            "user_id": "test",
        }

        result = await execute_step(state, mock_session)
        assert len(result["step_results"]) == 1  # unchanged


# ── replan_node tests ────────────────────────────────────────


class TestReplanNode:
    @pytest.mark.asyncio
    async def test_continue_action(self):
        replan_response = json.dumps({
            "action": "continue",
            "revised_steps": [],
            "reason": "按计划继续",
        })

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(replan_response))

            state: OverallState = {
                "plan": [
                    {"step": 1, "intent": "gap_analysis", "params": {}, "depends_on": []},
                    {"step": 2, "intent": "qa", "params": {"query": "讲解"}, "depends_on": [1]},
                ],
                "current_step_index": 1,
                "step_results": [{"step": 1, "intent": "gap_analysis", "result": "诊断完成"}],
                "user_query": "分析盲区然后讲讲",
                "replan_count": 0,
            }

            result = await replan_node(state)
            assert result["replan_action"] == "continue"
            assert result["replan_count"] == 1

    @pytest.mark.asyncio
    async def test_finish_action(self):
        replan_response = json.dumps({
            "action": "finish",
            "revised_steps": [],
            "reason": "所有步骤已执行完毕",
        })

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(replan_response))

            state: OverallState = {
                "plan": [{"step": 1, "intent": "qa", "params": {}, "depends_on": []}],
                "current_step_index": 1,
                "step_results": [{"step": 1, "intent": "qa", "result": "回答完毕"}],
                "user_query": "什么是Transformer?",
                "replan_count": 0,
            }

            result = await replan_node(state)
            assert result["replan_action"] == "finish"

    @pytest.mark.asyncio
    async def test_revise_action_replaces_remaining_plan(self):
        replan_response = json.dumps({
            "action": "revise",
            "revised_steps": [
                {"step": 2, "intent": "qa", "params": {"query": "讲解Attention"}, "depends_on": [1]},
                {"step": 3, "intent": "qa", "params": {"query": "讲解LSTM"}, "depends_on": [1]},
            ],
            "reason": "发现多个盲区，追加讲解步骤",
        })

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response(replan_response))

            state: OverallState = {
                "plan": [
                    {"step": 1, "intent": "gap_analysis", "params": {}, "depends_on": []},
                    {"step": 2, "intent": "qa", "params": {"query": "讲解最弱概念"}, "depends_on": [1]},
                ],
                "current_step_index": 1,
                "step_results": [{"step": 1, "intent": "gap_analysis", "result": "盲区: Attention, LSTM"}],
                "user_query": "分析盲区然后讲讲",
                "replan_count": 0,
            }

            result = await replan_node(state)
            assert result["replan_action"] == "revise"
            assert len(result["plan"]) == 3  # 1 completed + 2 revised
            assert result["plan"][0]["intent"] == "gap_analysis"  # kept
            assert result["plan"][1]["params"]["query"] == "讲解Attention"

    @pytest.mark.asyncio
    async def test_max_replan_forces_finish(self):
        """When replan_count >= MAX_REPLAN_COUNT, auto-finish without LLM call."""
        state: OverallState = {
            "plan": [{"step": 1, "intent": "qa", "params": {}, "depends_on": []}],
            "current_step_index": 0,
            "step_results": [],
            "user_query": "test",
            "replan_count": 3,  # at limit
        }

        # Should NOT call LLM
        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock()
            result = await replan_node(state)
            assert result["replan_action"] == "finish"
            mock_llm.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_parse_failure_defaults_to_continue(self):
        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=_make_llm_response("这不是JSON"))

            state: OverallState = {
                "plan": [{"step": 1, "intent": "qa", "params": {}, "depends_on": []}],
                "current_step_index": 1,
                "step_results": [{"step": 1, "intent": "qa", "result": "done"}],
                "user_query": "test",
                "replan_count": 0,
            }

            result = await replan_node(state)
            assert result["replan_action"] == "continue"


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
        """Test full graph: single QA → plan → execute → replan(finish) → passthrough → END."""
        mock_session = AsyncMock()

        plan_response = json.dumps({
            "steps": [{"step": 1, "intent": "qa", "params": {"query": "什么是Transformer?"}, "depends_on": []}],
        })
        replan_response = json.dumps({
            "action": "finish", "revised_steps": [], "reason": "所有步骤已执行完毕",
        })

        call_count = {"n": 0}

        async def mock_chat(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_llm_response(plan_response)
            elif call_count["n"] == 2:
                return _make_llm_response(replan_response)
            else:
                return _make_llm_response("unexpected call")

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
    async def test_quiz_only_flow(self):
        """Test full graph: pure quiz → plan → execute → replan(finish) → passthrough → END."""
        mock_session = AsyncMock()

        plan_response = json.dumps({
            "steps": [{"step": 1, "intent": "quiz", "params": {"topics": ["NLP"]}, "depends_on": []}],
        })
        replan_response = json.dumps({
            "action": "finish", "revised_steps": [], "reason": "Quiz完成",
        })

        call_count = {"n": 0}

        async def mock_chat(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_llm_response(plan_response)
            else:
                return _make_llm_response(replan_response)

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm, \
             patch("app.agent.nodes.orchestrator.run_quiz_agent", new_callable=AsyncMock) as mock_quiz, \
             patch("app.agent.graph.load_memory_context", new_callable=AsyncMock, return_value="") as _, \
             patch("app.agent.graph.save_conversation", new_callable=AsyncMock) as _:

            mock_llm.chat = mock_chat
            mock_quiz.return_value = {"questions": [{"topic": "NLP", "question": "Q1"}], "session_complete": True}

            from app.agent.graph import run_orchestrator

            result = await run_orchestrator(
                user_query="来几道NLP的题",
                user_id="test",
                session=mock_session,
            )

            assert "final_answer" in result
            mock_quiz.assert_called_once()

    @pytest.mark.asyncio
    async def test_multi_intent_flow(self):
        """Test full graph: gap + qa → plan → execute(2 steps) → replan → synthesize → END."""
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
        synthesize_response = "综合分析：你的NLP盲区在Attention。Attention是一种注意力机制..."

        # LLM calls: 1=plan, 2=replan(continue after step1), 3=rewrite(step2 params), 4=replan(finish after step2), 5=synthesize
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
                return _make_llm_response(synthesize_response)

        with patch("app.agent.nodes.orchestrator.llm_client") as mock_llm, \
             patch("app.agent.nodes.orchestrator.run_gap_agent", new_callable=AsyncMock) as mock_gap, \
             patch("app.agent.nodes.orchestrator.run_qa_agent", new_callable=AsyncMock) as mock_qa, \
             patch("app.agent.graph.load_memory_context", new_callable=AsyncMock, return_value="") as _, \
             patch("app.agent.graph.save_conversation", new_callable=AsyncMock) as _:

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
