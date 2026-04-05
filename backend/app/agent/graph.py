"""LangGraph main graph: Orchestrator Plan & Execute with Replan loop.

Graph structure:
  START → memory → plan → route_after_plan
    ├── no_steps → empty_response → END
    └── has_steps → execute_step → replan_node → route
                        ▲                          ├── execute → ──┘
                        └──────────────────────────
                                                   └── finish → route_to_output
                                                                  ├── single → passthrough → END
                                                                  └── multi  → synthesize → END

Usage:
    from app.agent.graph import build_graph

    graph = build_graph(session)
    result = await graph.ainvoke({
        "user_query": "...",
        "user_id": "default",
    })
"""

import asyncio
import logging
from collections.abc import AsyncGenerator

from langgraph.graph import StateGraph, END
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.state import OverallState
from app.agent.nodes.orchestrator import (
    plan_node,
    execute_step,
    replan_node,
    synthesize_node,
    passthrough_node,
    route_after_plan,
    route_after_replan,
    route_to_output,
    build_synthesize_messages,
)
from app.agent.nodes.memory_manager import load_memory_context, save_conversation
from app.services.llm_client import llm_client

logger = logging.getLogger(__name__)


def _empty_response(state: OverallState) -> dict:
    """Handle empty plan (e.g., unrecognized intent)."""
    return {"final_answer": "抱歉，我没有理解你的请求。"}


def build_graph(session: AsyncSession) -> StateGraph:
    """Build the Orchestrator LangGraph with Replan loop.

    Args:
        session: Async SQLAlchemy session for DB-dependent SubAgents.

    Returns:
        Compiled LangGraph ready for ainvoke().
    """

    # Wrap memory_node to bind session
    async def _memory_node(state: OverallState) -> dict:
        user_id = state.get("user_id", "default")
        user_query = state.get("user_query", "")
        memory_context = await load_memory_context(
            session=session,
            user_id=user_id,
            user_query=user_query,
        )
        return {"memory_context": memory_context}

    # Wrap execute_step to bind session
    async def _execute_step(state: OverallState) -> dict:
        return await execute_step(state, session)

    graph = StateGraph(OverallState)

    # Add nodes
    graph.add_node("memory", _memory_node)
    graph.add_node("plan", plan_node)
    graph.add_node("execute_step", _execute_step)
    graph.add_node("replan", replan_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("passthrough", passthrough_node)
    graph.add_node("empty_response", _empty_response)

    # Entry point: memory → plan
    graph.set_entry_point("memory")
    graph.add_edge("memory", "plan")

    # plan → route
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "no_steps": "empty_response",
            "has_steps": "execute_step",
        },
    )

    # execute_step → replan (always)
    graph.add_edge("execute_step", "replan")

    # replan → route (3-way: execute / single / multi)
    def _replan_routing(state: OverallState) -> str:
        replan_result = route_after_replan(state)
        if replan_result == "execute":
            return "execute"
        # finish → decide passthrough or synthesize
        return route_to_output(state)

    graph.add_conditional_edges(
        "replan",
        _replan_routing,
        {
            "execute": "execute_step",
            "single": "passthrough",
            "multi": "synthesize",
        },
    )

    # Terminal edges
    graph.add_edge("empty_response", END)
    graph.add_edge("passthrough", END)
    graph.add_edge("synthesize", END)

    return graph.compile()


async def run_orchestrator(
    user_query: str,
    user_id: str,
    session: AsyncSession,
    session_id: str = "default",
) -> dict:
    """Convenience function to run the orchestrator graph.

    Args:
        user_query: The user's input text.
        user_id: User identifier.
        session: Async DB session.
        session_id: Conversation session identifier.

    Returns:
        dict with keys:
          - final_answer (str): The response to show the user.
          - step_results (list[dict]): Individual step results for debugging.
    """
    graph = build_graph(session)

    result = await graph.ainvoke({
        "user_query": user_query,
        "user_id": user_id,
        "messages": [],
    })

    final_answer = result.get("final_answer", "")

    # Save conversation round (user query + assistant answer) to short-term memory
    await save_conversation(
        session=session,
        user_id=user_id,
        session_id=session_id,
        user_message=user_query,
        assistant_message=final_answer,
    )

    return {
        "final_answer": final_answer,
        "step_results": result.get("step_results", []),
    }


def _make_status_event(stage: str, message: str, **extra: object) -> dict:
    event = {"type": "status", "stage": stage, "message": message}
    event.update(extra)
    return event


async def _stream_text_chunks(text: str, chunk_size: int = 24) -> AsyncGenerator[str, None]:
    """Pseudo-stream plain text when upstream only returns a full string."""
    if not text:
        return

    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]
        await asyncio.sleep(0)


def _describe_step_start(step: dict, current: int, total: int) -> str:
    intent = step.get("intent")
    if intent == "qa":
        return f"正在执行第 {current}/{total} 步：检索并回答问题"
    if intent == "gap_analysis":
        return f"正在执行第 {current}/{total} 步：分析知识盲区"
    if intent == "quiz":
        return f"正在执行第 {current}/{total} 步：生成测验题目"
    return f"正在执行第 {current}/{total} 步"


def _describe_step_done(step_result: dict) -> str:
    intent = step_result.get("intent")
    step = step_result.get("step")
    if intent == "qa":
        return f"第 {step} 步已完成：问答结果已生成"
    if intent == "gap_analysis":
        return f"第 {step} 步已完成：盲区分析已生成"
    if intent == "quiz":
        return f"第 {step} 步已完成：测验题目已生成"
    return f"第 {step} 步已完成"


async def run_orchestrator_stream(
    user_query: str,
    user_id: str,
    session: AsyncSession,
    session_id: str = "default",
) -> AsyncGenerator[dict, None]:
    """Run orchestrator while yielding stage events and answer chunks."""
    state: OverallState = {
        "user_query": user_query,
        "user_id": user_id,
        "messages": [],
    }

    yield _make_status_event("memory", "正在读取历史记忆...")
    memory_context = await load_memory_context(
        session=session,
        user_id=user_id,
        user_query=user_query,
    )
    state["memory_context"] = memory_context
    if memory_context:
        yield _make_status_event("memory", "已加载历史记忆，正在结合上下文理解问题")
    else:
        yield _make_status_event("memory", "未命中历史记忆，直接开始规划")

    yield _make_status_event("plan", "正在规划执行步骤...")
    state.update(await plan_node(state))

    if route_after_plan(state) == "no_steps":
        final_answer = _empty_response(state)["final_answer"]
        yield _make_status_event("output", "未识别到有效步骤，直接返回结果")
        async for chunk in _stream_text_chunks(final_answer):
            yield {"type": "token", "text": chunk}

        await save_conversation(
            session=session,
            user_id=user_id,
            session_id=session_id,
            user_message=user_query,
            assistant_message=final_answer,
        )
        yield {"type": "final", "answer": final_answer, "step_results": []}
        return

    plan = state.get("plan", [])
    yield _make_status_event(
        "plan",
        f"执行计划已生成，共 {len(plan)} 步",
        total_steps=len(plan),
    )

    while True:
        current_index = state.get("current_step_index", 0)
        plan = state.get("plan", [])

        if current_index < len(plan):
            current_step = plan[current_index]
            yield _make_status_event(
                "execute",
                _describe_step_start(current_step, current_index + 1, len(plan)),
                step=current_step.get("step"),
                intent=current_step.get("intent"),
            )
            state.update(await execute_step(state, session))
            latest_step = state.get("step_results", [])[-1]
            yield _make_status_event(
                "execute",
                _describe_step_done(latest_step),
                step=latest_step.get("step"),
                intent=latest_step.get("intent"),
            )

        yield _make_status_event("replan", "正在检查是否需要调整后续步骤...")
        replan_update = await replan_node(state)
        state.update(replan_update)

        action = state.get("replan_action", "finish")
        if action == "continue":
            yield _make_status_event("replan", "后续步骤无需调整，继续执行")
        elif action == "revise":
            remaining = max(len(state.get("plan", [])) - state.get("current_step_index", 0), 0)
            yield _make_status_event("replan", f"已调整执行计划，剩余 {remaining} 步待执行")
        else:
            yield _make_status_event("replan", "执行阶段完成，正在整理最终回答")

        if route_after_replan(state) == "execute":
            continue

        output_route = route_to_output(state)
        step_results = state.get("step_results", [])
        final_answer = ""

        if output_route == "multi":
            yield _make_status_event("synthesize", "正在汇总多个子任务结果...")
            messages = build_synthesize_messages(user_query, step_results)
            chunks: list[str] = []
            async for chunk in llm_client.stream_chat(
                messages=messages,
                temperature=0.3,
                max_tokens=4096,
            ):
                chunks.append(chunk)
                yield {"type": "token", "text": chunk}
            final_answer = "".join(chunks)
        else:
            final_answer = (await passthrough_node(state))["final_answer"]
            yield _make_status_event("output", "正在输出最终回答...")
            async for chunk in _stream_text_chunks(final_answer):
                yield {"type": "token", "text": chunk}

        await save_conversation(
            session=session,
            user_id=user_id,
            session_id=session_id,
            user_message=user_query,
            assistant_message=final_answer,
        )

        yield {
            "type": "final",
            "answer": final_answer,
            "step_results": step_results,
        }
        return
