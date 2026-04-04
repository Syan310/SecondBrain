"""LangGraph main graph: Orchestrator Plan & Execute.

Graph structure:
  START → plan_node → route_after_plan
    ├── no_steps (quiz-only) → quiz_passthrough → END
    └── has_steps → execute_node → route_after_execute
                      ├── single → passthrough_node → END
                      └── multi  → synthesize_node → END

Usage:
    from app.agent.graph import build_graph

    graph = build_graph(session)
    result = await graph.ainvoke({
        "user_query": "...",
        "user_id": "default",
    })
"""

import logging

from langgraph.graph import StateGraph, END
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.state import OverallState
from app.agent.nodes.orchestrator import (
    plan_node,
    execute_node,
    synthesize_node,
    passthrough_node,
    route_after_plan,
    route_after_execute,
)

logger = logging.getLogger(__name__)


def _quiz_passthrough(state: OverallState) -> dict:
    """Handle quiz-only requests: no steps to execute."""
    followup = state.get("suggested_followup")
    if followup:
        return {
            "final_answer": "好的，准备开始测验！请确认后我将为你出题。",
        }
    return {"final_answer": "抱歉，我没有理解你的请求。"}


def build_graph(session: AsyncSession) -> StateGraph:
    """Build the Orchestrator LangGraph, binding the DB session.

    The session is needed by execute_node to pass to Gap SubAgent.

    Args:
        session: Async SQLAlchemy session for DB-dependent SubAgents.

    Returns:
        Compiled LangGraph ready for ainvoke().
    """

    # Wrap execute_node to bind session
    async def _execute_node(state: OverallState) -> dict:
        return await execute_node(state, session)

    # Build graph
    graph = StateGraph(OverallState)

    # Add nodes
    graph.add_node("plan", plan_node)
    graph.add_node("execute", _execute_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("passthrough", passthrough_node)
    graph.add_node("quiz_passthrough", _quiz_passthrough)

    # Set entry point
    graph.set_entry_point("plan")

    # Conditional edge after plan
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "no_steps": "quiz_passthrough",
            "has_steps": "execute",
        },
    )

    # Conditional edge after execute
    graph.add_conditional_edges(
        "execute",
        route_after_execute,
        {
            "single": "passthrough",
            "multi": "synthesize",
        },
    )

    # Terminal edges
    graph.add_edge("quiz_passthrough", END)
    graph.add_edge("passthrough", END)
    graph.add_edge("synthesize", END)

    return graph.compile()


async def run_orchestrator(
    user_query: str,
    user_id: str,
    session: AsyncSession,
) -> dict:
    """Convenience function to run the orchestrator graph.

    Args:
        user_query: The user's input text.
        user_id: User identifier.
        session: Async DB session.

    Returns:
        dict with keys:
          - final_answer (str): The response to show the user.
          - suggested_followup (dict|None): E.g. {"intent": "quiz", "params": {...}}
          - step_results (list[dict]): Individual step results for debugging.
    """
    graph = build_graph(session)

    result = await graph.ainvoke({
        "user_query": user_query,
        "user_id": user_id,
        "messages": [],
    })

    return {
        "final_answer": result.get("final_answer", ""),
        "suggested_followup": result.get("suggested_followup"),
        "step_results": result.get("step_results", []),
    }
