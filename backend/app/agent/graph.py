"""LangGraph main graph: Orchestrator Plan & Execute with Replan loop.

Graph structure:
  START → plan_node → route_after_plan
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

import logging

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
)

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

    # Wrap execute_step to bind session
    async def _execute_step(state: OverallState) -> dict:
        return await execute_step(state, session)

    graph = StateGraph(OverallState)

    # Add nodes
    graph.add_node("plan", plan_node)
    graph.add_node("execute_step", _execute_step)
    graph.add_node("replan", replan_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("passthrough", passthrough_node)
    graph.add_node("empty_response", _empty_response)

    # Entry point
    graph.set_entry_point("plan")

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
) -> dict:
    """Convenience function to run the orchestrator graph.

    Args:
        user_query: The user's input text.
        user_id: User identifier.
        session: Async DB session.

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

    return {
        "final_answer": result.get("final_answer", ""),
        "step_results": result.get("step_results", []),
    }
