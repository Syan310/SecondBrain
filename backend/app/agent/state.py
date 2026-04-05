"""LangGraph State definitions.

OverallState: top-level shared state for Orchestrator routing.
QAState / QuizState / GapState: independent SubAgent states.
"""

from __future__ import annotations

from typing import Optional, TypedDict


# ── Top-level Orchestrator State ─────────────────────────────


class OverallState(TypedDict, total=False):
    """Shared state for the Orchestrator graph.

    Plan & Execute with Replan loop:
    - plan_node: LLM generates execution plan (steps with depends_on)
    - execute_step: executes ONE step per iteration, calling QA/Gap/Quiz SubAgents
    - replan_node: reviews step result, decides continue/revise/finish
    - synthesize_node: merges multiple step results (skipped for single-step)
    """

    messages: list[dict]          # full conversation history
    user_id: str
    user_query: str               # original user input

    # Plan & Execute fields
    plan: list[dict]              # execution steps: [{step, intent, params, depends_on}]
    step_results: list[dict]      # [{step, intent, result}]

    # Replan loop fields
    current_step_index: int       # index into plan[] for next step to execute, starts at 0
    replan_action: str            # "continue" | "revise" | "finish"
    replan_count: int             # number of replan iterations (max 3)

    final_answer: str             # final output to user


# ── QA SubAgent State ────────────────────────────────────────


class QAState(TypedDict, total=False):
    """State for the QA SubAgent (ReAct + decompose + QC)."""

    messages: list[dict]      # agent message history (system + tool calls + responses)
    user_id: str
    original_query: str
    sub_queries: list[str]    # after decompose (or just [original_query])
    retrieved_chunks: list    # chunks from rag_search
    sub_answers: list[dict]   # [{query, answer, chunks_used}]
    final_answer: str
    confidence: str           # "high" | "low"
    retries: int              # QC retry count (max 2)


# ── Quiz SubAgent State ──────────────────────────────────────


class QuizState(TypedDict, total=False):
    """State for the Quiz SubAgent (2-question generation as orchestrator step).

    Quiz agent generates questions only; grading happens in API layer.
    """

    user_id: str
    topics: list[str]          # topics to quiz on (from prior step or mastery-based)
    prior_context: Optional[str]  # context from prior QA/Gap step
    questions: list[dict]      # generated questions: [{topic, question, choices, correct_answer, explanation}]


# ── Gap Analysis SubAgent State ──────────────────────────────


class GapState(TypedDict, total=False):
    """State for the Gap Analysis SubAgent."""

    messages: list[dict]      # agent message history
    user_id: str
    scope: Optional[str]      # optional source_doc filter
    mastery_records: list     # read_mastery results
    all_concepts: list        # search_concepts results
    gap_result: list[dict]    # [{topic, status, priority}]
    final_report: str
