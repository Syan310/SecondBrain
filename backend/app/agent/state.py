"""LangGraph State definitions.

OverallState: top-level shared state for Orchestrator routing.
QAState / QuizState / GapState: independent SubAgent states.
"""

from typing import TypedDict


# ── Top-level Orchestrator State ─────────────────────────────


class OverallState(TypedDict, total=False):
    """Shared state for the Orchestrator graph."""

    messages: list[dict]      # full conversation history
    user_id: str
    intent: str               # "qa" | "quiz" | "gap_analysis"
    final_answer: str         # result returned by the SubAgent


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
    """State for the Quiz SubAgent (10-question agent-driven quiz)."""

    messages: list[dict]      # agent message history
    user_id: str
    session_id: str
    target_topics: list[str]  # user-specified or weighted-random topics
    current_round: int        # current question number (1-10)
    quiz_history: list[dict]  # [{topic, question, user_answer, is_correct, score_change}]
    final_summary: str


# ── Gap Analysis SubAgent State ──────────────────────────────


class GapState(TypedDict, total=False):
    """State for the Gap Analysis SubAgent."""

    messages: list[dict]      # agent message history
    user_id: str
    scope: str | None         # optional source_doc filter
    mastery_records: list     # read_mastery results
    all_concepts: list        # search_concepts results
    gap_result: list[dict]    # [{topic, status, priority}]
    final_report: str
