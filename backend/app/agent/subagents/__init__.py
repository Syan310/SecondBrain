"""SubAgents — autonomous agents for QA, Quiz, and Gap Analysis."""

from app.agent.subagents.qa_agent import run_qa_agent
from app.agent.subagents.quiz_agent import run_quiz_agent
from app.agent.subagents.gap_agent import run_gap_agent

__all__ = [
    "run_qa_agent",
    "run_quiz_agent",
    "run_gap_agent",
]
