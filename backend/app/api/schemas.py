"""Pydantic request/response models for API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Chat (unified conversation endpoint) ─────────────────────


class ChatRequest(BaseModel):
    user_query: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(default="default")
    session_id: str = Field(default="default")


class StepResult(BaseModel):
    step: int
    intent: str
    result: str


class ChatResponse(BaseModel):
    answer: str
    step_results: list[StepResult] = []


# ── Quiz Answer (grading endpoint) ───────────────────────────


class QuizAnswerRequest(BaseModel):
    user_id: str = Field(default="default")
    topic: str = Field(..., min_length=1)
    user_answer: str = Field(..., min_length=1, max_length=1,
                             description="User's choice letter, e.g. A/B/C/D")
    correct_answer: str = Field(..., min_length=1, max_length=1,
                                description="Correct choice letter from question")


class MasteryUpdate(BaseModel):
    topic: str
    old_effective_mastery: float
    new_mastery_score: float
    total_attempts: int
    correct_count: int


class QuizAnswerResponse(BaseModel):
    is_correct: bool
    correct_answer: str
    mastery_update: MasteryUpdate
