"""Quiz grading API — rule-based MCQ grading + mastery update.

Does NOT go through the Orchestrator. The Orchestrator generates quiz
questions via chat.py; this endpoint handles answer submission separately.
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.api.schemas import QuizAnswerRequest, QuizAnswerResponse, MasteryUpdate
from app.agent.tools.update_mastery import update_mastery

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quiz", tags=["quiz"])


@router.post("/answer", response_model=QuizAnswerResponse)
async def submit_answer(
    req: QuizAnswerRequest,
    session: AsyncSession = Depends(get_db),
):
    """提交 Quiz 答案 — 规则比对字母判分 → 更新 mastery → 返回结果。"""
    is_correct = req.user_answer.upper() == req.correct_answer.upper()

    result = await update_mastery(
        session=session,
        user_id=req.user_id,
        topic=req.topic,
        is_correct=is_correct,
    )

    return QuizAnswerResponse(
        is_correct=is_correct,
        correct_answer=req.correct_answer.upper(),
        mastery_update=MasteryUpdate(
            topic=result.topic,
            old_effective_mastery=result.old_effective_mastery,
            new_mastery_score=result.new_mastery_score,
            total_attempts=result.total_attempts,
            correct_count=result.correct_count,
        ),
    )
