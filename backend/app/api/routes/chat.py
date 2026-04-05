"""Unified conversation API — routes all intents through Orchestrator."""

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.api.schemas import (
    ChatRequest, ChatResponse, ChatHistoryResponse, HistoryMessage, StepResult,
)
from app.agent.graph import run_orchestrator
from app.db.crud import get_recent_conversations

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    session: AsyncSession = Depends(get_db),
):
    """统一对话入口 — QA / Gap Analysis / Quiz 全走 Orchestrator。"""
    result = await run_orchestrator(
        user_query=req.user_query,
        user_id=req.user_id,
        session=session,
        session_id=req.session_id,
    )

    step_results = [
        StepResult(step=sr.get("step", i), intent=sr.get("intent", ""), result=sr.get("result", ""))
        for i, sr in enumerate(result.get("step_results", []))
    ]

    return ChatResponse(
        answer=result.get("final_answer", ""),
        step_results=step_results,
    )


@router.get("/history", response_model=ChatHistoryResponse)
async def chat_history(
    user_id: str = Query(default="default"),
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
):
    """返回用户的历史对话记录。"""
    rows = await get_recent_conversations(session, user_id, limit_rounds=limit)
    messages = [
        HistoryMessage(role=row.role, content=row.content)
        for row in rows
    ]
    return ChatHistoryResponse(messages=messages)
