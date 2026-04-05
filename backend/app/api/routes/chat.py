"""Unified conversation API — routes all intents through Orchestrator."""

import json
import logging

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.api.schemas import (
    ChatRequest, ChatResponse, ChatHistoryResponse, HistoryMessage, StepResult,
)
from app.agent.graph import run_orchestrator, run_orchestrator_stream
from app.db.crud import get_recent_conversations

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def _format_sse(event_type: str, payload: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


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


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    session: AsyncSession = Depends(get_db),
):
    """流式对话入口 — 推送阶段事件与最终答案 token。"""

    async def event_generator():
        try:
            async for event in run_orchestrator_stream(
                user_query=req.user_query,
                user_id=req.user_id,
                session=session,
                session_id=req.session_id,
            ):
                event_type = event.get("type", "message")
                payload = {k: v for k, v in event.items() if k != "type"}
                yield _format_sse(event_type, payload)
        except Exception as exc:
            logger.exception("Streaming chat failed")
            yield _format_sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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
