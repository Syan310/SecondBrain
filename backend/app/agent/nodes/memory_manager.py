"""Memory Manager: short-term (SQLite) + long-term (Milvus) memory for Agent.

Flow per request:
  1. load_memory_context(): read short-term + retrieve long-term → formatted context string
  2. save_conversation():   after agent responds, save the round to short-term
  3. (auto) consolidate:    if short-term exceeds limit, compress oldest half → Milvus

Short-term: SQLite conversation_history (last 20 rounds, full text)
Long-term:  Milvus conversation_memory  (LLM-summarized, vector-searchable)
"""

from __future__ import annotations

import json
import logging
import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import crud
from app.services.embedding import embedding_service
from app.services.llm_client import llm_client
from app.services.milvus_client import memory_milvus_service

logger = logging.getLogger(__name__)

# ── Consolidation Prompt ────────────────────────────────────

CONSOLIDATE_SYSTEM_PROMPT = """\
你是一个对话记忆压缩器。你需要将一段学习对话压缩成一条简洁的记忆摘要。

## 提取要求

从对话中提取以下信息，整合为一段 100-200 字的自然语言摘要：

1. **讨论主题** — 涉及了哪些知识概念
2. **关键问答** — 用户问了什么，核心结论是什么
3. **理解难点** — 用户在哪些地方表现出困惑或误解（如果有）
4. **关键类比/例子** — 给出过的帮助理解的类比或示例（如果有）

## 输出格式

```json
{
  "summary": "自然语言摘要（100-200字）",
  "topics": ["主题1", "主题2"]
}
```

## 注意

- 摘要用自然语言，不要用列表，要像一段描述性文字
- topics 提取对话涉及的核心知识概念关键词（2-5个）
- 不要包含对话的形式细节（如"用户说"、"助手回答"），只保留内容
"""


# ── Load Memory Context ─────────────────────────────────────


async def load_memory_context(
    session: AsyncSession,
    user_id: str,
    user_query: str,
) -> str:
    """Build memory context string from short-term + long-term memory.

    Args:
        session: Async DB session.
        user_id: User identifier.
        user_query: Current user query (for long-term retrieval).

    Returns:
        Formatted context string to inject into Orchestrator prompt.
        Empty string if no memories found.
    """
    parts: list[str] = []

    # 1. Short-term: recent conversation rounds from SQLite
    recent = await crud.get_recent_conversations(
        session, user_id, limit_rounds=settings.memory_short_term_limit
    )
    if recent:
        lines = []
        for row in recent:
            prefix = "用户" if row.role == "user" else "助手"
            lines.append(f"{prefix}: {row.content}")
        parts.append("## 近期对话\n" + "\n".join(lines))

    # 2. Long-term: relevant memories from Milvus
    try:
        encoded = embedding_service.encode([user_query])
        dense = encoded.dense[0]
        sparse = encoded.sparse[0]

        hits = memory_milvus_service.hybrid_search(
            query_vector=dense,
            sparse_vector=sparse,
            user_id=user_id,
            top_k=settings.memory_long_term_top_k,
        )

        # Filter by similarity threshold
        relevant = [h for h in hits if h.score >= settings.memory_similarity_threshold]

        if relevant:
            lines = [h.summary for h in relevant]
            parts.append("## 相关历史记忆\n" + "\n---\n".join(lines))

    except Exception as e:
        logger.warning("Long-term memory retrieval failed: %s", e)

    return "\n\n".join(parts)


# ── Save Conversation Round ─────────────────────────────────


async def save_conversation(
    session: AsyncSession,
    user_id: str,
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """Save one conversation round and trigger consolidation if needed.

    Args:
        session: Async DB session.
        user_id: User identifier.
        session_id: Current chat session ID.
        user_message: The user's input text.
        assistant_message: The agent's final answer (not intermediate steps).
    """
    # Get next round number
    round_number = await crud.get_next_round_number(session, user_id)

    # Save to short-term
    await crud.add_conversation_round(
        session=session,
        user_id=user_id,
        session_id=session_id,
        round_number=round_number,
        user_message=user_message,
        assistant_message=assistant_message,
    )

    logger.info(
        "Saved conversation round %d for user '%s'.", round_number, user_id
    )

    # Check if consolidation is needed
    total_rounds = await crud.count_rounds(session, user_id)
    if total_rounds > settings.memory_short_term_limit:
        await _consolidate(session, user_id, session_id)


# ── Consolidation: Short-term → Long-term ───────────────────


async def _consolidate(
    session: AsyncSession,
    user_id: str,
    session_id: str,
) -> None:
    """Compress oldest N rounds into a long-term memory entry in Milvus.

    Steps:
      1. Read oldest N rounds from SQLite
      2. LLM summarize into structured text
      3. Embed and write to Milvus
      4. Delete the consolidated rounds from SQLite
    """
    count = settings.memory_consolidate_count

    # 1. Read oldest rounds
    oldest = await crud.get_oldest_rounds(session, user_id, count)
    if not oldest:
        return

    # Format conversation for LLM
    lines = []
    for row in oldest:
        prefix = "用户" if row.role == "user" else "助手"
        lines.append(f"{prefix}: {row.content}")
    conversation_text = "\n".join(lines)

    # 2. LLM summarize
    try:
        messages = [
            {"role": "system", "content": CONSOLIDATE_SYSTEM_PROMPT},
            {"role": "user", "content": f"请压缩以下对话：\n\n{conversation_text}"},
        ]
        response = await llm_client.chat(
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )

        content = response["content"].strip()

        # Parse JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(content)
        summary = parsed.get("summary", "")
        topics = parsed.get("topics", [])

        if not summary:
            logger.warning("Consolidation produced empty summary, skipping.")
            return

    except Exception as e:
        logger.error("Consolidation LLM call failed: %s", e)
        return

    # 3. Embed and write to Milvus
    try:
        encoded = embedding_service.encode([summary])
        dense = encoded.dense[0]
        sparse = encoded.sparse[0]

        memory_milvus_service.insert(
            user_id=user_id,
            summary=summary,
            topics=json.dumps(topics, ensure_ascii=False),
            session_id=session_id,
            created_at=time.time(),
            embedding=dense,
            sparse_embedding=sparse,
        )

        logger.info(
            "Consolidated %d rounds into long-term memory for user '%s'. Topics: %s",
            count, user_id, topics,
        )

    except Exception as e:
        logger.error("Consolidation Milvus write failed: %s", e)
        return

    # 4. Delete consolidated rounds from SQLite
    round_numbers = sorted({row.round_number for row in oldest})
    deleted = await crud.delete_rounds(session, user_id, round_numbers)
    logger.info("Deleted %d short-term rows (rounds %s).", deleted, round_numbers)
