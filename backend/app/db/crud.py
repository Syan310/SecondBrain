from __future__ import annotations

"""CRUD operations for SQLite tables."""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ConceptSource, ConversationHistory, KnowledgeConcept, UserMastery


# ── Conversation History (Short-term Memory) ───────────────


async def add_conversation_round(
    session: AsyncSession,
    user_id: str,
    session_id: str,
    round_number: int,
    user_message: str,
    assistant_message: str,
) -> None:
    """Append one conversation round (user + assistant) to short-term memory."""
    session.add(ConversationHistory(
        user_id=user_id,
        session_id=session_id,
        role="user",
        content=user_message,
        round_number=round_number,
    ))
    session.add(ConversationHistory(
        user_id=user_id,
        session_id=session_id,
        role="assistant",
        content=assistant_message,
        round_number=round_number,
    ))
    await session.commit()


async def get_recent_conversations(
    session: AsyncSession,
    user_id: str,
    limit_rounds: int = 20,
) -> list[ConversationHistory]:
    """Get the most recent N rounds of conversation, ordered by round_number ASC.

    Returns all rows (user + assistant) for the latest `limit_rounds` rounds.
    """
    # Subquery: get the latest N distinct round_numbers
    from sqlalchemy import func, desc
    sub = (
        select(ConversationHistory.round_number)
        .where(ConversationHistory.user_id == user_id)
        .group_by(ConversationHistory.round_number)
        .order_by(desc(ConversationHistory.round_number))
        .limit(limit_rounds)
    ).subquery()

    stmt = (
        select(ConversationHistory)
        .where(
            ConversationHistory.user_id == user_id,
            ConversationHistory.round_number.in_(select(sub)),
        )
        .order_by(ConversationHistory.round_number, ConversationHistory.id)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_oldest_rounds(
    session: AsyncSession,
    user_id: str,
    count: int,
) -> list[ConversationHistory]:
    """Get the oldest N rounds (for consolidation into long-term memory)."""
    from sqlalchemy import desc
    sub = (
        select(ConversationHistory.round_number)
        .where(ConversationHistory.user_id == user_id)
        .group_by(ConversationHistory.round_number)
        .order_by(ConversationHistory.round_number)
        .limit(count)
    ).subquery()

    stmt = (
        select(ConversationHistory)
        .where(
            ConversationHistory.user_id == user_id,
            ConversationHistory.round_number.in_(select(sub)),
        )
        .order_by(ConversationHistory.round_number, ConversationHistory.id)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def delete_rounds(
    session: AsyncSession,
    user_id: str,
    round_numbers: list[int],
) -> int:
    """Delete conversation rows by round_numbers. Returns deleted count."""
    if not round_numbers:
        return 0
    from sqlalchemy import delete
    stmt = delete(ConversationHistory).where(
        ConversationHistory.user_id == user_id,
        ConversationHistory.round_number.in_(round_numbers),
    )
    result = await session.execute(stmt)
    await session.commit()
    return result.rowcount


async def count_rounds(
    session: AsyncSession,
    user_id: str,
) -> int:
    """Count distinct conversation rounds for a user."""
    from sqlalchemy import func
    stmt = (
        select(func.count(func.distinct(ConversationHistory.round_number)))
        .where(ConversationHistory.user_id == user_id)
    )
    result = await session.execute(stmt)
    return result.scalar() or 0


async def get_next_round_number(
    session: AsyncSession,
    user_id: str,
) -> int:
    """Get the next round_number for a user (max + 1, or 1 if none)."""
    from sqlalchemy import func
    stmt = (
        select(func.max(ConversationHistory.round_number))
        .where(ConversationHistory.user_id == user_id)
    )
    result = await session.execute(stmt)
    max_round = result.scalar()
    return (max_round or 0) + 1


# ── Knowledge Concepts ──────────────────────────────────────


async def delete_concepts_by_source(
    session: AsyncSession,
    source_doc: str,
) -> int:
    """Delete all concept sources for a document, and orphaned concepts.

    Used during re-ingestion to clear old data before inserting new ones.

    Returns:
        Number of deleted source rows.
    """
    from sqlalchemy import delete

    # 1. Delete source mappings for this document
    stmt = delete(ConceptSource).where(ConceptSource.source_doc == source_doc)
    result = await session.execute(stmt)
    deleted_sources = result.rowcount

    # 2. Delete orphaned concepts (no remaining source mappings)
    orphan_stmt = (
        delete(KnowledgeConcept)
        .where(
            ~KnowledgeConcept.id.in_(
                select(ConceptSource.concept_id).distinct()
            )
        )
    )
    await session.execute(orphan_stmt)
    await session.commit()
    return deleted_sources


async def upsert_concepts(
    session: AsyncSession,
    topics: list[str],
    source_doc: str,
    parent_chunk_id: str,
) -> int:
    """Insert new concepts and create source mappings.

    For each topic: get-or-create KnowledgeConcept, then create ConceptSource link.

    Returns:
        Number of newly created concepts (not source mappings).
    """
    if not topics:
        return 0

    # Fetch existing concepts by topic in one query
    result = await session.execute(
        select(KnowledgeConcept).where(KnowledgeConcept.topic.in_(topics))
    )
    existing_map: dict[str, KnowledgeConcept] = {
        row.topic: row for row in result.scalars().all()
    }

    new_count = 0
    for topic in topics:
        concept = existing_map.get(topic)
        if concept is None:
            concept = KnowledgeConcept(topic=topic)
            session.add(concept)
            await session.flush()  # get id
            existing_map[topic] = concept
            new_count += 1

        # Create source mapping (skip if already exists)
        existing_link = await session.execute(
            select(ConceptSource).where(
                ConceptSource.concept_id == concept.id,
                ConceptSource.parent_chunk_id == parent_chunk_id,
            )
        )
        if existing_link.scalar_one_or_none() is None:
            session.add(ConceptSource(
                concept_id=concept.id,
                source_doc=source_doc,
                parent_chunk_id=parent_chunk_id,
            ))

    await session.commit()
    return new_count


async def get_all_concept_topics(
    session: AsyncSession,
) -> list[str]:
    """Get all unique concept topics (for LLM deduplication during extraction)."""
    result = await session.execute(select(KnowledgeConcept.topic))
    return [row[0] for row in result.fetchall()]


async def list_documents(
    session: AsyncSession,
) -> list[dict]:
    """List all uploaded documents with concept counts.

    Returns:
        List of dicts with keys: source_doc, concept_count.
    """
    from sqlalchemy import func
    stmt = (
        select(
            ConceptSource.source_doc,
            func.count(func.distinct(ConceptSource.concept_id)).label("concept_count"),
        )
        .group_by(ConceptSource.source_doc)
        .order_by(ConceptSource.source_doc)
    )
    result = await session.execute(stmt)
    return [
        {"source_doc": row.source_doc, "concept_count": row.concept_count}
        for row in result.fetchall()
    ]


async def search_concepts(
    session: AsyncSession,
    scope: str | None = None,
    keyword: str | None = None,
) -> list[KnowledgeConcept]:
    """Query knowledge_concepts with optional filters.

    Args:
        scope: Filter by source_doc (via concept_sources join).
        keyword: Filter by topic (case-insensitive LIKE match).

    Returns:
        List of KnowledgeConcept rows (deduplicated).
    """
    stmt = select(KnowledgeConcept)
    if scope:
        stmt = stmt.join(
            ConceptSource, ConceptSource.concept_id == KnowledgeConcept.id
        ).where(
            ConceptSource.source_doc == scope
        ).distinct()
    if keyword:
        escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        stmt = stmt.where(KnowledgeConcept.topic.ilike(f"%{escaped}%", escape="\\"))
    result = await session.execute(stmt)
    return list(result.scalars().all())


# ── User Mastery ────────────────────────────────────────────


async def get_mastery(
    session: AsyncSession,
    user_id: str,
    topics: list[str] | None = None,
) -> list[UserMastery]:
    """Read mastery records for a user, optionally filtered by topics."""
    stmt = select(UserMastery).where(UserMastery.user_id == user_id)
    if topics:
        stmt = stmt.where(UserMastery.topic.in_(topics))
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def upsert_mastery(
    session: AsyncSession,
    user_id: str,
    topic: str,
    new_mastery_score: float,
    is_correct: bool,
) -> UserMastery:
    """Update mastery after a quiz answer; create if not exists.

    Args:
        user_id: User identifier.
        topic: The topic being tested.
        new_mastery_score: Pre-calculated new mastery score (caller applies decay + delta).
        is_correct: Whether the user answered correctly.

    Returns:
        The updated UserMastery record.
    """
    is_counted_correct = is_correct

    stmt = select(UserMastery).where(
        UserMastery.user_id == user_id,
        UserMastery.topic == topic,
    )
    result = await session.execute(stmt)
    record = result.scalar_one_or_none()

    now = datetime.now(timezone.utc)

    if record is None:
        record = UserMastery(
            user_id=user_id,
            topic=topic,
            mastery_score=new_mastery_score,
            total_attempts=1,
            correct_count=1 if is_counted_correct else 0,
            last_reviewed=now,
        )
        session.add(record)
    else:
        record.mastery_score = new_mastery_score
        record.total_attempts += 1
        if is_counted_correct:
            record.correct_count += 1
        record.last_reviewed = now

    await session.commit()
    await session.refresh(record)
    return record
