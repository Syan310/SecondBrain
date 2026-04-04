from __future__ import annotations

"""CRUD operations for SQLite tables."""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import KnowledgeConcept, UserMastery
from app.services.doc_processor import ConceptInfo


# ── Knowledge Concepts ──────────────────────────────────────


async def upsert_concepts(
    session: AsyncSession,
    concepts: list[ConceptInfo],
) -> int:
    """Insert new concepts, skip existing ones (by topic). Returns count of newly inserted."""
    if not concepts:
        return 0

    # Fetch existing topics in one query
    topics = [c.topic for c in concepts]
    result = await session.execute(
        select(KnowledgeConcept.topic).where(KnowledgeConcept.topic.in_(topics))
    )
    existing = {row[0] for row in result.fetchall()}

    inserted = 0
    for c in concepts:
        if c.topic in existing:
            continue
        session.add(
            KnowledgeConcept(
                topic=c.topic,
                source_doc=c.source_doc,
                heading_path=c.heading_path,
            )
        )
        existing.add(c.topic)
        inserted += 1

    if inserted:
        await session.commit()
    return inserted


async def search_concepts(
    session: AsyncSession,
    scope: str | None = None,
    keyword: str | None = None,
) -> list[KnowledgeConcept]:
    """Query knowledge_concepts with optional filters.

    Args:
        scope: Filter by source_doc (exact match).
        keyword: Filter by topic (case-insensitive LIKE match).

    Returns:
        List of KnowledgeConcept rows.
    """
    stmt = select(KnowledgeConcept)
    if scope:
        stmt = stmt.where(KnowledgeConcept.source_doc == scope)
    if keyword:
        stmt = stmt.where(KnowledgeConcept.topic.ilike(f"%{keyword}%"))
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
    quality: str,
) -> UserMastery:
    """Update mastery after a quiz answer; create if not exists.

    Args:
        user_id: User identifier.
        topic: The topic being tested.
        new_mastery_score: Pre-calculated new mastery score (caller applies decay + delta).
        quality: Answer quality — "low", "medium", or "high".
                 medium/high count as correct for correct_count.

    Returns:
        The updated UserMastery record.
    """
    is_counted_correct = quality in ("medium", "high")

    stmt = select(UserMastery).where(
        UserMastery.user_id == user_id,
        UserMastery.topic == topic,
    )
    result = await session.execute(stmt)
    record = result.scalar_one_or_none()

    now = datetime.utcnow()

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
