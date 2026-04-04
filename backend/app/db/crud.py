"""CRUD operations for SQLite tables."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import KnowledgeConcept
from app.services.doc_processor import ConceptInfo


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
