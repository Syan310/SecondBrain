from __future__ import annotations

"""Search Concepts tool: query knowledge_concepts table.

Supports filtering by:
  - scope (source_doc exact match)
  - keyword (topic case-insensitive LIKE match)

Used by Gap Analysis SubAgent.
"""

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import crud

logger = logging.getLogger(__name__)


@dataclass
class ConceptRecord:
    """A knowledge concept from the index."""

    topic: str


async def search_concepts(
    session: AsyncSession,
    scope: str | None = None,
    keyword: str | None = None,
) -> list[ConceptRecord]:
    """Search knowledge concepts with optional filters.

    Args:
        session: Async DB session.
        scope: Filter by source document name (exact match, via concept_sources join).
        keyword: Filter by topic keyword (case-insensitive substring match).

    Returns:
        List of ConceptRecord.
    """
    rows = await crud.search_concepts(session, scope=scope, keyword=keyword)

    records = [ConceptRecord(topic=row.topic) for row in rows]

    logger.info(
        "search_concepts: scope=%s keyword=%s | returned=%d concepts",
        scope,
        keyword,
        len(records),
    )
    return records
