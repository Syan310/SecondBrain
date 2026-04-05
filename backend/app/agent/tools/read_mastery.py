from __future__ import annotations

"""Read Mastery tool: fetch user mastery with time-decay calculation.

Formula:
  effective_mastery = mastery_score * exp(-decay_rate * days_since_review)

Used by Quiz SubAgent and Gap Analysis SubAgent.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import crud

logger = logging.getLogger(__name__)


@dataclass
class MasteryRecord:
    """A mastery record with effective (time-decayed) mastery."""

    topic: str
    mastery_score: float       # raw score from DB
    effective_mastery: float   # after time decay
    decay_rate: float
    days_since_review: float
    total_attempts: int
    correct_count: int


def _compute_effective_mastery(
    mastery_score: float,
    decay_rate: float,
    last_reviewed: datetime | None,
) -> tuple[float, float]:
    """Compute effective mastery and days since review.

    Returns:
        (effective_mastery, days_since_review)
    """
    if last_reviewed is None:
        return mastery_score, 0.0

    now = datetime.now(timezone.utc)
    # Ensure last_reviewed is timezone-aware for safe subtraction
    if last_reviewed.tzinfo is None:
        last_reviewed = last_reviewed.replace(tzinfo=timezone.utc)
    delta_days = (now - last_reviewed).total_seconds() / 86400.0
    effective = mastery_score * math.exp(-decay_rate * delta_days)
    return effective, delta_days


async def read_mastery(
    session: AsyncSession,
    user_id: str,
    topics: list[str] | None = None,
) -> list[MasteryRecord]:
    """Read user mastery records with time-decay applied.

    Args:
        session: Async DB session.
        user_id: User identifier.
        topics: Optional list of topics to filter. If None, returns all.

    Returns:
        List of MasteryRecord with effective_mastery computed.
    """
    rows = await crud.get_mastery(session, user_id, topics)

    records = []
    for row in rows:
        effective, days = _compute_effective_mastery(
            row.mastery_score, row.decay_rate, row.last_reviewed
        )
        records.append(
            MasteryRecord(
                topic=row.topic,
                mastery_score=row.mastery_score,
                effective_mastery=round(effective, 4),
                decay_rate=row.decay_rate,
                days_since_review=round(days, 2),
                total_attempts=row.total_attempts,
                correct_count=row.correct_count,
            )
        )

    logger.info(
        "read_mastery: user=%s | topics_filter=%s | returned=%d records",
        user_id,
        topics,
        len(records),
    )
    return records
