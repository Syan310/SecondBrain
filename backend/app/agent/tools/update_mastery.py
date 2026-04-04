"""Update Mastery tool: update mastery_score after a quiz answer.

Three-tier quality scoring:
  high:    new_score = min(1.0, effective_mastery + 0.1)
  medium:  new_score = min(1.0, effective_mastery + 0.05)
  low:     new_score = max(0.0, effective_mastery - 0.15)

medium/high count as correct for correct_count.
last_reviewed is reset to now on every update.

Used by Quiz SubAgent.
"""

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import crud
from app.agent.tools.read_mastery import _compute_effective_mastery

logger = logging.getLogger(__name__)

# Mastery delta by quality tier
QUALITY_DELTA = {
    "high": 0.1,
    "medium": 0.05,
    "low": -0.15,
}

VALID_QUALITIES = frozenset(QUALITY_DELTA.keys())


@dataclass
class MasteryUpdateResult:
    """Result of a mastery update."""

    topic: str
    old_effective_mastery: float
    new_mastery_score: float
    quality: str              # "low" | "medium" | "high"
    total_attempts: int
    correct_count: int


async def update_mastery(
    session: AsyncSession,
    user_id: str,
    topic: str,
    quality: str,
) -> MasteryUpdateResult:
    """Update mastery for a topic after answering a quiz question.

    Reads current effective mastery (with decay), applies quality-based delta,
    then persists the new score with last_reviewed = now.

    Args:
        session: Async DB session.
        user_id: User identifier.
        topic: The topic being tested.
        quality: Answer quality — "low", "medium", or "high".

    Returns:
        MasteryUpdateResult with old and new values.
    """
    if quality not in VALID_QUALITIES:
        raise ValueError(f"Invalid quality '{quality}', must be one of {VALID_QUALITIES}")

    # Read current record (if any)
    rows = await crud.get_mastery(session, user_id, [topic])

    if rows:
        row = rows[0]
        old_effective, _ = _compute_effective_mastery(
            row.mastery_score, row.decay_rate, row.last_reviewed
        )
    else:
        # First attempt on this topic — default mastery 0.3
        old_effective = 0.3

    # Apply quality-based delta
    delta = QUALITY_DELTA[quality]
    new_score = max(0.0, min(1.0, old_effective + delta))

    # Persist
    record = await crud.upsert_mastery(
        session, user_id, topic, round(new_score, 4), quality
    )

    logger.info(
        "update_mastery: user=%s topic=%s quality=%s | %.4f → %.4f (delta=%+.2f)",
        user_id,
        topic,
        quality,
        old_effective,
        new_score,
        delta,
    )

    return MasteryUpdateResult(
        topic=topic,
        old_effective_mastery=round(old_effective, 4),
        new_mastery_score=round(new_score, 4),
        quality=quality,
        total_attempts=record.total_attempts,
        correct_count=record.correct_count,
    )
