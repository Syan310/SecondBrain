"""Update Mastery tool: update mastery_score after a quiz answer.

Binary scoring:
  correct:   new_score = min(1.0, effective_mastery + 0.1)
  incorrect: new_score = max(0.0, effective_mastery - 0.15)

last_reviewed is reset to now on every update.

Used by Quiz answer grading API.
"""

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import crud
from app.agent.tools.read_mastery import _compute_effective_mastery

logger = logging.getLogger(__name__)

# Mastery delta by correctness
CORRECT_DELTA = 0.1
INCORRECT_DELTA = -0.15


@dataclass
class MasteryUpdateResult:
    """Result of a mastery update."""

    topic: str
    old_effective_mastery: float
    new_mastery_score: float
    is_correct: bool
    total_attempts: int
    correct_count: int


async def update_mastery(
    session: AsyncSession,
    user_id: str,
    topic: str,
    is_correct: bool,
) -> MasteryUpdateResult:
    """Update mastery for a topic after answering a quiz question.

    Reads current effective mastery (with decay), applies binary delta,
    then persists the new score with last_reviewed = now.

    Args:
        session: Async DB session.
        user_id: User identifier.
        topic: The topic being tested.
        is_correct: Whether the user answered correctly.

    Returns:
        MasteryUpdateResult with old and new values.
    """
    # Read current record (if any)
    rows = await crud.get_mastery(session, user_id, [topic])

    if rows:
        row = rows[0]
        old_effective, _ = _compute_effective_mastery(
            row.mastery_score, row.decay_rate, row.last_reviewed
        )
    else:
        # First attempt on this topic
        old_effective = settings.default_mastery_score

    # Apply binary delta
    delta = CORRECT_DELTA if is_correct else INCORRECT_DELTA
    new_score = max(0.0, min(1.0, old_effective + delta))

    # Persist
    record = await crud.upsert_mastery(
        session, user_id, topic, round(new_score, 4), is_correct
    )

    logger.info(
        "update_mastery: user=%s topic=%s correct=%s | %.4f → %.4f (delta=%+.2f)",
        user_id,
        topic,
        is_correct,
        old_effective,
        new_score,
        delta,
    )

    return MasteryUpdateResult(
        topic=topic,
        old_effective_mastery=round(old_effective, 4),
        new_mastery_score=round(new_score, 4),
        is_correct=is_correct,
        total_attempts=record.total_attempts,
        correct_count=record.correct_count,
    )
