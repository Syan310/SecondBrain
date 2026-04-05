"""Tests for conversation history CRUD operations.

Uses a real async SQLite in-memory database to verify:
- add_conversation_round
- get_recent_conversations
- get_oldest_rounds
- delete_rounds
- count_rounds
- get_next_round_number
"""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base
from app.db.crud import (
    add_conversation_round,
    count_rounds,
    delete_rounds,
    get_next_round_number,
    get_oldest_rounds,
    get_recent_conversations,
)


# ── Fixtures ────────────────────────────────────────────────


@pytest_asyncio.fixture
async def session():
    """Create an in-memory SQLite database and yield a session."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as sess:
        yield sess

    await engine.dispose()


USER_ID = "test_user"
SESSION_ID = "test_session"


async def _add_rounds(session: AsyncSession, count: int, start_round: int = 1) -> None:
    """Helper: add N conversation rounds."""
    for i in range(count):
        r = start_round + i
        await add_conversation_round(
            session=session,
            user_id=USER_ID,
            session_id=SESSION_ID,
            round_number=r,
            user_message=f"用户消息 {r}",
            assistant_message=f"助手回复 {r}",
        )


# ── Tests ───────────────────────────────────────────────────


class TestAddAndCount:
    @pytest.mark.asyncio
    async def test_add_single_round(self, session):
        await _add_rounds(session, 1)
        total = await count_rounds(session, USER_ID)
        assert total == 1

    @pytest.mark.asyncio
    async def test_add_multiple_rounds(self, session):
        await _add_rounds(session, 5)
        total = await count_rounds(session, USER_ID)
        assert total == 5

    @pytest.mark.asyncio
    async def test_count_zero_for_unknown_user(self, session):
        total = await count_rounds(session, "unknown_user")
        assert total == 0


class TestGetNextRoundNumber:
    @pytest.mark.asyncio
    async def test_first_round(self, session):
        r = await get_next_round_number(session, USER_ID)
        assert r == 1

    @pytest.mark.asyncio
    async def test_after_three_rounds(self, session):
        await _add_rounds(session, 3)
        r = await get_next_round_number(session, USER_ID)
        assert r == 4


class TestGetRecentConversations:
    @pytest.mark.asyncio
    async def test_get_all_when_under_limit(self, session):
        await _add_rounds(session, 3)
        rows = await get_recent_conversations(session, USER_ID, limit_rounds=10)
        # 3 rounds × 2 rows (user + assistant) = 6 rows
        assert len(rows) == 6

    @pytest.mark.asyncio
    async def test_limit_to_recent_n(self, session):
        await _add_rounds(session, 10)
        rows = await get_recent_conversations(session, USER_ID, limit_rounds=3)
        # Should only get the last 3 rounds (rounds 8, 9, 10) × 2 = 6 rows
        assert len(rows) == 6
        round_numbers = sorted({r.round_number for r in rows})
        assert round_numbers == [8, 9, 10]

    @pytest.mark.asyncio
    async def test_order_ascending(self, session):
        await _add_rounds(session, 5)
        rows = await get_recent_conversations(session, USER_ID, limit_rounds=5)
        round_nums = [r.round_number for r in rows]
        # Should be in ascending order
        assert round_nums == sorted(round_nums)

    @pytest.mark.asyncio
    async def test_user_isolation(self, session):
        """Conversations from different users should not mix."""
        await _add_rounds(session, 3)
        # Add rounds for another user
        await add_conversation_round(
            session=session,
            user_id="other_user",
            session_id="other_session",
            round_number=1,
            user_message="other msg",
            assistant_message="other reply",
        )
        rows = await get_recent_conversations(session, USER_ID, limit_rounds=20)
        assert all(r.user_id == USER_ID for r in rows)
        assert len(rows) == 6  # only test_user's 3 rounds

    @pytest.mark.asyncio
    async def test_content_preserved(self, session):
        await add_conversation_round(
            session=session,
            user_id=USER_ID,
            session_id=SESSION_ID,
            round_number=1,
            user_message="什么是Transformer?",
            assistant_message="Transformer是一种模型架构。",
        )
        rows = await get_recent_conversations(session, USER_ID, limit_rounds=1)
        user_row = [r for r in rows if r.role == "user"][0]
        assistant_row = [r for r in rows if r.role == "assistant"][0]
        assert user_row.content == "什么是Transformer?"
        assert assistant_row.content == "Transformer是一种模型架构。"


class TestGetOldestRounds:
    @pytest.mark.asyncio
    async def test_get_oldest_n(self, session):
        await _add_rounds(session, 10)
        rows = await get_oldest_rounds(session, USER_ID, count=3)
        round_numbers = sorted({r.round_number for r in rows})
        assert round_numbers == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_get_oldest_when_fewer_exist(self, session):
        await _add_rounds(session, 2)
        rows = await get_oldest_rounds(session, USER_ID, count=5)
        # Should return all 2 rounds (4 rows)
        assert len(rows) == 4


class TestDeleteRounds:
    @pytest.mark.asyncio
    async def test_delete_specific_rounds(self, session):
        await _add_rounds(session, 5)
        deleted = await delete_rounds(session, USER_ID, [1, 2, 3])
        # 3 rounds × 2 rows = 6 deleted
        assert deleted == 6
        total = await count_rounds(session, USER_ID)
        assert total == 2

    @pytest.mark.asyncio
    async def test_delete_empty_list(self, session):
        await _add_rounds(session, 3)
        deleted = await delete_rounds(session, USER_ID, [])
        assert deleted == 0
        total = await count_rounds(session, USER_ID)
        assert total == 3

    @pytest.mark.asyncio
    async def test_remaining_after_delete(self, session):
        """After deleting oldest, remaining should be the newer rounds."""
        await _add_rounds(session, 5)
        await delete_rounds(session, USER_ID, [1, 2])
        rows = await get_recent_conversations(session, USER_ID, limit_rounds=20)
        remaining_rounds = sorted({r.round_number for r in rows})
        assert remaining_rounds == [3, 4, 5]
