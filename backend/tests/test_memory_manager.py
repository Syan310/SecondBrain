"""Tests for Memory Manager: load_memory_context, save_conversation, consolidation.

Mocks: LLM, embedding_service, memory_milvus_service (Milvus).
Real:  SQLite in-memory database for short-term memory.

Strategy: test_orchestrator.py pollutes sys.modules at collection time.
We use a session-scoped fixture to restore real modules at TEST EXECUTION
time, then use monkeypatch to swap dependencies on the real module object.
"""

import json
import importlib
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Pre-mock heavy third-party deps at collection time (safe, these are never real).
for mod_name in [
    "FlagEmbedding", "pymilvus", "transformers", "docx", "torch",
    "magic_pdf", "magic_pdf.data", "magic_pdf.data.data_reader_writer",
    "magic_pdf.pipe", "magic_pdf.pipe.UNIPipe",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def _restore_real_modules():
    """Restore real app modules that test_orchestrator may have mocked.

    Runs once at the start of this test file's session (after collection).
    """
    modules_to_restore = [
        "aiosqlite",
        "app.db", "app.db.models", "app.db.crud", "app.db.database",
        "app.services.embedding", "app.services.milvus_client",
        "app.services.llm_client",
        "app.agent.nodes.memory_manager",
    ]
    for mod_name in modules_to_restore:
        if mod_name in sys.modules and isinstance(sys.modules[mod_name], MagicMock):
            del sys.modules[mod_name]

    # Re-import in dependency order: db layer → services → memory_manager
    for mod_name in modules_to_restore:
        try:
            mod = importlib.import_module(mod_name)
            importlib.reload(mod)
        except Exception:
            pass


@pytest_asyncio.fixture
async def session():
    """Create an in-memory SQLite database and yield a session."""
    from app.db.models import Base

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as sess:
        yield sess
    await engine.dispose()


@pytest.fixture
def mm():
    """Return the real memory_manager module (freshly imported)."""
    import app.agent.nodes.memory_manager as _mm
    return _mm


USER_ID = "test_user"
SESSION_ID = "test_session"


def _make_llm_response(content: str) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": None}


def _fake_encode_result():
    """Return a mock EmbeddingOutput-like object."""
    mock_output = MagicMock()
    mock_output.__getitem__ = lambda self, key: {
        "dense": [[0.1] * 1024],
        "sparse": [{1: 0.5, 2: 0.3}],
    }[key]
    return mock_output


async def _populate_rounds(session: AsyncSession, count: int) -> None:
    """Add N conversation rounds to short-term memory."""
    from app.db.crud import add_conversation_round

    for i in range(1, count + 1):
        await add_conversation_round(
            session=session,
            user_id=USER_ID,
            session_id=SESSION_ID,
            round_number=i,
            user_message=f"用户问题 {i}",
            assistant_message=f"助手回答 {i}",
        )


# ── load_memory_context tests ───────────────────────────────


class TestLoadMemoryContext:
    @pytest.mark.asyncio
    async def test_empty_when_no_history(self, session, mm, monkeypatch):
        mock_emb = MagicMock()
        mock_emb.encode.return_value = _fake_encode_result()
        mock_milvus = MagicMock()
        mock_milvus.hybrid_search.return_value = []

        monkeypatch.setattr(mm, "embedding_service", mock_emb)
        monkeypatch.setattr(mm, "memory_milvus_service", mock_milvus)

        context = await mm.load_memory_context(session, USER_ID, "test query")
        assert context == ""

    @pytest.mark.asyncio
    async def test_includes_short_term(self, session, mm, monkeypatch):
        await _populate_rounds(session, 3)

        mock_emb = MagicMock()
        mock_emb.encode.return_value = _fake_encode_result()
        mock_milvus = MagicMock()
        mock_milvus.hybrid_search.return_value = []

        monkeypatch.setattr(mm, "embedding_service", mock_emb)
        monkeypatch.setattr(mm, "memory_milvus_service", mock_milvus)

        context = await mm.load_memory_context(session, USER_ID, "test query")

        assert "近期对话" in context
        assert "用户问题 1" in context
        assert "助手回答 3" in context

    @pytest.mark.asyncio
    async def test_includes_long_term(self, session, mm, monkeypatch):
        mock_hit = MagicMock()
        mock_hit.summary = "用户之前学习了Transformer的Self-Attention机制"
        mock_hit.score = 0.8

        mock_emb = MagicMock()
        mock_emb.encode.return_value = _fake_encode_result()
        mock_milvus = MagicMock()
        mock_milvus.hybrid_search.return_value = [mock_hit]

        monkeypatch.setattr(mm, "embedding_service", mock_emb)
        monkeypatch.setattr(mm, "memory_milvus_service", mock_milvus)

        context = await mm.load_memory_context(session, USER_ID, "Attention是什么")

        assert "相关历史记忆" in context
        assert "Self-Attention" in context

    @pytest.mark.asyncio
    async def test_filters_by_similarity_threshold(self, session, mm, monkeypatch):
        mock_hit_high = MagicMock()
        mock_hit_high.summary = "高相关记忆"
        mock_hit_high.score = 0.8

        mock_hit_low = MagicMock()
        mock_hit_low.summary = "低相关记忆"
        mock_hit_low.score = 0.1

        mock_emb = MagicMock()
        mock_emb.encode.return_value = _fake_encode_result()
        mock_milvus = MagicMock()
        mock_milvus.hybrid_search.return_value = [mock_hit_high, mock_hit_low]

        monkeypatch.setattr(mm, "embedding_service", mock_emb)
        monkeypatch.setattr(mm, "memory_milvus_service", mock_milvus)

        context = await mm.load_memory_context(session, USER_ID, "test query")

        assert "高相关记忆" in context
        assert "低相关记忆" not in context

    @pytest.mark.asyncio
    async def test_both_short_and_long_term(self, session, mm, monkeypatch):
        await _populate_rounds(session, 2)

        mock_hit = MagicMock()
        mock_hit.summary = "历史记忆摘要"
        mock_hit.score = 0.5

        mock_emb = MagicMock()
        mock_emb.encode.return_value = _fake_encode_result()
        mock_milvus = MagicMock()
        mock_milvus.hybrid_search.return_value = [mock_hit]

        monkeypatch.setattr(mm, "embedding_service", mock_emb)
        monkeypatch.setattr(mm, "memory_milvus_service", mock_milvus)

        context = await mm.load_memory_context(session, USER_ID, "test query")

        assert "近期对话" in context
        assert "相关历史记忆" in context

    @pytest.mark.asyncio
    async def test_long_term_failure_graceful(self, session, mm, monkeypatch):
        """If Milvus fails, should still return short-term context."""
        await _populate_rounds(session, 2)

        mock_emb = MagicMock()
        mock_emb.encode.side_effect = RuntimeError("Milvus down")

        monkeypatch.setattr(mm, "embedding_service", mock_emb)

        context = await mm.load_memory_context(session, USER_ID, "test query")

        assert "近期对话" in context
        assert "相关历史记忆" not in context


# ── save_conversation tests ──────────────────────────────────


class TestSaveConversation:
    @pytest.mark.asyncio
    async def test_saves_to_short_term(self, session, mm, monkeypatch):
        from app.db.crud import count_rounds, get_recent_conversations

        mock_settings = MagicMock()
        mock_settings.memory_short_term_limit = 20
        mock_settings.memory_consolidate_count = 10
        monkeypatch.setattr(mm, "settings", mock_settings)

        await mm.save_conversation(
            session=session,
            user_id=USER_ID,
            session_id=SESSION_ID,
            user_message="什么是BERT?",
            assistant_message="BERT是一种预训练模型。",
        )

        total = await count_rounds(session, USER_ID)
        assert total == 1

        rows = await get_recent_conversations(session, USER_ID, limit_rounds=10)
        user_row = [r for r in rows if r.role == "user"][0]
        assistant_row = [r for r in rows if r.role == "assistant"][0]
        assert user_row.content == "什么是BERT?"
        assert assistant_row.content == "BERT是一种预训练模型。"

    @pytest.mark.asyncio
    async def test_increments_round_number(self, session, mm, monkeypatch):
        from app.db.crud import count_rounds, get_recent_conversations

        mock_settings = MagicMock()
        mock_settings.memory_short_term_limit = 20
        mock_settings.memory_consolidate_count = 10
        monkeypatch.setattr(mm, "settings", mock_settings)

        await mm.save_conversation(session, USER_ID, SESSION_ID, "问题1", "回答1")
        await mm.save_conversation(session, USER_ID, SESSION_ID, "问题2", "回答2")

        total = await count_rounds(session, USER_ID)
        assert total == 2

        rows = await get_recent_conversations(session, USER_ID, limit_rounds=10)
        round_numbers = sorted({r.round_number for r in rows})
        assert round_numbers == [1, 2]


# ── consolidation tests ─────────────────────────────────────


class TestConsolidation:
    @pytest.mark.asyncio
    async def test_triggers_when_over_limit(self, session, mm, monkeypatch):
        """When short-term exceeds limit, oldest rounds should be consolidated."""
        from app.db.crud import count_rounds

        await _populate_rounds(session, 20)

        consolidate_response = json.dumps({
            "summary": "用户学习了多个概念，从问题1到问题10的讨论涵盖了基础知识。",
            "topics": ["基础概念"],
        })

        mock_settings = MagicMock()
        mock_settings.memory_short_term_limit = 20
        mock_settings.memory_consolidate_count = 10

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_llm_response(consolidate_response))

        mock_emb = MagicMock()
        mock_emb.encode.return_value = _fake_encode_result()

        mock_milvus = MagicMock()
        mock_milvus.insert.return_value = "new_memory_id"

        monkeypatch.setattr(mm, "settings", mock_settings)
        monkeypatch.setattr(mm, "llm_client", mock_llm)
        monkeypatch.setattr(mm, "embedding_service", mock_emb)
        monkeypatch.setattr(mm, "memory_milvus_service", mock_milvus)

        await mm.save_conversation(
            session=session,
            user_id=USER_ID,
            session_id=SESSION_ID,
            user_message="第21个问题",
            assistant_message="第21个回答",
        )

        # LLM should have been called for summarization
        mock_llm.chat.assert_called_once()

        # Milvus should have received the memory
        mock_milvus.insert.assert_called_once()
        call_kwargs = mock_milvus.insert.call_args.kwargs
        assert call_kwargs["user_id"] == USER_ID
        assert "基础概念" in call_kwargs["topics"]

        # After consolidation: 21 total - 10 deleted = 11 rounds
        total = await count_rounds(session, USER_ID)
        assert total == 11

    @pytest.mark.asyncio
    async def test_no_trigger_when_under_limit(self, session, mm, monkeypatch):
        """Should not consolidate when under the limit."""
        from app.db.crud import count_rounds

        await _populate_rounds(session, 5)

        mock_settings = MagicMock()
        mock_settings.memory_short_term_limit = 20
        mock_settings.memory_consolidate_count = 10

        mock_llm = MagicMock()

        monkeypatch.setattr(mm, "settings", mock_settings)
        monkeypatch.setattr(mm, "llm_client", mock_llm)

        await mm.save_conversation(session, USER_ID, SESSION_ID, "问题", "回答")

        # LLM should NOT have been called
        mock_llm.chat.assert_not_called()

        total = await count_rounds(session, USER_ID)
        assert total == 6

    @pytest.mark.asyncio
    async def test_consolidation_llm_failure_graceful(self, session, mm, monkeypatch):
        """If LLM fails during consolidation, short-term data should be preserved."""
        from app.db.crud import count_rounds

        await _populate_rounds(session, 20)

        mock_settings = MagicMock()
        mock_settings.memory_short_term_limit = 20
        mock_settings.memory_consolidate_count = 10

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=RuntimeError("LLM error"))

        monkeypatch.setattr(mm, "settings", mock_settings)
        monkeypatch.setattr(mm, "llm_client", mock_llm)
        monkeypatch.setattr(mm, "embedding_service", MagicMock())
        monkeypatch.setattr(mm, "memory_milvus_service", MagicMock())

        await mm.save_conversation(session, USER_ID, SESSION_ID, "问题", "回答")

        # All rounds should still be there (no deletion on failure)
        total = await count_rounds(session, USER_ID)
        assert total == 21

    @pytest.mark.asyncio
    async def test_consolidation_milvus_failure_preserves_data(self, session, mm, monkeypatch):
        """If Milvus write fails, short-term data should be preserved."""
        from app.db.crud import count_rounds

        await _populate_rounds(session, 20)

        consolidate_response = json.dumps({
            "summary": "摘要内容",
            "topics": ["topic1"],
        })

        mock_settings = MagicMock()
        mock_settings.memory_short_term_limit = 20
        mock_settings.memory_consolidate_count = 10

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_llm_response(consolidate_response))

        mock_emb = MagicMock()
        mock_emb.encode.return_value = _fake_encode_result()

        mock_milvus = MagicMock()
        mock_milvus.insert.side_effect = RuntimeError("Milvus write failed")

        monkeypatch.setattr(mm, "settings", mock_settings)
        monkeypatch.setattr(mm, "llm_client", mock_llm)
        monkeypatch.setattr(mm, "embedding_service", mock_emb)
        monkeypatch.setattr(mm, "memory_milvus_service", mock_milvus)

        await mm.save_conversation(session, USER_ID, SESSION_ID, "问题", "回答")

        # Data should be preserved since Milvus write failed
        total = await count_rounds(session, USER_ID)
        assert total == 21

    @pytest.mark.asyncio
    async def test_consolidation_deletes_correct_rounds(self, session, mm, monkeypatch):
        """After consolidation, the oldest 10 rounds should be gone, newest kept."""
        from app.db.crud import get_recent_conversations

        await _populate_rounds(session, 20)

        consolidate_response = json.dumps({
            "summary": "概括摘要",
            "topics": ["test"],
        })

        mock_settings = MagicMock()
        mock_settings.memory_short_term_limit = 20
        mock_settings.memory_consolidate_count = 10

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_llm_response(consolidate_response))

        mock_emb = MagicMock()
        mock_emb.encode.return_value = _fake_encode_result()

        mock_milvus = MagicMock()
        mock_milvus.insert.return_value = "memory_id"

        monkeypatch.setattr(mm, "settings", mock_settings)
        monkeypatch.setattr(mm, "llm_client", mock_llm)
        monkeypatch.setattr(mm, "embedding_service", mock_emb)
        monkeypatch.setattr(mm, "memory_milvus_service", mock_milvus)

        await mm.save_conversation(session, USER_ID, SESSION_ID, "第21个问题", "第21个回答")

        # Should have rounds 11-21 remaining
        rows = await get_recent_conversations(session, USER_ID, limit_rounds=50)
        remaining_rounds = sorted({r.round_number for r in rows})
        assert min(remaining_rounds) == 11
        assert max(remaining_rounds) == 21
        assert len(remaining_rounds) == 11
