"""Tests for LLM-based concept extraction and new concept CRUD.

Tests:
  1. concept_extractor: parse response, handle edge cases
  2. CRUD: upsert_concepts, delete_concepts_by_source, search_concepts,
     get_all_concept_topics, list_documents with new schema
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base, ConceptSource, KnowledgeConcept
from app.db.crud import (
    delete_concepts_by_source,
    get_all_concept_topics,
    list_documents,
    search_concepts,
    upsert_concepts,
)
from app.services.concept_extractor import _parse_response, extract_concepts


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


# ── _parse_response tests ───────────────────────────────────


class TestParseResponse:
    def test_valid_json_array(self):
        content = '["Self-Attention机制", "Transformer位置编码"]'
        assert _parse_response(content) == ["Self-Attention机制", "Transformer位置编码"]

    def test_with_code_fences(self):
        content = '```json\n["概念A", "概念B"]\n```'
        assert _parse_response(content) == ["概念A", "概念B"]

    def test_empty_array(self):
        assert _parse_response("[]") == []

    def test_invalid_json(self):
        assert _parse_response("not json") == []

    def test_non_list_response(self):
        assert _parse_response('{"key": "value"}') == []

    def test_deduplicates(self):
        content = '["A", "A", "B"]'
        assert _parse_response(content) == ["A", "B"]

    def test_filters_non_strings(self):
        content = '["有效概念", 123, null, "另一个"]'
        assert _parse_response(content) == ["有效概念", "另一个"]

    def test_strips_whitespace(self):
        content = '["  概念A  ", "概念B"]'
        assert _parse_response(content) == ["概念A", "概念B"]

    def test_filters_empty_strings(self):
        content = '["概念A", "", "  ", "概念B"]'
        assert _parse_response(content) == ["概念A", "概念B"]


# ── extract_concepts tests ──────────────────────────────────


class TestExtractConcepts:
    @pytest.mark.asyncio
    async def test_empty_text_returns_empty(self):
        result = await extract_concepts("", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_calls_llm_and_parses(self):
        mock_response = {
            "content": '["Self-Attention机制", "残差连接"]',
        }
        with patch("app.services.concept_extractor.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=mock_response)
            result = await extract_concepts("Transformer uses self-attention...", [])

        assert result == ["Self-Attention机制", "残差连接"]
        mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_existing_concepts_passed_to_prompt(self):
        mock_response = {"content": '["新概念"]'}
        with patch("app.services.concept_extractor.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=mock_response)
            await extract_concepts("some text", ["已有概念A", "已有概念B"])

        call_args = mock_llm.chat.call_args
        system_msg = call_args.kwargs["messages"][0]["content"]
        assert "已有概念A" in system_msg
        assert "已有概念B" in system_msg

    @pytest.mark.asyncio
    async def test_llm_returns_empty_content(self):
        mock_response = {"content": None}
        with patch("app.services.concept_extractor.llm_client") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=mock_response)
            result = await extract_concepts("some text", [])

        assert result == []


# ── CRUD tests ──────────────────────────────────────────────


class TestUpsertConcepts:
    @pytest.mark.asyncio
    async def test_insert_new_concepts(self, session):
        count = await upsert_concepts(
            session, ["概念A", "概念B"], source_doc="doc1.md", parent_chunk_id="chunk-1",
        )
        assert count == 2

        topics = await get_all_concept_topics(session)
        assert set(topics) == {"概念A", "概念B"}

    @pytest.mark.asyncio
    async def test_skip_existing_concepts(self, session):
        await upsert_concepts(session, ["概念A"], source_doc="doc1.md", parent_chunk_id="chunk-1")
        count = await upsert_concepts(
            session, ["概念A", "概念B"], source_doc="doc2.md", parent_chunk_id="chunk-2",
        )
        # Only 概念B is new
        assert count == 1

        topics = await get_all_concept_topics(session)
        assert set(topics) == {"概念A", "概念B"}

    @pytest.mark.asyncio
    async def test_creates_source_mappings(self, session):
        await upsert_concepts(session, ["概念A"], source_doc="doc1.md", parent_chunk_id="chunk-1")
        await upsert_concepts(session, ["概念A"], source_doc="doc2.md", parent_chunk_id="chunk-2")

        from sqlalchemy import select
        result = await session.execute(select(ConceptSource))
        sources = result.scalars().all()
        assert len(sources) == 2
        assert {s.source_doc for s in sources} == {"doc1.md", "doc2.md"}

    @pytest.mark.asyncio
    async def test_empty_topics_returns_zero(self, session):
        count = await upsert_concepts(session, [], source_doc="doc1.md", parent_chunk_id="chunk-1")
        assert count == 0

    @pytest.mark.asyncio
    async def test_same_chunk_same_concept_no_duplicate(self, session):
        await upsert_concepts(session, ["概念A"], source_doc="doc1.md", parent_chunk_id="chunk-1")
        await upsert_concepts(session, ["概念A"], source_doc="doc1.md", parent_chunk_id="chunk-1")

        from sqlalchemy import select
        result = await session.execute(select(ConceptSource))
        sources = result.scalars().all()
        assert len(sources) == 1


class TestDeleteConceptsBySource:
    @pytest.mark.asyncio
    async def test_delete_removes_source_mappings(self, session):
        await upsert_concepts(session, ["概念A", "概念B"], source_doc="doc1.md", parent_chunk_id="c1")
        deleted = await delete_concepts_by_source(session, "doc1.md")
        assert deleted == 2

    @pytest.mark.asyncio
    async def test_delete_removes_orphaned_concepts(self, session):
        await upsert_concepts(session, ["概念A"], source_doc="doc1.md", parent_chunk_id="c1")
        await delete_concepts_by_source(session, "doc1.md")

        topics = await get_all_concept_topics(session)
        assert topics == []

    @pytest.mark.asyncio
    async def test_delete_preserves_concepts_with_other_sources(self, session):
        await upsert_concepts(session, ["概念A"], source_doc="doc1.md", parent_chunk_id="c1")
        await upsert_concepts(session, ["概念A"], source_doc="doc2.md", parent_chunk_id="c2")

        await delete_concepts_by_source(session, "doc1.md")

        topics = await get_all_concept_topics(session)
        assert topics == ["概念A"]

    @pytest.mark.asyncio
    async def test_delete_nonexistent_source(self, session):
        deleted = await delete_concepts_by_source(session, "nonexistent.md")
        assert deleted == 0


class TestSearchConcepts:
    @pytest.mark.asyncio
    async def test_search_all(self, session):
        await upsert_concepts(session, ["概念A", "概念B"], source_doc="doc1.md", parent_chunk_id="c1")
        results = await search_concepts(session)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_by_scope(self, session):
        await upsert_concepts(session, ["概念A"], source_doc="doc1.md", parent_chunk_id="c1")
        await upsert_concepts(session, ["概念B"], source_doc="doc2.md", parent_chunk_id="c2")

        results = await search_concepts(session, scope="doc1.md")
        assert len(results) == 1
        assert results[0].topic == "概念A"

    @pytest.mark.asyncio
    async def test_search_by_keyword(self, session):
        await upsert_concepts(
            session, ["Self-Attention", "位置编码"], source_doc="doc1.md", parent_chunk_id="c1",
        )
        results = await search_concepts(session, keyword="Attention")
        assert len(results) == 1
        assert results[0].topic == "Self-Attention"

    @pytest.mark.asyncio
    async def test_search_scope_and_keyword(self, session):
        await upsert_concepts(session, ["概念A"], source_doc="doc1.md", parent_chunk_id="c1")
        await upsert_concepts(session, ["概念A变种"], source_doc="doc2.md", parent_chunk_id="c2")

        results = await search_concepts(session, scope="doc1.md", keyword="概念A")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_scope_deduplicates(self, session):
        """Same concept from multiple chunks in same doc should appear once."""
        await upsert_concepts(session, ["概念A"], source_doc="doc1.md", parent_chunk_id="c1")
        await upsert_concepts(session, ["概念A"], source_doc="doc1.md", parent_chunk_id="c2")

        results = await search_concepts(session, scope="doc1.md")
        assert len(results) == 1


class TestListDocuments:
    @pytest.mark.asyncio
    async def test_list_with_concept_counts(self, session):
        await upsert_concepts(session, ["A", "B"], source_doc="doc1.md", parent_chunk_id="c1")
        await upsert_concepts(session, ["C"], source_doc="doc2.md", parent_chunk_id="c2")

        docs = await list_documents(session)
        assert len(docs) == 2
        doc_map = {d["source_doc"]: d["concept_count"] for d in docs}
        assert doc_map["doc1.md"] == 2
        assert doc_map["doc2.md"] == 1

    @pytest.mark.asyncio
    async def test_list_shared_concept_counted_per_doc(self, session):
        """Concept shared across docs should be counted in each doc."""
        await upsert_concepts(session, ["共享概念"], source_doc="doc1.md", parent_chunk_id="c1")
        await upsert_concepts(session, ["共享概念"], source_doc="doc2.md", parent_chunk_id="c2")

        docs = await list_documents(session)
        assert len(docs) == 2
        for d in docs:
            assert d["concept_count"] == 1

    @pytest.mark.asyncio
    async def test_list_empty(self, session):
        docs = await list_documents(session)
        assert docs == []


class TestGetAllConceptTopics:
    @pytest.mark.asyncio
    async def test_returns_all_topics(self, session):
        await upsert_concepts(session, ["A", "B"], source_doc="doc1.md", parent_chunk_id="c1")
        await upsert_concepts(session, ["C"], source_doc="doc2.md", parent_chunk_id="c2")

        topics = await get_all_concept_topics(session)
        assert set(topics) == {"A", "B", "C"}

    @pytest.mark.asyncio
    async def test_empty_db(self, session):
        topics = await get_all_concept_topics(session)
        assert topics == []
