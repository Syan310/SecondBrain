"""Tests for the gap classification timing script (temporary SQLite file, no LLM)."""

import gc

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base, ConceptSource, KnowledgeConcept, UserMastery
from evaluation import time_gap_classify as tg


def test_percentile_interpolates_like_numpy_default():
    values = [float(v) for v in range(1, 11)]  # 1..10
    assert tg.percentile(values, 50) == pytest.approx(5.5)
    assert tg.percentile(values, 95) == pytest.approx(9.55)
    assert tg.percentile([3.0], 95) == 3.0
    with pytest.raises(ValueError):
        tg.percentile([], 50)


def test_summarize_ms():
    s = tg.summarize_ms([1.0, 2.0, 3.0])
    assert s["n"] == 3 and s["p50_ms"] == 2.0 and s["max_ms"] == 3.0 and s["min_ms"] == 1.0


def test_gen2_gc_timer_records_only_full_collections_and_unregisters():
    with tg.Gen2GcTimer() as timer:
        gc.collect(0)
        gc.collect(2)
    assert len(timer.pauses_ms) == 1 and timer.pauses_ms[0] >= 0
    assert timer not in gc.callbacks


@pytest_asyncio.fixture
async def temp_db(tmp_path, monkeypatch):
    """A file SQLite DB with 4 concepts (3 in a.md), patched into the script."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 't.db').as_posix()}", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        for i, doc in enumerate(["a.md", "a.md", "a.md", "b.md"]):
            concept = KnowledgeConcept(topic=f"t{i}")
            s.add(concept)
            await s.flush()
            s.add(ConceptSource(concept_id=concept.id, source_doc=doc, parent_chunk_id=f"p{i}"))
        s.add(UserMastery(user_id="someone_else", topic="t0", mastery_score=0.9))
        await s.commit()
    monkeypatch.setattr(tg, "async_session_factory", factory)
    monkeypatch.setattr(tg, "engine", engine)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_measure_times_both_groups_and_run_cleans_up(temp_db):
    result = await tg.measure(scope=None, runs=3)
    assert result["topics_loaded"] == 4
    assert result["mastery_records"] == 4
    assert result["configured_echo"] is False
    for group in ("echo_on", "echo_off"):
        g = result["groups"][group]
        assert g["total_topics"] == 4 and g["n"] == 3 and len(g["times_ms"]) == 3
        assert len(g["gen2_gc_ms"]) == 3
        assert g["calls_with_gen2_gc"] == sum(1 for x in g["gen2_gc_ms"] if x > 0)
        # __half_weak__ on sorted topics: t0, t1 mastered (0.8); t2, t3 weak (0.15)
        assert g["categories"] == {"未接触盲区": 0, "薄弱盲区": 2, "需巩固": 0, "已掌握": 2}

    assert await tg.delete_test_user() == 0
    async with temp_db() as s:
        rows = (await s.execute(select(UserMastery.user_id))).scalars().all()
    assert rows == ["someone_else"]  # other users are untouched


@pytest.mark.asyncio
async def test_measure_with_scope_counts_only_that_source(temp_db):
    result = await tg.measure(scope="a.md", runs=1)
    assert result["topics_loaded"] == 3
    assert result["groups"]["echo_off"]["total_topics"] == 3
    await tg.delete_test_user()
