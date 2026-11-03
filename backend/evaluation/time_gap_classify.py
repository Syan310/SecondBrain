"""Time the deterministic gap classification step (gap_agent._classify_topics). No LLM calls.

Only the classification step is timed (search_concepts + read_mastery + Python
rules). The LLM report that run_gap_agent writes afterwards is not included.

Steps:
  1. Load all concept topics (optionally limited by --scope = one source_doc).
  2. eval_agent._setup_mastery() gives the test user the "__half_weak__" preset
     (first half of the sorted topics 0.8, second half 0.15).
  3. For each group (SQL echo on / off): 1 warm-up call, then --runs timed calls
     with time.perf_counter(). Each call gets a new AsyncSession, as one request would.
  4. Delete every mastery record of the test user (also on errors) and check
     that 0 are left.

Garbage collection: gc stays on, as in the backend. gc.callbacks records every
generation-2 collection, so each timed call says how many ms of it were a full
GC pause. The summary is given with all calls and without the calls that
contained a generation-2 collection.

SQL echo: app/db/database.py turns on SQLAlchemy echo when APP_ENV=development
(the default). The "echo_on" group uses the engine as configured; the "echo_off"
group sets engine.echo = False in this process only.

Usage:
    cd backend
    PYTHONPATH=. .venv/Scripts/python -m evaluation.time_gap_classify
    SQLITE_DB_PATH=./data/secondbrain_l1.db PYTHONPATH=. .venv/Scripts/python -m evaluation.time_gap_classify
    PYTHONPATH=. .venv/Scripts/python -m evaluation.time_gap_classify --cleanup-only
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import sqlalchemy
from sqlalchemy import delete, func, select

from app.agent.subagents.gap_agent import _classify_topics
from app.agent.tools.search_concepts import search_concepts
from app.config import settings
from app.db.database import async_session_factory, engine
from app.db.models import UserMastery
from app.services import tracing
from evaluation.eval_agent import _setup_mastery
from evaluation.eval_rag import OUTPUT_DIR, _git
from evaluation.eval_utils import EVAL_TZ

logger = logging.getLogger(__name__)

GAP_TIMING_OUTPUT_DIR = OUTPUT_DIR / "gap_timing"
TEST_USER = "timing_gap_user"
PRESET = "__half_weak__"
DEFAULT_RUNS = 50


# ── Stats ─────────────────────────────────────────────────────


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (numpy's default method), q in [0, 100]."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize_ms(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "max_ms": max(values),
        "min_ms": min(values),
        "mean_ms": statistics.fmean(values),
    }


def category_counts(categories: dict[str, list]) -> dict[str, int]:
    return {name: len(items) for name, items in categories.items()}


class Gen2GcTimer:
    """Context manager: records the duration of every generation-2 collection."""

    def __init__(self) -> None:
        self.pauses_ms: list[float] = []
        self._t0: float | None = None

    def __call__(self, phase: str, info: dict[str, Any]) -> None:
        if info.get("generation") != 2:
            return
        if phase == "start":
            self._t0 = time.perf_counter()
        elif self._t0 is not None:
            self.pauses_ms.append((time.perf_counter() - self._t0) * 1000)
            self._t0 = None

    def __enter__(self) -> "Gen2GcTimer":
        gc.callbacks.append(self)
        return self

    def __exit__(self, *exc: object) -> None:
        gc.callbacks.remove(self)


# ── Environment ───────────────────────────────────────────────


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def db_files() -> dict[str, Any]:
    """Resolved SQLite path and the size of the file and its -wal / -shm files."""
    db_path = Path(engine.sync_engine.url.database).resolve()
    sizes = {}
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        sizes[p.name] = p.stat().st_size if p.exists() else None
    return {"path": str(db_path), "size_bytes": sizes}


def environment() -> dict[str, Any]:
    return {
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "sqlalchemy": sqlalchemy.__version__,
        "app_env": settings.app_env,
        "sqlite": db_files(),
        "tracing_enabled": tracing.enabled,
        "gc": {"enabled": gc.isenabled(), "threshold": list(gc.get_threshold())},
    }


# ── Measurement ───────────────────────────────────────────────


async def mastery_count(user_id: str) -> int:
    async with async_session_factory() as session:
        return await session.scalar(
            select(func.count()).select_from(UserMastery).where(UserMastery.user_id == user_id)
        )


async def delete_test_user(user_id: str = TEST_USER) -> int:
    """Delete all mastery records of the test user; return how many are left (should be 0)."""
    async with async_session_factory() as session:
        await session.execute(delete(UserMastery).where(UserMastery.user_id == user_id))
        await session.commit()
    return await mastery_count(user_id)


async def time_group(user_id: str, scope: str | None, runs: int) -> dict[str, Any]:
    """1 warm-up call, then ``runs`` timed calls, each with a new session."""
    async with async_session_factory() as session:
        categories, total = await _classify_topics(session, user_id, scope)
    times_ms: list[float] = []
    gen2_gc_ms: list[float] = []
    with Gen2GcTimer() as gc_timer:
        for _ in range(runs):
            async with async_session_factory() as session:
                n0 = len(gc_timer.pauses_ms)
                t0 = time.perf_counter()
                categories, total = await _classify_topics(session, user_id, scope)
                times_ms.append((time.perf_counter() - t0) * 1000)
                gen2_gc_ms.append(sum(gc_timer.pauses_ms[n0:]))
    no_gc = [t for t, g in zip(times_ms, gen2_gc_ms) if g == 0]
    return {
        "total_topics": total,
        "categories": category_counts(categories),
        "times_ms": times_ms,
        "gen2_gc_ms": gen2_gc_ms,
        "calls_with_gen2_gc": sum(1 for g in gen2_gc_ms if g > 0),
        **summarize_ms(times_ms),
        "without_gen2_gc": summarize_ms(no_gc) if no_gc else None,
    }


async def measure(scope: str | None, runs: int) -> dict[str, Any]:
    async with async_session_factory() as session:
        concepts = await search_concepts(session, scope=scope)
        topics = sorted({c.topic for c in concepts})
        await _setup_mastery(session, TEST_USER, PRESET, topics)
        await session.commit()
    records = await mastery_count(TEST_USER)
    logger.info("test user %s: %d topics, %d mastery records", TEST_USER, len(topics), records)

    configured_echo = engine.echo
    live_objects = len(gc.get_objects())
    groups = {}
    try:
        engine.echo = True
        groups["echo_on"] = await time_group(TEST_USER, scope, runs)
        engine.echo = False
        groups["echo_off"] = await time_group(TEST_USER, scope, runs)
    finally:
        engine.echo = configured_echo
    return {
        "topics_loaded": len(topics),
        "mastery_records": records,
        "configured_echo": configured_echo,
        "gc_tracked_objects_before": live_objects,
        "groups": groups,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--scope", help="limit to one source_doc (same as the gap agent's scope)")
    parser.add_argument("--cleanup-only", action="store_true", help="only delete the test user's records")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    if args.cleanup_only:
        left = await delete_test_user()
        logger.info("deleted %s mastery records; %d left", TEST_USER, left)
        return 0 if left == 0 else 1

    started = datetime.now(EVAL_TZ)
    env = environment()
    result: dict[str, Any] | None = None
    try:
        result = await measure(args.scope, args.runs)
    finally:
        left = await delete_test_user()
        logger.info("cleanup: %s mastery records left = %d", TEST_USER, left)

    report = {
        "meta": {
            "started_at": started.isoformat(),
            "commit": {
                "sha": _git("rev-parse", "HEAD"),
                "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            },
            "test_user": TEST_USER,
            "preset": PRESET,
            "scope": args.scope,
            "runs": args.runs,
            "warmup_calls": 1,
            "timer": "time.perf_counter around _classify_topics (new AsyncSession per call)",
            "excludes": "LLM report generation (_generate_report)",
            "llm_used": False,
            "environment": env,
            "cleanup": {"mastery_records_left": left},
        },
        **result,
    }
    GAP_TIMING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    db_stem = Path(env["sqlite"]["path"]).stem
    scope_tag = "_scoped" if args.scope else ""
    out = GAP_TIMING_OUTPUT_DIR / f"gap_timing_{db_stem}{scope_tag}_{started:%Y%m%d_%H%M%S}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    for name, g in result["groups"].items():
        logger.info(
            "%s: topics=%d P50=%.2fms P95=%.2fms max=%.2fms | gen2 GC in %d calls",
            name, g["total_topics"], g["p50_ms"], g["p95_ms"], g["max_ms"], g["calls_with_gen2_gc"],
        )
    logger.info("Saved %s", out)
    return 0 if left == 0 else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
