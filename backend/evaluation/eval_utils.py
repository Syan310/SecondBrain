"""Shared utilities for evaluation scripts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

EVAL_TZ = ZoneInfo("America/Los_Angeles")

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_run_logging(log_file: Path) -> None:
    """Console + one UTF-8 file per run (root logger, captures httpx/sqlalchemy/app)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    root.addHandler(sh)


async def warmup_and_health_check(
    *,
    check_db: bool = False,
    _logger: logging.Logger | None = None,
) -> list[str]:
    """Pre-flight: warm up GPU models + verify infrastructure.

    Returns a list of issues (empty means all checks passed).
    Raises nothing — callers decide whether to abort.
    """
    log = _logger or logging.getLogger(__name__)
    issues: list[str] = []

    # 1. Warm up embedding model (single-threaded, prevents concurrent first-load)
    try:
        from app.services.embedding import embedding_service
        log.info("Health check: warming up embedding model...")
        embedding_service.warmup()
        log.info("Health check: embedding model OK")
    except Exception as e:
        issues.append(f"embedding warmup failed: {e}")
        log.error("Health check FAIL: embedding — %s", e)

    # 2. Warm up reranker model
    try:
        from app.services.reranker import reranker_service
        log.info("Health check: warming up reranker model...")
        reranker_service.warmup()
        log.info("Health check: reranker model OK")
    except Exception as e:
        issues.append(f"reranker warmup failed: {e}")
        log.error("Health check FAIL: reranker — %s", e)

    # 3. Milvus collection has data
    try:
        from app.services.embedding import embedding_service
        from app.services.milvus_client import milvus_service
        results = milvus_service.hybrid_search(
            query_vector=embedding_service.encode_dense("health check")[0].tolist(),
            query_text="health check",
            top_k=1,
        )
        if not results:
            issues.append("Milvus collection is empty (0 chunks)")
            log.error("Health check FAIL: Milvus collection empty")
        else:
            log.info("Health check: Milvus collection has data")
    except Exception as e:
        issues.append(f"Milvus health check failed: {e}")
        log.error("Health check FAIL: Milvus — %s", e)

    # 4. knowledge_concepts > 0 (only needed for agent eval)
    if check_db:
        try:
            from app.db.database import async_session_factory
            from app.agent.tools.search_concepts import search_concepts
            async with async_session_factory() as session:
                concepts = await search_concepts(session, scope=None, keyword=None)
                if not concepts:
                    issues.append("knowledge_concepts table is empty")
                    log.error("Health check FAIL: knowledge_concepts empty")
                else:
                    log.info("Health check: %d knowledge concepts found", len(concepts))
        except Exception as e:
            issues.append(f"knowledge_concepts check failed: {e}")
            log.error("Health check FAIL: knowledge_concepts — %s", e)

    if issues:
        log.warning("Health check completed with %d issue(s)", len(issues))
    else:
        log.info("Health check: ALL PASSED")
    return issues


def safe_p50(values: list[float]) -> float:
    """Return the median value, or 0 for empty lists."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def safe_p95(values: list[float]) -> float:
    """Return the 95th-percentile value, or 0 for empty lists."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * 0.95), len(ordered) - 1)
    return ordered[idx]
