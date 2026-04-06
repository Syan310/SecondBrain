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
