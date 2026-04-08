"""Run RAG and Agent evaluations in parallel.

Usage:
    cd backend
    MILVUS_HOST=localhost python -m evaluation.eval_all
    MILVUS_HOST=localhost python -m evaluation.eval_all --profile smoke
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG + Agent evaluations in parallel.")
    parser.add_argument(
        "--profile",
        choices=("smoke", "standard", "full"),
        default=os.environ.get("EVAL_PROFILE", "standard"),
        help="Evaluation profile preset (passed to both eval scripts).",
    )
    return parser.parse_args()


async def _run_subprocess(label: str, module: str, profile: str) -> int:
    """Run an evaluation module as a subprocess."""
    env = {**os.environ, "EVAL_PROFILE": profile}
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", module, "--profile", profile,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    # Stream output line by line with label prefix
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip()
        print(f"[{label}] {text}", flush=True)

    await proc.wait()
    return proc.returncode or 0


async def main() -> None:
    args = parse_args()
    profile = args.profile
    print(f"=== eval_all: launching RAG + Agent evals in parallel (profile={profile}) ===")
    t0 = time.perf_counter()

    rag_task = _run_subprocess("RAG", "evaluation.eval_rag", profile)
    agent_task = _run_subprocess("AGENT", "evaluation.eval_agent", profile)

    rag_rc, agent_rc = await asyncio.gather(rag_task, agent_task)

    elapsed = time.perf_counter() - t0
    print(f"=== eval_all done in {elapsed:.0f}s ===")
    print(f"  RAG   exit code: {rag_rc}")
    print(f"  Agent exit code: {agent_rc}")

    if rag_rc != 0 or agent_rc != 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
