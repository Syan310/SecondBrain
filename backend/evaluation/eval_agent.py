"""Agent Evaluation — task-driven metrics.

Evaluates:
  1. Intent Classification Accuracy (Orchestrator routing)
  2. Task Success Rate (E2E via LLM-as-Judge)
  3. E2E Latency

Usage:
    cd backend
    MILVUS_HOST=localhost python -m evaluation.eval_agent
    MILVUS_HOST=localhost python -m evaluation.eval_agent --profile smoke

Each run writes a timestamped log under ``evaluation/results/agentlog/eval_agent_YYYYMMDD_HHMMSS.log``
(UTF-8) and one timestamped JSON report under ``evaluation/results/agent/``.
Timestamps use America/Los_Angeles local wall clock.

Requires Milvus, embedding/reranker models, and LLM APIs to be available.
"""

from __future__ import annotations

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any

from app.services.milvus_client import milvus_service, memory_milvus_service
from app.agent.graph import run_orchestrator
from app.agent.nodes.orchestrator import plan_node
from app.db.database import async_session_factory
from evaluation.judge_client import judge_client
from evaluation.profiles import build_intent_eval_subset, build_stratified_subset, get_eval_profile

logger = logging.getLogger(__name__)

DATASET_DIR = Path(__file__).parent / "datasets"
OUTPUT_DIR = Path(__file__).parent / "results"
AGENT_OUTPUT_DIR = OUTPUT_DIR / "agent"
LOG_DIR = OUTPUT_DIR / "agentlog"
EVAL_TZ = ZoneInfo("America/Los_Angeles")

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
def _configure_run_logging(log_file: Path) -> None:
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


def _agent_report_path(run_ts: str) -> Path:
    """Single JSON report path for one eval run (under results/agent/)."""
    return AGENT_OUTPUT_DIR / f"agent_eval_{run_ts}.json"


# ── 1. Intent Classification ──────────────────────────────────


def _safe_p50(values: list[float]) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _safe_p95(values: list[float]) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(int(len(ordered) * 0.95), len(ordered) - 1)
    return ordered[idx]


async def classify_intent(query: str) -> dict[str, Any]:
    """Evaluate real orchestrator planning instead of a parallel prompt."""
    plan_update = await plan_node(
        {
            "user_query": query,
            "user_id": "eval_intent",
            "messages": [],
            "memory_context": "",
        }
    )
    plan = plan_update.get("plan", [])
    step_intents = [step.get("intent", "") for step in plan]
    primary_intent = step_intents[0] if len(step_intents) == 1 else ""
    return {
        "primary_intent": primary_intent,
        "step_intents": step_intents,
        "plan_length": len(plan),
        "plan": plan,
    }


async def eval_intent_classification(dataset: list[dict]) -> dict:
    """Evaluate intent classification accuracy."""
    correct = 0
    results = []

    for i, item in enumerate(dataset):
        query = item["query"]
        expected = item["expected_intent"]
        logger.info("  [%d/%d] %s -> expected: %s", i + 1, len(dataset), query[:50], expected)

        predicted = await classify_intent(query)
        is_correct = predicted["primary_intent"] == expected and predicted["plan_length"] == 1
        if is_correct:
            correct += 1

        results.append({
            "query": query,
            "expected": expected,
            "predicted": predicted["primary_intent"],
            "predicted_step_intents": predicted["step_intents"],
            "plan_length": predicted["plan_length"],
            "correct": is_correct,
        })
        logger.info(
            "    predicted: %s steps=%s %s",
            predicted["primary_intent"] or "(multi/invalid)",
            predicted["step_intents"],
            "✓" if is_correct else "✗",
        )

    accuracy = correct / len(dataset) if dataset else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(dataset),
        "details": results,
    }


# ── 2. Task Success Rate (E2E + LLM-as-Judge) ─────────────────


JUDGE_PROMPT = """\
你是一个评估专家。请判断AI助手的回答是否成功完成了用户的任务。

评分标准：
- **pass**: 回答准确、完整地回答了用户问题，关键信息无遗漏
- **partial**: 回答部分正确但有遗漏或不够准确
- **fail**: 回答与问题无关、严重错误、或未能完成任务

用户问题: {question}

参考答案（标准答案）: {ground_truth}

AI助手的回答: {answer}

请按以下JSON格式输出评判结果：
{{"verdict": "pass|partial|fail", "reason": "简要原因"}}

只输出JSON，不要其他文字。"""


async def judge_task_success(question: str, ground_truth: str, answer: str) -> dict:
    """Use the configured GPT-5.4 judge to evaluate task completion."""
    prompt = JUDGE_PROMPT.format(
        question=question, ground_truth=ground_truth, answer=answer
    )
    resp = await judge_client.judge(
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=1024,
    )

    # Parse JSON
    resp = resp.strip()
    if resp.startswith("```"):
        lines = resp.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        resp = "\n".join(lines).strip()

    try:
        result = json.loads(resp)
    except json.JSONDecodeError:
        # Try extracting JSON
        start = resp.find("{")
        end = resp.rfind("}")
        if start != -1 and end != -1:
            try:
                result = json.loads(resp[start:end + 1])
            except json.JSONDecodeError:
                result = None
        else:
            result = None

    if not isinstance(result, dict):
        return {"verdict": "error", "reason": f"Failed to parse judge response: {resp[:100]}"}

    verdict = str(result.get("verdict", "error")).strip().lower()
    if verdict not in {"pass", "partial", "fail"}:
        return {"verdict": "error", "reason": f"Invalid judge verdict: {result}"}
    return {
        "verdict": verdict,
        "reason": str(result.get("reason", "")).strip(),
    }


async def eval_task_success(qa_dataset: list[dict]) -> dict:
    """Run isolated QA E2E evaluation and keep judge errors separate."""
    results = []
    latencies = []
    verdicts = {"pass": 0, "partial": 0, "fail": 0, "error": 0}
    judge_error_total = 0
    orchestrator_error_total = 0

    for i, item in enumerate(qa_dataset):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        logger.info("  [%d/%d] %s", i + 1, len(qa_dataset), question[:60])
        sample_id = f"qa_eval_{i + 1:03d}"
        user_id = f"{sample_id}_user"
        session_id = f"{sample_id}_session"

        # Run orchestrator E2E
        t0 = time.perf_counter()
        orchestrator_failed = False
        try:
            async with async_session_factory() as session:
                result = await run_orchestrator(
                    question,
                    user_id=user_id,
                    session=session,
                    session_id=session_id,
                )
                answer = result.get("final_answer", "")
        except Exception as e:
            logger.error("    Orchestrator error: %s", e)
            orchestrator_failed = True
            orchestrator_error_total += 1
            answer = f"Error: {type(e).__name__}: {e}"
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)

        judge_call_failed = False
        judge_skipped = False
        if orchestrator_failed:
            judge_skipped = True
            judge_result = {
                "verdict": "fail",
                "reason": "orchestrator_exception: assistant pipeline failed before judge",
            }
        else:
            try:
                judge_result = await judge_task_success(question, ground_truth, answer)
            except Exception as e:
                judge_call_failed = True
                judge_error_total += 1
                logger.warning(
                    "    Judge call failed (recorded as error, continuing): %s: %s",
                    type(e).__name__,
                    e,
                )
                judge_result = {
                    "verdict": "error",
                    "reason": f"judge_exception: {type(e).__name__}: {e}",
                }

        verdict = judge_result.get("verdict", "error")
        verdicts[verdict] = verdicts.get(verdict, 0) + 1

        results.append({
            "sample_id": sample_id,
            "question": question,
            "question_type": item.get("question_type", "unknown"),
            "ground_truth": ground_truth,
            "answer": answer,
            "verdict": verdict,
            "reason": judge_result.get("reason", ""),
            "latency_ms": latency_ms,
            "user_id": user_id,
            "session_id": session_id,
            "orchestrator_failed": orchestrator_failed,
            "judge_call_failed": judge_call_failed,
            "judge_skipped": judge_skipped,
        })
        logger.info("    verdict: %s (%.0fms)", verdict, latency_ms)

    total = len(qa_dataset)
    evaluated_total = total - verdicts["error"]

    return {
        "pass_rate_all": verdicts["pass"] / total if total else 0,
        "partial_rate_all": verdicts["partial"] / total if total else 0,
        "fail_rate_all": verdicts["fail"] / total if total else 0,
        "judge_error_rate": verdicts["error"] / total if total else 0,
        "pass_rate_evaluated": verdicts["pass"] / evaluated_total if evaluated_total else 0,
        "partial_rate_evaluated": verdicts["partial"] / evaluated_total if evaluated_total else 0,
        "fail_rate_evaluated": verdicts["fail"] / evaluated_total if evaluated_total else 0,
        "verdicts": verdicts,
        "evaluated_total": evaluated_total,
        "judge_error_total": judge_error_total,
        "orchestrator_error_total": orchestrator_error_total,
        "latency_p50_ms": _safe_p50(latencies),
        "latency_p95_ms": _safe_p95(latencies),
        "total": total,
        "details": results,
    }


# ── 3. Main ───────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent evaluation with a preset profile.")
    parser.add_argument(
        "--profile",
        choices=("smoke", "standard", "full"),
        default=os.environ.get("EVAL_PROFILE", "standard"),
        help="Evaluation profile preset.",
    )
    return parser.parse_args()


async def main(run_log_file: Path, profile_name: str) -> None:
    t_start = datetime.now(EVAL_TZ)
    started_at = t_start.isoformat()
    run_ts = t_start.strftime("%Y%m%d_%H%M%S")
    profile = get_eval_profile(profile_name)
    logger.info("=== eval_agent run start ===")
    logger.info("Log file: %s", run_log_file.resolve())
    logger.info("cwd=%s python=%s", os.getcwd(), sys.version.split()[0])
    logger.info("MILVUS_HOST=%s", os.environ.get("MILVUS_HOST", "(unset)"))
    logger.info("EVAL_PROFILE=%s", profile.name)

    # Init services
    logger.info("Connecting to Milvus...")
    milvus_service.connect()
    milvus_service.init_collection()
    try:
        memory_milvus_service.init_collection()
    except Exception:
        logger.warning("Memory collection init failed, continuing without memory.")

    # Initialize SQLite
    from app.db.database import init_db
    await init_db()

    # Load datasets
    intent_path = DATASET_DIR / "intent_dataset.json"
    qa_path = DATASET_DIR / "qa_dataset.json"

    with open(intent_path, "r", encoding="utf-8") as f:
        intent_dataset = json.load(f)
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)

    logger.info("Loaded %d intent items, %d QA items.", len(intent_dataset), len(qa_dataset))
    intent_subset, intent_subset_meta = build_intent_eval_subset(
        intent_dataset,
        profile.agent_intent_limit,
    )
    qa_subset, qa_subset_meta = build_stratified_subset(
        qa_dataset,
        profile.agent_qa_single_hop_limit,
        profile.agent_qa_multi_hop_limit,
    )

    # ── Phase 1: Intent Classification ──
    logger.info("=== Phase 1: Intent Classification ===")
    logger.info(
        "Running intent eval on %d items (limit=%s)...",
        intent_subset_meta["intent_selected"],
        intent_subset_meta["intent_limit"],
    )
    intent_results = await eval_intent_classification(intent_subset)
    logger.info("Intent Accuracy: %.1f%% (%d/%d)",
                intent_results["accuracy"] * 100,
                intent_results["correct"],
                intent_results["total"])

    # ── Phase 2: QA E2E Task Success (stratified subset) ──
    logger.info("=== Phase 2: QA E2E Task Success ===")
    logger.info(
        "Running E2E on %d QA items (single-hop=%d, multi-hop=%d)...",
        qa_subset_meta["total_selected"],
        qa_subset_meta["single_hop_selected"],
        qa_subset_meta["multi_hop_selected"],
    )
    task_results = await eval_task_success(qa_subset)
    logger.info("QA E2E Pass Rate (all): %.1f%%", task_results["pass_rate_all"] * 100)
    logger.info("QA E2E Pass Rate (judge-ok): %.1f%%", task_results["pass_rate_evaluated"] * 100)

    # ── Save results ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AGENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _agent_report_path(run_ts)
    finished_at = datetime.now(EVAL_TZ).isoformat()
    full_report = {
        "meta": {
            "run_log_file": str(run_log_file.resolve()),
            "timezone": "America/Los_Angeles",
            "started_at": started_at,
            "finished_at": finished_at,
            "profile": profile.name,
            "output_file": str(output_path.resolve()),
        },
        "intent_classification": {
            "metric_scope": "single-intent plan_node accuracy",
            "subset": intent_subset_meta,
            "accuracy": intent_results["accuracy"],
            "correct": intent_results["correct"],
            "total": intent_results["total"],
            "details": intent_results["details"],
        },
        "qa_e2e_task_success": {
            "metric_scope": "isolated QA subset E2E + GPT-5.4 judge",
            "subset": qa_subset_meta,
            "pass_rate_all": task_results["pass_rate_all"],
            "partial_rate_all": task_results["partial_rate_all"],
            "fail_rate_all": task_results["fail_rate_all"],
            "judge_error_rate": task_results["judge_error_rate"],
            "pass_rate_evaluated": task_results["pass_rate_evaluated"],
            "partial_rate_evaluated": task_results["partial_rate_evaluated"],
            "fail_rate_evaluated": task_results["fail_rate_evaluated"],
            "verdicts": task_results["verdicts"],
            "evaluated_total": task_results["evaluated_total"],
            "judge_error_total": task_results["judge_error_total"],
            "orchestrator_error_total": task_results["orchestrator_error_total"],
            "latency_p50_ms": task_results["latency_p50_ms"],
            "latency_p95_ms": task_results["latency_p95_ms"],
            "total": task_results["total"],
            "details": task_results["details"],
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Agent report saved to %s", output_path)

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("AGENT EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info("  Intent Accuracy:   %.1f%%  (target ≥ 95%%)", intent_results["accuracy"] * 100)
    logger.info("  QA Pass Rate(all): %.1f%%  (target ≥ 80%%)", task_results["pass_rate_all"] * 100)
    logger.info("  QA Pass Rate(ok):  %.1f%%", task_results["pass_rate_evaluated"] * 100)
    logger.info("  E2E Latency P50:   %.0f ms", task_results["latency_p50_ms"])
    logger.info("  E2E Latency P95:   %.0f ms (target < 2000ms)", task_results["latency_p95_ms"])
    logger.info("  Verdicts:          %s", task_results["verdicts"])
    logger.info("  QA Subset:         %s", qa_subset_meta)
    logger.info("=" * 60)
    logger.info("=== eval_agent run end; full log: %s ===", run_log_file.resolve())


if __name__ == "__main__":
    args = parse_args()
    _ts = datetime.now(EVAL_TZ).strftime("%Y%m%d_%H%M%S")
    _log_path = LOG_DIR / f"eval_agent_{_ts}.log"
    _configure_run_logging(_log_path)
    try:
        asyncio.run(main(_log_path, args.profile))
    except Exception:
        logging.exception("eval_agent aborted with exception (see log file above)")
        raise
