"""Agent Evaluation — task-driven metrics.

Evaluates:
  1. Intent Classification Accuracy (Orchestrator routing)
  2. QA E2E Task Success + Replan Efficiency (E2E via LLM-as-Judge)
  3. Quiz Question Quality (binary pass/fail via LLM-as-Judge)
  4. Gap Detection Precision (set coverage vs expected gaps)

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.milvus_client import milvus_service, memory_milvus_service
from app.agent.graph import run_orchestrator
from app.agent.nodes.orchestrator import plan_node
from app.agent.subagents.quiz_agent import run_quiz_agent
from app.agent.subagents.gap_agent import run_gap_agent
from app.agent.tools.search_concepts import search_concepts as _search_concepts
from app.db.database import async_session_factory
from app.db.models import UserMastery
from evaluation.judge_client import judge_client
from evaluation.eval_utils import EVAL_TZ, configure_run_logging, safe_p50, safe_p95, warmup_and_health_check
from evaluation.profiles import (
    build_intent_eval_subset,
    build_stratified_subset,
    get_eval_profile,
    limit_items,
)

logger = logging.getLogger(__name__)

DATASET_DIR = Path(__file__).parent / "datasets"
OUTPUT_DIR = Path(__file__).parent / "results"
AGENT_OUTPUT_DIR = OUTPUT_DIR / "agent"
LOG_DIR = OUTPUT_DIR / "agentlog"

ORCHESTRATOR_TIMEOUT_S = 120


def _agent_report_path(run_ts: str) -> Path:
    """Single JSON report path for one eval run (under results/agent/)."""
    return AGENT_OUTPUT_DIR / f"agent_eval_{run_ts}.json"


# ── 1. Intent Classification ──────────────────────────────────


async def classify_intent(query: str) -> dict[str, Any]:
    """Evaluate real orchestrator planning instead of a parallel prompt.

    Returns a dict with primary_intent, step_intents, plan_length, plan, and
    an error field (None on success).
    """
    try:
        plan_update = await plan_node(
            {
                "user_query": query,
                "user_id": "eval_intent",
                "messages": [],
                "memory_context": "",
            }
        )
    except Exception as e:
        logger.warning("    plan_node raised %s: %s", type(e).__name__, e)
        return {
            "primary_intent": "",
            "step_intents": [],
            "plan_length": 0,
            "plan": [],
            "error": f"{type(e).__name__}: {e}",
        }

    plan = plan_update.get("plan", [])
    step_intents = [step.get("intent", "") for step in plan]
    primary_intent = step_intents[0] if len(step_intents) == 1 else ""
    return {
        "primary_intent": primary_intent,
        "step_intents": step_intents,
        "plan_length": len(plan),
        "plan": plan,
        "error": None,
    }


async def eval_intent_classification(
    dataset: list[dict],
    concurrency: int = 5,
) -> dict:
    """Evaluate intent classification accuracy (parallel)."""
    sem = asyncio.Semaphore(concurrency)
    total = len(dataset)

    async def _classify_one(i: int, item: dict) -> dict:
        query = item["query"]
        expected = item["expected_intent"]
        async with sem:
            logger.info("  [intent %d/%d] %s -> expected: %s", i + 1, total, query[:50], expected)
            predicted = await classify_intent(query)

        if predicted["error"]:
            is_correct = False
        else:
            is_correct = predicted["primary_intent"] == expected and predicted["plan_length"] == 1

        logger.info(
            "  [intent %d/%d] predicted: %s steps=%s %s%s",
            i + 1, total,
            predicted["primary_intent"] or "(multi/invalid)",
            predicted["step_intents"],
            "✓" if is_correct else "✗",
            f" ERROR: {predicted['error']}" if predicted["error"] else "",
        )
        return {
            "query": query,
            "expected": expected,
            "predicted": predicted["primary_intent"],
            "predicted_step_intents": predicted["step_intents"],
            "plan_length": predicted["plan_length"],
            "correct": is_correct,
            "error": predicted["error"],
        }

    results = await asyncio.gather(*[_classify_one(i, item) for i, item in enumerate(dataset)])
    results = list(results)

    correct = sum(1 for r in results if r["correct"])
    error_count = sum(1 for r in results if r["error"])
    accuracy = correct / total if total else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "error_count": error_count,
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
    """Use the configured OpenAI judge to evaluate task completion."""
    prompt = JUDGE_PROMPT.format(
        question=question, ground_truth=ground_truth, answer=answer
    )
    resp = await judge_client.judge(
        messages=[{"role": "user", "content": prompt}],
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


async def _run_one_qa(
    sem: asyncio.Semaphore,
    i: int,
    total: int,
    item: dict,
) -> dict:
    """Run a single QA E2E case with concurrency control."""
    question = item["question"]
    ground_truth = item.get("ground_truth", "")
    sample_id = f"qa_eval_{i + 1:03d}"
    user_id = f"{sample_id}_user"
    session_id = f"{sample_id}_session"

    async with sem:
        logger.info("  [qa %d/%d] %s", i + 1, total, question[:60])

        # Run orchestrator E2E with timeout
        t0 = time.perf_counter()
        orchestrator_failed = False
        timeout_exceeded = False
        replan_count = 0
        replan_action = ""
        step_intents: list[str] = []
        try:
            async with async_session_factory() as session:
                result = await asyncio.wait_for(
                    run_orchestrator(
                        question,
                        user_id=user_id,
                        session=session,
                        session_id=session_id,
                    ),
                    timeout=ORCHESTRATOR_TIMEOUT_S,
                )
                answer = result.get("final_answer", "")
                replan_count = result.get("replan_count", 0)
                replan_action = result.get("replan_action", "")
                step_intents = [s.get("intent", "") for s in result.get("plan", [])]
        except asyncio.TimeoutError:
            logger.error("  [qa %d/%d] Orchestrator timed out after %ds", i + 1, total, ORCHESTRATOR_TIMEOUT_S)
            orchestrator_failed = True
            timeout_exceeded = True
            answer = f"Error: TimeoutError: exceeded {ORCHESTRATOR_TIMEOUT_S}s"
        except Exception as e:
            logger.error("  [qa %d/%d] Orchestrator error: %s", i + 1, total, e)
            orchestrator_failed = True
            answer = f"Error: {type(e).__name__}: {e}"
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000

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
                logger.warning(
                    "  [qa %d/%d] Judge call failed: %s: %s",
                    i + 1, total, type(e).__name__, e,
                )
                judge_result = {
                    "verdict": "error",
                    "reason": f"judge_exception: {type(e).__name__}: {e}",
                }

    verdict = judge_result.get("verdict", "error")
    logger.info("  [qa %d/%d] verdict: %s (%.0fms)", i + 1, total, verdict, latency_ms)

    return {
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
        "timeout_exceeded": timeout_exceeded,
        "judge_call_failed": judge_call_failed,
        "judge_skipped": judge_skipped,
        "replan_count": replan_count,
        "replan_action": replan_action,
        "step_intents": step_intents,
    }


async def eval_task_success(qa_dataset: list[dict], concurrency: int = 5) -> dict:
    """Run isolated QA E2E evaluation with parallel case execution."""
    total = len(qa_dataset)
    sem = asyncio.Semaphore(concurrency)
    logger.info("  QA E2E concurrency: %d", concurrency)

    results = await asyncio.gather(*[
        _run_one_qa(sem, i, total, item)
        for i, item in enumerate(qa_dataset)
    ])
    results = list(results)

    # Aggregate
    verdicts = {"pass": 0, "partial": 0, "fail": 0, "error": 0}
    ok_latencies: list[float] = []
    all_latencies: list[float] = []
    judge_error_total = 0
    orchestrator_error_total = 0
    replan_counts: list[int] = []
    replan_action_dist: dict[str, int] = {}

    for r in results:
        v = r["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1
        all_latencies.append(r["latency_ms"])
        if not r["orchestrator_failed"]:
            ok_latencies.append(r["latency_ms"])
            replan_counts.append(r["replan_count"])
            action = r["replan_action"] or "none"
            replan_action_dist[action] = replan_action_dist.get(action, 0) + 1
        if r["judge_call_failed"]:
            judge_error_total += 1
        if r["orchestrator_failed"]:
            orchestrator_error_total += 1

    evaluated_total = total - verdicts["error"]
    replan_triggered = sum(1 for c in replan_counts if c > 0)

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
        "latency_p50_ms": safe_p50(ok_latencies),
        "latency_p95_ms": safe_p95(ok_latencies),
        "latency_p50_ms_all": safe_p50(all_latencies),
        "latency_p95_ms_all": safe_p95(all_latencies),
        "replan_avg": sum(replan_counts) / len(replan_counts) if replan_counts else 0,
        "replan_rate": replan_triggered / len(replan_counts) if replan_counts else 0,
        "replan_action_distribution": replan_action_dist,
        "total": total,
        "details": results,
    }


# ── 3. Quiz Question Quality ──────────────────────────────────


QUIZ_JUDGE_PROMPT = """\
你是一个题目质量评估专家。请判断以下选择题是否合格。

## 判断标准
- 题目是否能从给定的知识材料中推导出答案
- 标注的正确答案是否确实正确
- 4 个选项是否完整且格式规范（A/B/C/D）

## 输入

主题: {topic}

知识材料:
{source_chunks}

题目: {question}

选项:
{choices}

标注正确答案: {correct_answer}

解析: {explanation}

## 输出

请按以下 JSON 格式输出：
{{"verdict": "pass|fail", "reason": "简要原因"}}

只输出 JSON，不要其他文字。"""


def _validate_quiz_structure(q: dict) -> tuple[bool, str]:
    """Rule-based structural validation for a quiz question."""
    if not q.get("question"):
        return False, "missing question text"
    choices = q.get("choices", [])
    if len(choices) != 4:
        return False, f"expected 4 choices, got {len(choices)}"
    answer = q.get("correct_answer", "").strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        return False, f"invalid correct_answer: {answer!r}"
    return True, ""


async def _judge_one_quiz_question(q: dict) -> dict:
    """Judge a single quiz question via LLM."""
    struct_ok, struct_reason = _validate_quiz_structure(q)
    if not struct_ok:
        return {"verdict": "fail", "reason": f"structure: {struct_reason}", "judge_skipped": True}

    prompt = QUIZ_JUDGE_PROMPT.format(
        topic=q.get("topic", ""),
        source_chunks="\n".join(q.get("source_chunks", [])),
        question=q["question"],
        choices="\n".join(q.get("choices", [])),
        correct_answer=q.get("correct_answer", ""),
        explanation=q.get("explanation", ""),
    )
    try:
        resp = await judge_client.judge(messages=[{"role": "user", "content": prompt}])
        resp = resp.strip()
        if resp.startswith("```"):
            lines = [l for l in resp.split("\n") if not l.strip().startswith("```")]
            resp = "\n".join(lines).strip()
        start, end = resp.find("{"), resp.rfind("}")
        result = json.loads(resp[start:end + 1]) if start != -1 else {}
        verdict = str(result.get("verdict", "error")).strip().lower()
        if verdict not in {"pass", "fail"}:
            verdict = "error"
        return {
            "verdict": verdict,
            "reason": str(result.get("reason", "")),
            "judge_skipped": False,
        }
    except Exception as e:
        return {"verdict": "error", "reason": f"judge_exception: {e}", "judge_skipped": False}


async def _run_one_quiz(
    sem: asyncio.Semaphore,
    i: int,
    total: int,
    item: dict,
) -> dict:
    """Run quiz agent for one topic and judge all generated questions."""
    topic = item["topic"]
    async with sem:
        logger.info("  [quiz %d/%d] topic=%s", i + 1, total, topic)
        t0 = time.perf_counter()
        error = None
        questions: list[dict] = []
        try:
            async with async_session_factory() as session:
                result = await asyncio.wait_for(
                    run_quiz_agent(
                        session=session,
                        user_id=f"quiz_eval_{i + 1:03d}_user",
                        topics=[topic],
                        num_questions=2,
                    ),
                    timeout=ORCHESTRATOR_TIMEOUT_S,
                )
                questions = result.get("questions", [])
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            logger.error("  [quiz %d/%d] error: %s", i + 1, total, error)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Judge each question
        question_verdicts: list[dict] = []
        if not error:
            for qi, q in enumerate(questions):
                judge_result = await _judge_one_quiz_question(q)
                question_verdicts.append({
                    "question": q.get("question", ""),
                    "verdict": judge_result["verdict"],
                    "reason": judge_result["reason"],
                })

    pass_count = sum(1 for v in question_verdicts if v["verdict"] == "pass")
    total_q = len(question_verdicts)
    logger.info(
        "  [quiz %d/%d] generated=%d pass=%d/%d (%.0fms)",
        i + 1, total, len(questions), pass_count, total_q, latency_ms,
    )
    return {
        "topic": topic,
        "num_generated": len(questions),
        "num_judged": total_q,
        "pass_count": pass_count,
        "valid_rate": pass_count / total_q if total_q else 0,
        "question_verdicts": question_verdicts,
        "latency_ms": latency_ms,
        "error": error,
    }


async def eval_quiz_quality(
    quiz_dataset: list[dict],
    concurrency: int = 5,
) -> dict:
    """Evaluate quiz question quality in parallel."""
    total = len(quiz_dataset)
    sem = asyncio.Semaphore(concurrency)
    logger.info("  Quiz eval: %d scenarios, concurrency=%d", total, concurrency)

    results = list(await asyncio.gather(*[
        _run_one_quiz(sem, i, total, item)
        for i, item in enumerate(quiz_dataset)
    ]))

    total_questions = sum(r["num_judged"] for r in results)
    total_pass = sum(r["pass_count"] for r in results)
    error_count = sum(1 for r in results if r["error"])

    return {
        "quiz_valid_rate": total_pass / total_questions if total_questions else 0,
        "total_questions": total_questions,
        "total_pass": total_pass,
        "total_scenarios": total,
        "error_count": error_count,
        "details": results,
    }


# ── 4. Gap Detection Precision ────────────────────────────────


GAP_EXTRACT_PROMPT = """\
从以下知识盲区诊断报告中，提取所有被识别为"盲区"或"薄弱"或"未接触"的主题名称。

诊断报告:
{report}

请按以下 JSON 格式输出：
{{"identified_gaps": ["主题1", "主题2", ...]}}

只输出 JSON，不要其他文字。如果报告中没有识别到任何盲区，输出空数组。"""


async def _setup_mastery(
    session: Any,
    user_id: str,
    preset: Any,
    all_topics: list[str],
) -> tuple[list[str], list[str]]:
    """Insert mastery records and return (expected_gaps, mastered_topics).

    Preset modes:
      - [] (empty list): no mastery records → all topics are gaps
      - "__all_mastered__": all topics get score 0.8
      - "__all_weak__": all topics get score 0.15
      - "__half_weak__": first half score 0.8, second half score 0.15
    """
    from sqlalchemy import delete
    await session.execute(
        delete(UserMastery).where(UserMastery.user_id == user_id)
    )
    await session.flush()

    now = datetime.now(timezone.utc)

    if preset == "__all_mastered__":
        for topic in all_topics:
            session.add(UserMastery(
                user_id=user_id, topic=topic, mastery_score=0.8,
                total_attempts=5, correct_count=4, last_reviewed=now,
            ))
        await session.flush()
        return [], all_topics

    if preset == "__all_weak__":
        for topic in all_topics:
            session.add(UserMastery(
                user_id=user_id, topic=topic, mastery_score=0.15,
                total_attempts=3, correct_count=0, last_reviewed=now,
            ))
        await session.flush()
        return all_topics, []

    if preset == "__half_weak__":
        mid = len(all_topics) // 2
        mastered = all_topics[:mid]
        weak = all_topics[mid:]
        for topic in mastered:
            session.add(UserMastery(
                user_id=user_id, topic=topic, mastery_score=0.8,
                total_attempts=5, correct_count=4, last_reviewed=now,
            ))
        for topic in weak:
            session.add(UserMastery(
                user_id=user_id, topic=topic, mastery_score=0.15,
                total_attempts=3, correct_count=0, last_reviewed=now,
            ))
        await session.flush()
        return weak, mastered

    # Default: empty preset → no records, all topics are gaps
    return all_topics, []


async def _extract_gaps_from_report(report: str) -> list[str]:
    """Use LLM to extract identified gap topics from an unstructured report."""
    prompt = GAP_EXTRACT_PROMPT.format(report=report)
    try:
        resp = await judge_client.judge(messages=[{"role": "user", "content": prompt}])
        resp = resp.strip()
        if resp.startswith("```"):
            lines = [l for l in resp.split("\n") if not l.strip().startswith("```")]
            resp = "\n".join(lines).strip()
        start, end = resp.find("{"), resp.rfind("}")
        result = json.loads(resp[start:end + 1]) if start != -1 else {}
        return result.get("identified_gaps", [])
    except Exception as e:
        logger.warning("Gap extraction failed: %s", e)
        return []


def _normalize_topic(text: str) -> str:
    """Lowercase, strip, and remove common noise for topic comparison."""
    import re
    t = text.lower().strip()
    # Remove surrounding quotes / brackets
    t = re.sub(r'^[\s"\'【】\[\]]+|[\s"\'【】\[\]]+$', "", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t)
    return t


def _fuzzy_topic_match(identified: str, expected: str) -> bool:
    """Check if an identified gap topic fuzzy-matches an expected topic.

    Guards against empty strings (which would trivially match everything)
    and normalizes both sides before doing bidirectional substring check.
    """
    id_norm = _normalize_topic(identified)
    ex_norm = _normalize_topic(expected)
    # Reject empty or trivially short strings (< 2 chars) to avoid false matches
    if len(id_norm) < 2 or len(ex_norm) < 2:
        return False
    return ex_norm in id_norm or id_norm in ex_norm


def _compute_gap_coverage(
    identified_gaps: list[str],
    expected_gaps: list[str],
    mastered_topics: list[str],
) -> dict:
    """Compute coverage rate and false positive rate."""
    covered = 0
    for eg in expected_gaps:
        if any(_fuzzy_topic_match(ig, eg) for ig in identified_gaps):
            covered += 1
    coverage_rate = covered / len(expected_gaps) if expected_gaps else 1.0

    false_positives = 0
    for ig in identified_gaps:
        if any(_fuzzy_topic_match(ig, mt) for mt in mastered_topics):
            false_positives += 1
    fp_rate = false_positives / len(identified_gaps) if identified_gaps else 0.0

    return {
        "coverage_rate": coverage_rate,
        "covered": covered,
        "expected_total": len(expected_gaps),
        "false_positive_rate": fp_rate,
        "false_positives": false_positives,
        "identified_total": len(identified_gaps),
    }


async def _run_one_gap(
    sem: asyncio.Semaphore,
    i: int,
    total: int,
    item: dict,
    all_topics: list[str],
) -> dict:
    """Run gap agent for one scenario."""
    scenario_id = item["scenario_id"]
    user_id = f"gap_eval_{scenario_id}_user"
    scope = item.get("scope")

    async with sem:
        logger.info("  [gap %d/%d] %s scope=%s", i + 1, total, scenario_id, scope)
        t0 = time.perf_counter()
        error = None
        report = ""
        try:
            async with async_session_factory() as session:
                # Filter topics by scope if needed
                if scope:
                    scoped_concepts = await _search_concepts(session, scope=scope)
                    scoped_topics = [c.topic for c in scoped_concepts]
                else:
                    scoped_topics = all_topics

                expected_gaps, mastered_topics = await _setup_mastery(
                    session, user_id, item.get("mastery_preset", []), scoped_topics,
                )
                await session.commit()

                result = await asyncio.wait_for(
                    run_gap_agent(
                        session=session,
                        user_id=user_id,
                        scope=scope,
                    ),
                    timeout=ORCHESTRATOR_TIMEOUT_S,
                )
                report = result.get("report", "")
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            expected_gaps, mastered_topics = [], []
            logger.error("  [gap %d/%d] error: %s", i + 1, total, error)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract and compare
        identified_gaps: list[str] = []
        gap_metrics: dict = {}
        if not error and report:
            identified_gaps = await _extract_gaps_from_report(report)
            gap_metrics = _compute_gap_coverage(identified_gaps, expected_gaps, mastered_topics)

    logger.info(
        "  [gap %d/%d] %s coverage=%.0f%% fp=%.0f%% (%.0fms)",
        i + 1, total, scenario_id,
        gap_metrics.get("coverage_rate", 0) * 100,
        gap_metrics.get("false_positive_rate", 0) * 100,
        latency_ms,
    )
    return {
        "scenario_id": scenario_id,
        "description": item.get("description", ""),
        "scope": scope,
        "expected_gap_mode": item.get("expected_gap_mode", ""),
        "expected_gaps_count": len(expected_gaps),
        "identified_gaps": identified_gaps,
        "gap_metrics": gap_metrics,
        "latency_ms": latency_ms,
        "error": error,
    }


async def eval_gap_precision(
    gap_dataset: list[dict],
    all_topics: list[str],
    concurrency: int = 5,
) -> dict:
    """Evaluate gap detection precision in parallel."""
    total = len(gap_dataset)
    sem = asyncio.Semaphore(concurrency)
    logger.info("  Gap eval: %d scenarios, concurrency=%d, KB topics=%d", total, concurrency, len(all_topics))

    results = list(await asyncio.gather(*[
        _run_one_gap(sem, i, total, item, all_topics)
        for i, item in enumerate(gap_dataset)
    ]))

    valid_results = [r for r in results if not r["error"] and r["gap_metrics"]]
    coverages = [r["gap_metrics"]["coverage_rate"] for r in valid_results]
    fp_rates = [r["gap_metrics"]["false_positive_rate"] for r in valid_results]
    error_count = sum(1 for r in results if r["error"])

    return {
        "avg_coverage_rate": sum(coverages) / len(coverages) if coverages else 0,
        "avg_false_positive_rate": sum(fp_rates) / len(fp_rates) if fp_rates else 0,
        "total_scenarios": total,
        "evaluated": len(valid_results),
        "error_count": error_count,
        "details": results,
    }


# ── 5. Main ───────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent evaluation with a preset profile.")
    parser.add_argument(
        "--profile",
        choices=("smoke", "standard", "full"),
        default=os.environ.get("EVAL_PROFILE", "standard"),
        help="Evaluation profile preset.",
    )
    return parser.parse_args()


def _save_report(report: dict, output_path: Path) -> None:
    """Atomically write partial/final report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    tmp.rename(output_path)


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

    # ── Health check: warm up models + verify infrastructure ──
    logger.info("=== Pre-flight health check ===")
    issues = await warmup_and_health_check(check_db=True, _logger=logger)
    if issues:
        logger.error("Health check FAILED with %d issue(s):", len(issues))
        for issue in issues:
            logger.error("  - %s", issue)

    # Prepare output path and report skeleton
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AGENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _agent_report_path(run_ts)

    report: dict[str, Any] = {
        "meta": {
            "run_log_file": str(run_log_file.resolve()),
            "timezone": "America/Los_Angeles",
            "started_at": started_at,
            "finished_at": None,
            "profile": profile.name,
            "output_file": str(output_path.resolve()),
            "status": "invalid" if issues else "running",
            "health_issues": issues,
            "completed_phases": [],
        },
    }

    if issues:
        report["meta"]["finished_at"] = datetime.now(EVAL_TZ).isoformat()
        _save_report(report, output_path)
        logger.error("Run marked invalid — aborting. Report: %s", output_path)
        return

    # Load datasets
    intent_path = DATASET_DIR / "intent_dataset.json"
    qa_path = DATASET_DIR / "qa_dataset.json"
    quiz_path = DATASET_DIR / "quiz_eval_dataset.json"
    gap_path = DATASET_DIR / "gap_eval_dataset.json"

    with open(intent_path, "r", encoding="utf-8") as f:
        intent_dataset = json.load(f)
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)
    with open(quiz_path, "r", encoding="utf-8") as f:
        quiz_dataset = json.load(f)
    with open(gap_path, "r", encoding="utf-8") as f:
        gap_dataset = json.load(f)

    logger.info(
        "Loaded %d intent, %d QA, %d quiz, %d gap items.",
        len(intent_dataset), len(qa_dataset), len(quiz_dataset), len(gap_dataset),
    )

    intent_subset, intent_subset_meta = build_intent_eval_subset(
        intent_dataset,
        profile.agent_intent_limit,
    )
    qa_subset, qa_subset_meta = build_stratified_subset(
        qa_dataset,
        profile.agent_qa_single_hop_limit,
        profile.agent_qa_multi_hop_limit,
    )
    quiz_subset = limit_items(quiz_dataset, profile.agent_quiz_limit)
    gap_subset = limit_items(gap_dataset, profile.agent_gap_limit)

    # Get all KB topics for gap eval
    all_topics: list[str] = []
    try:
        async with async_session_factory() as session:
            concepts = await _search_concepts(session, scope=None, keyword=None)
            all_topics = list({c.topic for c in concepts})
        logger.info("KB topics for gap eval: %d", len(all_topics))
    except Exception as e:
        logger.warning("Failed to load KB topics: %s", e)

    logger.info("=== Sequential eval phases (concurrency=%d per phase) ===", profile.concurrency)
    logger.info("  Intent:  %d items", len(intent_subset))
    logger.info("  QA E2E:  %d items (single=%d, multi=%d)",
                qa_subset_meta["total_selected"],
                qa_subset_meta["single_hop_selected"],
                qa_subset_meta["multi_hop_selected"])
    logger.info("  Quiz:    %d scenarios", len(quiz_subset))
    logger.info("  Gap:     %d scenarios", len(gap_subset))

    # ── Phase 1/4: Intent Classification (API-only, lightweight) ──
    logger.info("=== Phase 1/4: Intent Classification ===")
    intent_results: dict[str, Any] = {}
    try:
        intent_results = await eval_intent_classification(intent_subset, concurrency=profile.concurrency)
        logger.info("Intent Accuracy: %.1f%% (%d/%d)",
                     intent_results["accuracy"] * 100,
                     intent_results["correct"],
                     intent_results["total"])
        report["intent_classification"] = {
            "metric_scope": "single-intent plan_node accuracy",
            "subset": intent_subset_meta,
            "accuracy": intent_results["accuracy"],
            "correct": intent_results["correct"],
            "total": intent_results["total"],
            "error_count": intent_results["error_count"],
            "details": intent_results["details"],
        }
        report["meta"]["completed_phases"].append("intent")
    except Exception as e:
        logger.exception("Intent phase failed: %s", e)
        report["intent_classification"] = {"error": f"{type(e).__name__}: {e}"}
    _save_report(report, output_path)

    # ── Phase 2/4: QA E2E (GPU + API, heaviest) ──
    logger.info("=== Phase 2/4: QA E2E Task Success ===")
    task_results: dict[str, Any] = {}
    try:
        task_results = await eval_task_success(qa_subset, concurrency=profile.concurrency)
        logger.info("QA E2E Pass Rate (all): %.1f%%", task_results["pass_rate_all"] * 100)
        logger.info("QA E2E Pass Rate (judge-ok): %.1f%%", task_results["pass_rate_evaluated"] * 100)
        report["qa_e2e_task_success"] = {
            "metric_scope": "isolated QA subset E2E + LLM-as-Judge + replan diagnostics",
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
            "latency_p50_ms_all": task_results["latency_p50_ms_all"],
            "latency_p95_ms_all": task_results["latency_p95_ms_all"],
            "replan_avg": task_results["replan_avg"],
            "replan_rate": task_results["replan_rate"],
            "replan_action_distribution": task_results["replan_action_distribution"],
            "total": task_results["total"],
            "details": task_results["details"],
        }
        report["meta"]["completed_phases"].append("qa_e2e")
    except Exception as e:
        logger.exception("QA E2E phase failed: %s", e)
        report["qa_e2e_task_success"] = {"error": f"{type(e).__name__}: {e}"}
    _save_report(report, output_path)

    # ── Phase 3/4: Quiz Quality (GPU + API + SQLite) ──
    logger.info("=== Phase 3/4: Quiz Question Quality ===")
    quiz_results: dict[str, Any] = {}
    try:
        quiz_results = await eval_quiz_quality(quiz_subset, concurrency=profile.concurrency)
        logger.info("Quiz Valid Rate: %.1f%%", quiz_results["quiz_valid_rate"] * 100)
        report["quiz_quality"] = {
            "metric_scope": "quiz agent question validity (structure + LLM-as-Judge)",
            "quiz_valid_rate": quiz_results["quiz_valid_rate"],
            "total_questions": quiz_results["total_questions"],
            "total_pass": quiz_results["total_pass"],
            "total_scenarios": quiz_results["total_scenarios"],
            "error_count": quiz_results["error_count"],
            "details": quiz_results["details"],
        }
        report["meta"]["completed_phases"].append("quiz")
    except Exception as e:
        logger.exception("Quiz phase failed: %s", e)
        report["quiz_quality"] = {"error": f"{type(e).__name__}: {e}"}
    _save_report(report, output_path)

    # ── Phase 4/4: Gap Detection (SQLite + API, no GPU) ──
    logger.info("=== Phase 4/4: Gap Detection Precision ===")
    gap_results: dict[str, Any] = {}
    try:
        gap_results = await eval_gap_precision(gap_subset, all_topics, concurrency=profile.concurrency)
        logger.info("Gap Coverage: %.1f%%", gap_results["avg_coverage_rate"] * 100)
        report["gap_precision"] = {
            "metric_scope": "gap detection coverage + false positive rate",
            "avg_coverage_rate": gap_results["avg_coverage_rate"],
            "avg_false_positive_rate": gap_results["avg_false_positive_rate"],
            "total_scenarios": gap_results["total_scenarios"],
            "evaluated": gap_results["evaluated"],
            "error_count": gap_results["error_count"],
            "details": gap_results["details"],
        }
        report["meta"]["completed_phases"].append("gap")
    except Exception as e:
        logger.exception("Gap phase failed: %s", e)
        report["gap_precision"] = {"error": f"{type(e).__name__}: {e}"}

    # ── Final save ──
    report["meta"]["status"] = "completed"
    report["meta"]["finished_at"] = datetime.now(EVAL_TZ).isoformat()
    _save_report(report, output_path)
    logger.info("Agent report saved to %s", output_path)

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("AGENT EVALUATION SUMMARY  (phases: %s)", report["meta"]["completed_phases"])
    logger.info("=" * 60)
    if intent_results:
        logger.info("  Intent Accuracy:    %.1f%%  (target >= 95%%)", intent_results.get("accuracy", 0) * 100)
    if task_results:
        logger.info("  QA Pass Rate(all):  %.1f%%  (target >= 80%%)", task_results.get("pass_rate_all", 0) * 100)
        logger.info("  QA Pass Rate(ok):   %.1f%%", task_results.get("pass_rate_evaluated", 0) * 100)
        logger.info("  E2E Latency P50:    %.0f ms  (success only)", task_results.get("latency_p50_ms", 0))
        logger.info("  E2E Latency P95:    %.0f ms  (success only)", task_results.get("latency_p95_ms", 0))
        logger.info("  Replan Avg:         %.2f  (rate=%.0f%%)", task_results.get("replan_avg", 0), task_results.get("replan_rate", 0) * 100)
        logger.info("  Replan Actions:     %s", task_results.get("replan_action_distribution", {}))
        logger.info("  Verdicts:           %s", task_results.get("verdicts", {}))
    if quiz_results:
        logger.info("  Quiz Valid Rate:    %.1f%%  (%d/%d questions)",
                     quiz_results.get("quiz_valid_rate", 0) * 100,
                     quiz_results.get("total_pass", 0), quiz_results.get("total_questions", 0))
    if gap_results:
        logger.info("  Gap Coverage:       %.1f%%", gap_results.get("avg_coverage_rate", 0) * 100)
        logger.info("  Gap False Positive: %.1f%%", gap_results.get("avg_false_positive_rate", 0) * 100)
    logger.info("=" * 60)
    logger.info("=== eval_agent run end; full log: %s ===", run_log_file.resolve())


if __name__ == "__main__":
    args = parse_args()
    _ts = datetime.now(EVAL_TZ).strftime("%Y%m%d_%H%M%S")
    _log_path = LOG_DIR / f"eval_agent_{_ts}.log"
    configure_run_logging(_log_path)
    try:
        asyncio.run(main(_log_path, args.profile))
    except Exception:
        logging.exception("eval_agent aborted with exception (see log file above)")
        raise
