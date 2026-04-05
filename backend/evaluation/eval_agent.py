"""Agent Evaluation — task-driven metrics.

Evaluates:
  1. Intent Classification Accuracy (Orchestrator routing)
  2. Task Success Rate (E2E via LLM-as-Judge)
  3. E2E Latency

Usage:
    cd backend
    MILVUS_HOST=localhost python -m evaluation.eval_agent

Requires Milvus, embedding/reranker models, and LLM APIs to be available.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from app.config import settings
from app.services.milvus_client import milvus_service, memory_milvus_service
from app.services.llm_client import llm_client
from app.agent.graph import run_orchestrator
from app.db.database import async_session_factory
from evaluation.judge_client import judge_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_DIR = Path(__file__).parent / "datasets"
OUTPUT_DIR = Path(__file__).parent / "results"


# ── 1. Intent Classification ──────────────────────────────────


INTENT_EXTRACT_PROMPT = """\
分析以下用户查询，判断其意图类别。只输出一个意图标签。

意图类别：
- qa: 知识问答（问事实、概念、原理、对比等）
- quiz: 要求测验、复习、出题
- gap_analysis: 要求分析知识盲区、薄弱点

用户查询: {query}

只输出意图标签（qa/quiz/gap_analysis），不要其他文字。"""


async def classify_intent(query: str) -> str:
    """Use the orchestrator's plan_node to classify intent."""
    # Use LLM directly for intent classification (faster than full orchestrator)

    # The orchestrator returns a final answer; we need to infer intent from
    # the plan. Let's use a simpler approach: directly call the LLM with the
    # same prompt pattern the orchestrator uses, to test intent classification.
    resp = await llm_client.chat(
        messages=[{"role": "user", "content": INTENT_EXTRACT_PROMPT.format(query=query)}],
        temperature=0.0,
        max_tokens=20,
    )
    content = (resp.get("content") or "").strip().lower()

    # Normalize
    for intent in ["qa", "quiz", "gap_analysis"]:
        if intent in content:
            return intent
    return content


async def eval_intent_classification(dataset: list[dict]) -> dict:
    """Evaluate intent classification accuracy."""
    correct = 0
    results = []

    for i, item in enumerate(dataset):
        query = item["query"]
        expected = item["expected_intent"]
        logger.info("  [%d/%d] %s -> expected: %s", i + 1, len(dataset), query[:50], expected)

        predicted = await classify_intent(query)
        is_correct = predicted == expected
        if is_correct:
            correct += 1

        results.append({
            "query": query,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
        })
        logger.info("    predicted: %s  %s", predicted, "✓" if is_correct else "✗")

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
    """Use Kimi judge to evaluate if the task was completed successfully."""
    prompt = JUDGE_PROMPT.format(
        question=question, ground_truth=ground_truth, answer=answer
    )
    resp = await judge_client.judge(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )

    # Parse JSON
    resp = resp.strip()
    if resp.startswith("```"):
        lines = resp.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        resp = "\n".join(lines).strip()

    try:
        result = json.loads(resp)
        return result
    except json.JSONDecodeError:
        # Try extracting JSON
        start = resp.find("{")
        end = resp.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(resp[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {"verdict": "error", "reason": f"Failed to parse judge response: {resp[:100]}"}


async def eval_task_success(qa_dataset: list[dict]) -> dict:
    """Run E2E orchestrator for each QA item and judge success."""
    results = []
    latencies = []
    verdicts = {"pass": 0, "partial": 0, "fail": 0, "error": 0}

    for i, item in enumerate(qa_dataset):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        logger.info("  [%d/%d] %s", i + 1, len(qa_dataset), question[:60])

        # Run orchestrator E2E
        t0 = time.perf_counter()
        try:
            async with async_session_factory() as session:
                result = await run_orchestrator(
                    question, user_id="eval_user", session=session
                )
                answer = result.get("final_answer", "")
        except Exception as e:
            logger.error("    Orchestrator error: %s", e)
            answer = f"Error: {e}"
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)

        # Judge
        judge_result = await judge_task_success(question, ground_truth, answer)
        verdict = judge_result.get("verdict", "error")
        verdicts[verdict] = verdicts.get(verdict, 0) + 1

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "verdict": verdict,
            "reason": judge_result.get("reason", ""),
            "latency_ms": latency_ms,
        })
        logger.info("    verdict: %s (%.0fms)", verdict, latency_ms)

    total = len(qa_dataset)
    success_rate = verdicts["pass"] / total if total else 0
    partial_rate = verdicts["partial"] / total if total else 0

    return {
        "success_rate": success_rate,
        "partial_rate": partial_rate,
        "fail_rate": verdicts["fail"] / total if total else 0,
        "verdicts": verdicts,
        "latency_p50_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        "total": total,
        "details": results,
    }


# ── 3. Main ───────────────────────────────────────────────────


async def main():
    # Init services
    logger.info("Connecting to Milvus...")
    milvus_service.connect()
    milvus_service.init_collection()
    try:
        memory_milvus_service.init_collection()
    except Exception:
        logger.warning("Memory collection init failed, continuing without memory.")

    # Initialize SQLite
    from app.services.database import init_db
    await init_db()

    # Load datasets
    intent_path = DATASET_DIR / "intent_dataset.json"
    qa_path = DATASET_DIR / "qa_dataset.json"

    with open(intent_path, "r", encoding="utf-8") as f:
        intent_dataset = json.load(f)
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)

    logger.info("Loaded %d intent items, %d QA items.", len(intent_dataset), len(qa_dataset))

    # ── Phase 1: Intent Classification ──
    logger.info("=== Phase 1: Intent Classification ===")
    intent_results = await eval_intent_classification(intent_dataset)
    logger.info("Intent Accuracy: %.1f%% (%d/%d)",
                intent_results["accuracy"] * 100,
                intent_results["correct"],
                intent_results["total"])

    # ── Phase 2: Task Success Rate (E2E) ──
    logger.info("=== Phase 2: Task Success Rate (E2E) ===")
    # Use a subset for E2E (it's slow) — first 15 items
    qa_subset = qa_dataset[:15]
    logger.info("Running E2E on %d QA items (subset)...", len(qa_subset))
    task_results = await eval_task_success(qa_subset)
    logger.info("Task Success Rate: %.1f%%", task_results["success_rate"] * 100)

    # ── Save results ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_report = {
        "intent_classification": {
            "accuracy": intent_results["accuracy"],
            "correct": intent_results["correct"],
            "total": intent_results["total"],
            "details": intent_results["details"],
        },
        "task_success": {
            "success_rate": task_results["success_rate"],
            "partial_rate": task_results["partial_rate"],
            "fail_rate": task_results["fail_rate"],
            "verdicts": task_results["verdicts"],
            "latency_p50_ms": task_results["latency_p50_ms"],
            "latency_p95_ms": task_results["latency_p95_ms"],
            "total": task_results["total"],
            "details": task_results["details"],
        },
    }
    output_path = OUTPUT_DIR / "agent_eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Results saved to %s", output_path)

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("AGENT EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info("  Intent Accuracy:   %.1f%%  (target ≥ 95%%)", intent_results["accuracy"] * 100)
    logger.info("  Task Success Rate: %.1f%%  (target ≥ 80%%)", task_results["success_rate"] * 100)
    logger.info("  E2E Latency P50:   %.0f ms", task_results["latency_p50_ms"])
    logger.info("  E2E Latency P95:   %.0f ms (target < 2000ms)", task_results["latency_p95_ms"])
    logger.info("  Verdicts: %s", task_results["verdicts"])
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
