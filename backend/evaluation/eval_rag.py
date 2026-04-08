"""RAG Evaluation — RAGAS framework + classic IR metrics.

Evaluates:
  1. RAGAS metrics: Context Precision, Context Recall, Faithfulness, Answer Relevancy
  2. Classic IR: Recall@K, MRR, Reranker Lift
  3. Multi-hop Lift: single-hop vs multi-hop faithfulness gap

Usage:
    cd backend
    MILVUS_HOST=localhost python -m evaluation.eval_rag
    MILVUS_HOST=localhost python -m evaluation.eval_rag --profile smoke

Requires Milvus, embedding/reranker models, and LLM APIs to be available.
Report filename and meta timestamps use America/Los_Angeles local wall clock.
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
from typing import Any

from app.config import settings
from app.services.embedding import embedding_service
from app.services.milvus_client import milvus_service
from app.services.reranker import reranker_service
from app.services.llm_client import llm_client
from app.agent.tools.rag_search import rag_search
from evaluation.eval_utils import EVAL_TZ, configure_run_logging, safe_p50, safe_p95, warmup_and_health_check
from evaluation.profiles import build_balanced_subset, get_eval_profile
from evaluation.ragas_runner import (
    build_ragas_samples,
    compute_ragas_means,
    run_ragas_evaluation,
)

logger = logging.getLogger(__name__)

DATASET_DIR = Path(__file__).parent / "datasets"
OUTPUT_DIR = Path(__file__).parent / "results"
RAG_OUTPUT_DIR = OUTPUT_DIR / "rag"
RAG_LOG_DIR = OUTPUT_DIR / "logs_rag"
# ── 1. Run RAG pipeline per question ──────────────────────────


QA_SYSTEM_PROMPT = """\
你是一个知识问答助手。根据以下检索到的文档片段回答用户问题。

## 回答要求

- 你的回答必须完全基于以下检索片段中的信息
- 允许对多个片段的内容进行归纳、对比和因果推理
- 禁止引入检索内容中未提及的概念、数据、案例或结论
- 如果检索片段不足以回答问题，明确指出信息不足，不要用自身知识补充
- 篇幅控制在 200 字以内，只保留核心结论和关键依据

检索到的文档片段：
{contexts}
"""


async def run_rag_pipeline(question: str, top_k: int = 5) -> dict:
    """Run full RAG pipeline: retrieve + generate answer.

    Returns: {answer, contexts, context_texts, latency_ms}
    """
    t0 = time.perf_counter()

    # Retrieve
    ranked_chunks = rag_search(query=question, top_k=top_k)
    t_retrieve = time.perf_counter()

    context_texts = [c.text for c in ranked_chunks]
    contexts_str = "\n\n".join(
        f"[{i+1}] (来源: {c.source_doc} | {c.heading_path})\n{c.text}"
        for i, c in enumerate(ranked_chunks)
    )

    # Generate answer
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT.format(contexts=contexts_str)},
        {"role": "user", "content": question},
    ]
    resp = await llm_client.chat(messages=messages, temperature=0.1, max_tokens=1024)
    answer = resp.get("content", "")
    t_generate = time.perf_counter()

    return {
        "answer": answer,
        "contexts": ranked_chunks,
        "context_texts": context_texts,
        "retrieve_ms": (t_retrieve - t0) * 1000,
        "generate_ms": (t_generate - t_retrieve) * 1000,
        "total_ms": (t_generate - t0) * 1000,
    }


# ── 2. Classic IR metrics ─────────────────────────────────────


def compute_recall_at_k(
    retrieved_sources: list[str], relevant_source: str
) -> float:
    """1.0 if relevant_source is in retrieved sources, else 0.0."""
    return 1.0 if relevant_source in retrieved_sources else 0.0


def compute_reciprocal_rank(
    retrieved_sources: list[str], relevant_source: str
) -> float:
    """1/rank if found, else 0.0."""
    for i, src in enumerate(retrieved_sources):
        if src == relevant_source:
            return 1.0 / (i + 1)
    return 0.0


def compute_reranker_lift(question: str, top_k: int = 5) -> dict:
    """Compare MRR with/without reranker for one question.

    Returns raw search sources and reranked sources for external MRR comparison.
    """
    # Encode query
    query_out = embedding_service.encode_query(question)
    dense = query_out.dense.tolist()
    sparse = query_out.sparse[0]

    # Raw hybrid search (no rerank)
    raw_candidates = milvus_service.hybrid_search(
        query_vector=dense, sparse_vector=sparse, top_k=top_k
    )
    raw_sources = [c.source_doc for c in raw_candidates]

    # Reranked
    candidate_k = top_k * 3
    candidates = milvus_service.hybrid_search(
        query_vector=dense, sparse_vector=sparse, top_k=candidate_k
    )
    if candidates:
        passages = [c.text for c in candidates]
        reranked = reranker_service.rerank(query=question, passages=passages, top_n=top_k)
        reranked_sources = [candidates[r.original_index].source_doc for r in reranked]
    else:
        reranked_sources = []

    return {"raw_sources": raw_sources, "reranked_sources": reranked_sources}


# ── 3. RAGAS evaluation ───────────────────────────────────────


def _build_bge_ragas_embeddings():
    """Build a RAGAS-compatible embedding wrapper around our BGE-m3 service.

    All RAGAS imports are deferred here so the module stays importable without
    ragas installed.
    """
    from ragas.embeddings import BaseRagasEmbeddings

    class BGEm3RagasEmbeddings(BaseRagasEmbeddings):
        """Wrapper around our BGE-m3 embedding service for RAGAS."""

        def __init__(self):
            self._svc = embedding_service

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            output = self._svc.encode(texts, return_sparse=False)
            return output.dense.tolist()

        def embed_query(self, text: str) -> list[float]:
            output = self._svc.encode_query(text)
            return output.dense.tolist()

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embed_documents(texts)

        async def aembed_query(self, text: str) -> list[float]:
            return self.embed_query(text)

        def set_run_config(self, *args, **kwargs):
            pass

    return BGEm3RagasEmbeddings()


def _rag_report_path(run_ts: str) -> Path:
    """Single JSON report path for one eval run (under results/rag/)."""
    return RAG_OUTPUT_DIR / f"rag_eval_{run_ts}.json"


# ── 4. Main ───────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG evaluation with a preset profile.")
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


async def main(run_log_file: Path, profile_name: str):
    t_start = datetime.now(EVAL_TZ)
    started_at = t_start.isoformat()
    run_ts = t_start.strftime("%Y%m%d_%H%M%S")
    profile = get_eval_profile(profile_name)
    logger.info("=== eval_rag run start ===")
    logger.info("Log file: %s", run_log_file.resolve())
    logger.info("cwd=%s python=%s", os.getcwd(), sys.version.split()[0])
    logger.info("MILVUS_HOST=%s", os.environ.get("MILVUS_HOST", "(unset)"))
    logger.info("EVAL_PROFILE=%s", profile.name)

    # Init services
    logger.info("Connecting to Milvus...")
    milvus_service.connect()
    milvus_service.init_collection()

    # Prepare output path and report skeleton
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _rag_report_path(run_ts)

    # ── Health check: warm up models + verify infrastructure ──
    logger.info("=== Pre-flight health check ===")
    issues = await warmup_and_health_check(check_db=False, _logger=logger)

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

    # Load QA dataset (single-hop only; multi-hop is evaluated at Agent level)
    qa_path = DATASET_DIR / "qa_dataset.json"
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)
    single_hop_all = [q for q in qa_dataset if q.get("question_type") != "multi-hop"]
    logger.info("Loaded %d QA items (%d single-hop). EVAL_PROFILE=%s",
                len(qa_dataset), len(single_hop_all), profile.name)

    qa_subset, qa_subset_meta = build_balanced_subset(
        single_hop_all,
        profile.rag_single_hop_limit,
        group_key=lambda item: item.get("source_doc", "__unknown__"),
    )
    logger.info(
        "Selected %d single-hop QA items (limit=%s)",
        qa_subset_meta["selected"],
        qa_subset_meta["limit"],
    )

    # Intermediate results keyed by question text to avoid mutating dataset items
    rag_results_map: dict[str, dict] = {}
    ir_metrics: dict[str, Any] = {}

    # ── Phase 1: Classic IR metrics (parallel) ──
    logger.info("=== Phase 1: Classic IR Metrics (concurrency=%d) ===", profile.concurrency)
    try:
        sem = asyncio.Semaphore(profile.concurrency)
        total_items = len(qa_subset)

        async def _run_one_ir(i: int, item: dict) -> dict:
            question = item["question"]
            source = item.get("source_doc", "")
            async with sem:
                logger.info("  [ir %d/%d] %s", i + 1, total_items, question[:60])
                result = await run_rag_pipeline(question)
                lift_data = await asyncio.to_thread(compute_reranker_lift, question)

            retrieved_sources = [c.source_doc for c in result["contexts"]]
            return {
                "question": question,
                "rag_result": result,
                "recall": compute_recall_at_k(retrieved_sources, source),
                "mrr": compute_reciprocal_rank(retrieved_sources, source),
                "mrr_raw": compute_reciprocal_rank(lift_data["raw_sources"], source),
                "mrr_reranked": compute_reciprocal_rank(lift_data["reranked_sources"], source),
                "latency_ms": result["total_ms"],
            }

        ir_items = await asyncio.gather(*[
            _run_one_ir(i, item) for i, item in enumerate(qa_subset)
        ])

        recall_scores = [r["recall"] for r in ir_items]
        mrr_scores = [r["mrr"] for r in ir_items]
        mrr_raw_scores = [r["mrr_raw"] for r in ir_items]
        mrr_reranked_scores = [r["mrr_reranked"] for r in ir_items]
        latencies = [r["latency_ms"] for r in ir_items]
        for r in ir_items:
            rag_results_map[r["question"]] = r["rag_result"]

        ir_metrics = {
            "recall_at_k": sum(recall_scores) / len(recall_scores) if recall_scores else 0,
            "mrr": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0,
            "mrr_raw": sum(mrr_raw_scores) / len(mrr_raw_scores) if mrr_raw_scores else 0,
            "mrr_reranked": sum(mrr_reranked_scores) / len(mrr_reranked_scores) if mrr_reranked_scores else 0,
            "reranker_lift": (
                (sum(mrr_reranked_scores) - sum(mrr_raw_scores)) / len(mrr_raw_scores)
                if mrr_raw_scores
                else 0
            ),
            "latency_p50_ms": safe_p50(latencies),
            "latency_p95_ms": safe_p95(latencies),
        }
        logger.info("IR Metrics: %s", json.dumps(ir_metrics, indent=2))
        report["subset"] = qa_subset_meta
        report["ir_metrics"] = ir_metrics
        report["meta"]["completed_phases"].append("ir")
    except Exception as e:
        logger.exception("Phase 1 (IR) failed: %s", e)
        report["ir_metrics"] = {"error": f"{type(e).__name__}: {e}"}
    _save_report(report, output_path)

    # ── Phase 2: RAGAS evaluation ──
    ragas_means: dict[str, Any] = {}
    ragas_metric_status: dict[str, Any] = {}
    ragas_results: list = []
    ragas_samples: list = []

    if not rag_results_map:
        logger.warning("Phase 1 produced no results — skipping Phase 2 (RAGAS).")
        report["ragas_means"] = {"skipped": "Phase 1 failed or returned no results"}
    else:
        logger.info("=== Phase 2: RAGAS Evaluation ===")
        try:
            ragas_samples = build_ragas_samples(qa_subset, rag_results_map)
            ragas_eval = await run_ragas_evaluation(
                ragas_samples,
                judge_model=settings.judge_model,
                judge_api_key=settings.judge_api_key,
                judge_base_url=settings.judge_base_url,
                embeddings=_build_bge_ragas_embeddings(),
            )
            ragas_results = ragas_eval["results"]
            ragas_metric_status = ragas_eval["metric_status"]
            ragas_means = compute_ragas_means(ragas_results)
            logger.info("RAGAS Means: %s", json.dumps(ragas_means, indent=2))
            logger.info("RAGAS Metric Status: %s", json.dumps(ragas_metric_status, ensure_ascii=False, indent=2))
            report["ragas_means"] = ragas_means
            report["ragas_metric_status"] = ragas_metric_status
            report["sample_count"] = len(ragas_samples)
            report["ragas_per_sample"] = ragas_results
            report["meta"]["completed_phases"].append("ragas")
        except Exception as e:
            logger.exception("Phase 2 (RAGAS) failed: %s", e)
            report["ragas_means"] = {"error": f"{type(e).__name__}: {e}"}

    # ── Final save ──
    report["meta"]["status"] = "completed"
    report["meta"]["finished_at"] = datetime.now(EVAL_TZ).isoformat()
    _save_report(report, output_path)
    logger.info("RAG report saved to %s", output_path)

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("RAG EVALUATION SUMMARY  (phases: %s)", report["meta"]["completed_phases"])
    logger.info("=" * 60)
    if ir_metrics and "error" not in ir_metrics:
        logger.info("  Recall@K:          %.3f  (target >= 0.85)", ir_metrics["recall_at_k"])
        logger.info("  MRR:               %.3f  (target >= 0.70)", ir_metrics["mrr"])
        logger.info("  Reranker Lift:     %.3f  (target > 0)", ir_metrics["reranker_lift"])
        logger.info("  Latency P95:       %.0f ms  (target < 500ms)", ir_metrics["latency_p95_ms"])
    if ragas_means and "error" not in ragas_means:
        for k, v in ragas_means.items():
            logger.info("  %-20s %.3f", k + ":", v)
    for metric_name, status in ragas_metric_status.items():
        if status.get("status") == "error":
            logger.warning(
                "  RAGAS %-12s failed: %s: %s",
                metric_name,
                status.get("error_type"),
                status.get("error"),
            )
    logger.info("=" * 60)
    logger.info("=== eval_rag run end; full log: %s ===", run_log_file.resolve())


if __name__ == "__main__":
    args = parse_args()
    _ts = datetime.now(EVAL_TZ).strftime("%Y%m%d_%H%M%S")
    _log_path = RAG_LOG_DIR / f"eval_rag_{_ts}.log"
    configure_run_logging(_log_path)
    try:
        asyncio.run(main(_log_path, args.profile))
    except Exception:
        logging.exception("eval_rag aborted with exception (see log file above)")
        raise
