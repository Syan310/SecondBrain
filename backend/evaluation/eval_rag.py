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
from evaluation.eval_utils import EVAL_TZ, configure_run_logging, safe_p50, safe_p95
from evaluation.profiles import build_stratified_subset, get_eval_profile
from evaluation.ragas_runner import (
    build_ragas_samples,
    compute_multi_hop_lift,
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
你可以综合多个片段的信息进行推理和关联分析，不必局限于单个片段中的显式表述。
如果文档片段与问题完全无关，请说明无法回答。

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

    # Load QA dataset
    qa_path = DATASET_DIR / "qa_dataset.json"
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)
    logger.info("Loaded %d QA items. EVAL_PROFILE=%s", len(qa_dataset), profile.name)

    # Separate single-hop and multi-hop
    qa_subset, qa_subset_meta = build_stratified_subset(
        qa_dataset,
        profile.rag_single_hop_limit,
        profile.rag_multi_hop_limit,
    )
    single_hop = [q for q in qa_subset if q.get("question_type") != "multi-hop"]
    multi_hop = [q for q in qa_subset if q.get("question_type") == "multi-hop"]
    logger.info(
        "Selected %d QA items (single-hop=%d, multi-hop=%d)",
        qa_subset_meta["total_selected"],
        qa_subset_meta["single_hop_selected"],
        qa_subset_meta["multi_hop_selected"],
    )

    # Intermediate results keyed by question text to avoid mutating dataset items
    rag_results_map: dict[str, dict] = {}

    # ── Phase 1: Classic IR metrics (single-hop only, have source_doc ground truth) ──
    logger.info("=== Phase 1: Classic IR Metrics ===")
    recall_scores = []
    mrr_scores = []
    mrr_raw_scores = []
    mrr_reranked_scores = []
    latencies = []

    for i, item in enumerate(single_hop):
        question = item["question"]
        source = item.get("source_doc", "")
        logger.info("  [%d/%d] %s", i + 1, len(single_hop), question[:60])

        # Run RAG pipeline
        result = await run_rag_pipeline(question)
        retrieved_sources = [c.source_doc for c in result["contexts"]]
        latencies.append(result["total_ms"])

        # Recall@K and MRR
        recall = compute_recall_at_k(retrieved_sources, source)
        mrr = compute_reciprocal_rank(retrieved_sources, source)
        recall_scores.append(recall)
        mrr_scores.append(mrr)

        # Reranker lift
        lift_data = compute_reranker_lift(question)
        mrr_raw = compute_reciprocal_rank(lift_data["raw_sources"], source)
        mrr_reranked = compute_reciprocal_rank(lift_data["reranked_sources"], source)
        mrr_raw_scores.append(mrr_raw)
        mrr_reranked_scores.append(mrr_reranked)

        rag_results_map[question] = result

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

    # ── Phase 2: Run RAG for multi-hop ──
    logger.info("=== Phase 2: Multi-hop RAG ===")
    for i, item in enumerate(multi_hop):
        question = item["question"]
        logger.info("  [%d/%d] %s", i + 1, len(multi_hop), question[:60])
        result = await run_rag_pipeline(question)
        rag_results_map[question] = result

    # ── Phase 3: RAGAS evaluation (all items) ──
    logger.info("=== Phase 3: RAGAS Evaluation ===")
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

    # Compute means
    ragas_means = compute_ragas_means(ragas_results)

    logger.info("RAGAS Means: %s", json.dumps(ragas_means, indent=2))
    logger.info("RAGAS Metric Status: %s", json.dumps(ragas_metric_status, ensure_ascii=False, indent=2))

    # ── Phase 4: Multi-hop Lift ──
    logger.info("=== Phase 4: Multi-hop Lift ===")
    multi_hop_lift = compute_multi_hop_lift(
        qa_subset,
        ragas_results.get("faithfulness", []),
    )
    logger.info("Multi-hop Lift: %s", json.dumps(multi_hop_lift, indent=2))

    # ── Save results ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _rag_report_path(run_ts)
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
        "subset": qa_subset_meta,
        "ir_metrics": ir_metrics,
        "ragas_means": ragas_means,
        "ragas_metric_status": ragas_metric_status,
        "multi_hop_lift": multi_hop_lift,
        "sample_count": len(ragas_samples),
        "ragas_per_sample": ragas_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("RAG report saved to %s", output_path)

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("RAG EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info("  Recall@K:          %.3f  (target ≥ 0.85)", ir_metrics["recall_at_k"])
    logger.info("  MRR:               %.3f  (target ≥ 0.70)", ir_metrics["mrr"])
    logger.info("  Reranker Lift:     %.3f  (target > 0)", ir_metrics["reranker_lift"])
    logger.info("  Latency P95:       %.0f ms  (target < 500ms)", ir_metrics["latency_p95_ms"])
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
    if "lift" in multi_hop_lift:
        logger.info("  Multi-hop Lift:    %.3f  (target > 0)", multi_hop_lift["lift"])
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
