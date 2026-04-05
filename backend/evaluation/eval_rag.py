"""RAG Evaluation — RAGAS framework + classic IR metrics.

Evaluates:
  1. RAGAS metrics: Context Precision, Context Recall, Faithfulness, Answer Relevancy
  2. Classic IR: Recall@K, MRR, Reranker Lift
  3. Multi-hop Lift: decompose on/off comparison

Usage:
    cd backend
    MILVUS_HOST=localhost python -m evaluation.eval_rag

Requires Milvus, embedding/reranker models, and LLM APIs to be available.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from app.config import settings
from app.services.embedding import embedding_service
from app.services.milvus_client import milvus_service
from app.services.reranker import reranker_service
from app.services.llm_client import llm_client
from app.agent.tools.rag_search import rag_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_DIR = Path(__file__).parent / "datasets"
OUTPUT_DIR = Path(__file__).parent / "results"


# ── 1. Run RAG pipeline per question ──────────────────────────


QA_SYSTEM_PROMPT = """\
你是一个知识问答助手。根据以下检索到的文档片段回答用户问题。
如果文档中没有相关信息，请说明无法回答。

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


from ragas.embeddings.base import BaseRagasEmbeddings


class BGEm3RagasEmbeddings(BaseRagasEmbeddings):
    """Wrapper around our BGE-m3 embedding service for RAGAS."""

    def __init__(self):
        from app.services.embedding import embedding_service
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


async def run_ragas_evaluation(samples: list[dict]) -> dict:
    """Run RAGAS metrics using Kimi-K2.5 as judge.

    Each sample: {user_input, response, retrieved_contexts, reference}
    """
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        _Faithfulness,
        _AnswerRelevancy,
        _LLMContextPrecisionWithReference,
        _LLMContextRecall,
    )
    from ragas.llms import llm_factory
    from openai import OpenAI

    # Build judge LLM
    client = OpenAI(api_key=settings.judge_api_key, base_url=settings.judge_base_url)
    judge_llm = llm_factory(settings.judge_model, client=client)

    # Build embeddings wrapper (for AnswerRelevancy metric)
    bge_embeddings = BGEm3RagasEmbeddings()

    # Build RAGAS samples
    ragas_samples = []
    for s in samples:
        ragas_samples.append(
            SingleTurnSample(
                user_input=s["user_input"],
                response=s["response"],
                retrieved_contexts=s["retrieved_contexts"],
                reference=s["reference"],
            )
        )

    dataset = EvaluationDataset(samples=ragas_samples)

    # Metrics
    faithfulness = _Faithfulness(llm=judge_llm)
    answer_relevancy = _AnswerRelevancy(llm=judge_llm, embeddings=bge_embeddings)
    context_precision = _LLMContextPrecisionWithReference(llm=judge_llm)
    context_recall = _LLMContextRecall(llm=judge_llm)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    logger.info("Running RAGAS evaluation with %d samples, %d metrics...", len(ragas_samples), len(metrics))
    result = evaluate(dataset=dataset, metrics=metrics, embeddings=bge_embeddings)

    return result.to_pandas().to_dict(orient="list")


# ── 4. Main ───────────────────────────────────────────────────


async def main():
    # Init services
    logger.info("Connecting to Milvus...")
    milvus_service.connect()
    milvus_service.init_collection()

    # Load QA dataset
    qa_path = DATASET_DIR / "qa_dataset.json"
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)
    logger.info("Loaded %d QA items.", len(qa_dataset))

    # Separate single-hop and multi-hop
    single_hop = [q for q in qa_dataset if q.get("question_type") != "multi-hop"]
    multi_hop = [q for q in qa_dataset if q.get("question_type") == "multi-hop"]
    logger.info("Single-hop: %d, Multi-hop: %d", len(single_hop), len(multi_hop))

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

        # Store for RAGAS
        item["_rag_result"] = result

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
        "latency_p50_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
    }
    logger.info("IR Metrics: %s", json.dumps(ir_metrics, indent=2))

    # ── Phase 2: Run RAG for multi-hop ──
    logger.info("=== Phase 2: Multi-hop RAG ===")
    for i, item in enumerate(multi_hop):
        question = item["question"]
        logger.info("  [%d/%d] %s", i + 1, len(multi_hop), question[:60])
        result = await run_rag_pipeline(question)
        item["_rag_result"] = result

    # ── Phase 3: RAGAS evaluation (all items) ──
    logger.info("=== Phase 3: RAGAS Evaluation ===")
    ragas_samples = []
    for item in qa_dataset:
        rag_result = item.get("_rag_result")
        if not rag_result:
            continue
        ragas_samples.append({
            "user_input": item["question"],
            "response": rag_result["answer"],
            "retrieved_contexts": rag_result["context_texts"],
            "reference": item.get("ground_truth", ""),
        })

    ragas_results = await run_ragas_evaluation(ragas_samples)

    # Compute means
    ragas_means = {}
    for key, values in ragas_results.items():
        if key in ("user_input", "response", "retrieved_contexts", "reference"):
            continue
        numeric = [v for v in values if isinstance(v, (int, float)) and v == v]  # filter NaN
        if numeric:
            ragas_means[key] = sum(numeric) / len(numeric)

    logger.info("RAGAS Means: %s", json.dumps(ragas_means, indent=2))

    # ── Phase 4: Multi-hop Lift ──
    logger.info("=== Phase 4: Multi-hop Lift ===")
    single_faithfulness = []
    multi_faithfulness = []
    for key, values in ragas_results.items():
        if "faithfulness" in key.lower():
            for i, v in enumerate(values):
                if isinstance(v, (int, float)) and v == v:
                    if i < len(single_hop):
                        single_faithfulness.append(v)
                    else:
                        multi_faithfulness.append(v)

    multi_hop_lift = {
        "single_hop_faithfulness": sum(single_faithfulness) / len(single_faithfulness) if single_faithfulness else 0,
        "multi_hop_faithfulness": sum(multi_faithfulness) / len(multi_faithfulness) if multi_faithfulness else 0,
    }
    if single_faithfulness and multi_faithfulness:
        multi_hop_lift["lift"] = multi_hop_lift["multi_hop_faithfulness"] - multi_hop_lift["single_hop_faithfulness"]
    logger.info("Multi-hop Lift: %s", json.dumps(multi_hop_lift, indent=2))

    # ── Save results ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_report = {
        "ir_metrics": ir_metrics,
        "ragas_means": ragas_means,
        "multi_hop_lift": multi_hop_lift,
        "sample_count": len(ragas_samples),
        "ragas_per_sample": ragas_results,
    }
    output_path = OUTPUT_DIR / "rag_eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Results saved to %s", output_path)

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
    if "lift" in multi_hop_lift:
        logger.info("  Multi-hop Lift:    %.3f  (target > 0)", multi_hop_lift["lift"])
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
