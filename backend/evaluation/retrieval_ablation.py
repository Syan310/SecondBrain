"""Retrieval ablation: dense only vs BM25 only vs hybrid (RRF). No LLM calls.

For every answerable question, each mode takes CANDIDATE_K results from Milvus and
keeps the first TOP_K, the same way ``rag_search`` does when ENABLE_RERANKER=false:
  dense   milvus_service.search()         BGE-M3 vector only
  bm25    milvus_service.bm25_search()    BM25 only
  hybrid  milvus_service.hybrid_search()  dense + BM25, RRF k=60 (= eval_rag "rr")

Metrics:
  Document level (same functions as eval_rag.py): Recall@5, MRR.
  Section level (gold_sections): a chunk belongs to a gold section when its
  source_doc matches and its heading_path equals the gold heading_path or is a
  sub-heading of it (gold path + "/").
    Section Recall@5  share of the question's gold sections that at least one
                      top-5 chunk belongs to
    Section MRR       1 / rank of the first top-5 chunk in any gold section
    Section Hit@1     1.0 if the first chunk is in a gold section

Paired bootstrap (stats.paired_compare, 1000 resamples, seed 20261102):
hybrid vs dense and hybrid vs bm25. The diff is hybrid minus the other mode.

Self-check: hybrid document-level Recall@5 / MRR must equal the L1 task runs R0
(secondbrain_docs) and R1 (secondbrain_l1). If not, the script exits 1.

Usage:
    cd backend
    MILVUS_HOST=localhost MILVUS_COLLECTION_NAME=secondbrain_docs ENABLE_RERANKER=false \
      RERANKER_USE_HEADING=false CITATION_MODE=off \
      .venv/Scripts/python -m evaluation.retrieval_ablation --dataset qa_l0_v1.json \
      --reference evaluation/results/rag/rag_eval_20261102_215402.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import settings
from evaluation.eval_rag import (
    CANDIDATE_K,
    COLLECTION_SCOPES,
    EMBEDDING_MODEL_NAME,
    OUTPUT_DIR,
    RRF_K,
    TOP_K,
    _collection_meta,
    _git,
    _version,
    compute_recall_at_k,
    compute_reciprocal_rank,
    load_eval_dataset,
    require_explicit_env,
)
from evaluation.eval_utils import EVAL_TZ
from evaluation.stats import BOOTSTRAP_SEED, N_BOOTSTRAP, paired_compare

logger = logging.getLogger(__name__)

RETRIEVAL_OUTPUT_DIR = OUTPUT_DIR / "retrieval"
MODES = ("dense", "bm25", "hybrid")
COMPARISONS = (("dense", "hybrid"), ("bm25", "hybrid"))  # (a, b): diff = b - a
METRICS = ("recall_at_k", "rr", "section_recall_at_k", "section_rr", "section_hit_at_1")
HNSW_EF = 128  # MilvusService default, also what rag_search uses

# Hybrid results of the L1 task runs R0 / R1 (docs/status-l1-scale.md), rounded to 3 places.
EXPECTED_HYBRID = {
    "secondbrain_docs": {"recall_at_k": 0.960, "mrr": 0.950},
    "secondbrain_l1": {"recall_at_k": 0.880, "mrr": 0.816},
}


# ── Metrics ───────────────────────────────────────────────────


def in_section(source_doc: str, heading_path: str, gold: dict[str, str]) -> bool:
    """True if the chunk is in the gold section or one of its sub-headings."""
    if source_doc != gold["source_doc"]:
        return False
    path = heading_path or ""
    return path == gold["heading_path"] or path.startswith(gold["heading_path"] + "/")


def section_metrics(
    retrieved: list[dict[str, Any]], gold_sections: list[dict[str, str]]
) -> dict[str, float | None]:
    """Section Recall@K, reciprocal rank and Hit@1 for one question (None without gold)."""
    if not gold_sections:
        return {"section_recall_at_k": None, "section_rr": None, "section_hit_at_1": None}

    def _hits(chunk: dict[str, Any]) -> bool:
        return any(in_section(chunk["source_doc"], chunk["heading_path"], g) for g in gold_sections)

    covered = sum(
        1 for g in gold_sections
        if any(in_section(c["source_doc"], c["heading_path"], g) for c in retrieved)
    )
    rr = 0.0
    for i, chunk in enumerate(retrieved):
        if _hits(chunk):
            rr = 1.0 / (i + 1)
            break
    return {
        "section_recall_at_k": covered / len(gold_sections),
        "section_rr": rr,
        "section_hit_at_1": 1.0 if retrieved and _hits(retrieved[0]) else 0.0,
    }


def score_question(retrieved: list[dict[str, Any]], item: dict[str, Any]) -> dict[str, Any]:
    sources = [c["source_doc"] for c in retrieved]
    return {
        "recall_at_k": compute_recall_at_k(sources, item["source_doc"]),
        "rr": compute_reciprocal_rank(sources, item["source_doc"]),
        **section_metrics(retrieved, item.get("gold_sections") or []),
    }


def summarize_mode(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    """Mean of each metric over the questions that have a value."""
    out: dict[str, Any] = {}
    for metric in METRICS:
        values = [r["modes"][mode][metric] for r in records if r["modes"][mode][metric] is not None]
        out[metric] = sum(values) / len(values) if values else None
        out[f"{metric}_n"] = len(values)
    out["mrr"] = out["rr"]
    out["section_mrr"] = out["section_rr"]
    return out


def compare_modes(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired bootstrap per metric for each (a, b) in COMPARISONS."""
    out: dict[str, Any] = {}
    for a, b in COMPARISONS:
        per_metric = {}
        for metric in METRICS:
            va = {r["id"]: r["modes"][a][metric] for r in records if r["modes"][a][metric] is not None}
            vb = {r["id"]: r["modes"][b][metric] for r in records if r["modes"][b][metric] is not None}
            per_metric[metric] = paired_compare(va, vb, set(va) | set(vb))
        out[f"{b}_vs_{a}"] = per_metric
    return out


def check_hybrid(summary: dict[str, Any], collection: str) -> list[str]:
    """Problems if hybrid does not reproduce the R0 / R1 document-level numbers."""
    expected = EXPECTED_HYBRID.get(collection)
    if expected is None:
        return [f"no expected hybrid numbers for collection {collection!r}"]
    problems = []
    for metric, want in expected.items():
        got = summary["hybrid"][metric]
        if got is None or round(got, 3) != want:
            problems.append(f"hybrid {metric}: got {got}, expected {want:.3f}")
    return problems


def compare_with_reference(records: list[dict[str, Any]], reference: dict[str, Any]) -> dict[str, Any]:
    """Compare hybrid top-5 child_ids with the ``retrieved`` lists of an eval_rag result file."""
    ref = {
        r["id"]: [c["child_id"] for c in r.get("retrieved") or []]
        for r in reference.get("per_question", [])
        if r.get("answerable")
    }
    mismatched = [
        r["id"] for r in records
        if r["id"] in ref and [c["child_id"] for c in r["retrieved"]["hybrid"]] != ref[r["id"]]
    ]
    compared = sum(1 for r in records if r["id"] in ref)
    return {"compared": compared, "same_top5": compared - len(mismatched), "mismatched_ids": mismatched}


# ── Retrieval ─────────────────────────────────────────────────


def retrieve(mode: str, question: str, dense_vector: list[float]) -> list[dict[str, Any]]:
    from app.services.milvus_client import milvus_service

    if mode == "dense":
        hits = milvus_service.search(query_vector=dense_vector, top_k=CANDIDATE_K)
    elif mode == "bm25":
        hits = milvus_service.bm25_search(query_text=question, top_k=CANDIDATE_K)
    elif mode == "hybrid":
        hits = milvus_service.hybrid_search(
            query_vector=dense_vector, query_text=question, top_k=CANDIDATE_K
        )
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return [
        {
            "source_doc": h.source_doc,
            "heading_path": h.heading_path,
            "child_id": h.child_id,
            "score": h.score,
        }
        for h in hits[:TOP_K]
    ]


def run_questions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from app.services.embedding import embedding_service

    records = []
    for i, item in enumerate(items):
        dense_vector = embedding_service.encode_dense(item["question"])[0].tolist()
        retrieved = {mode: retrieve(mode, item["question"], dense_vector) for mode in MODES}
        records.append({
            "id": item["id"],
            "type": item["type"],
            "source_doc": item["source_doc"],
            "gold_sections": item.get("gold_sections") or [],
            "retrieved": retrieved,
            "modes": {mode: score_question(retrieved[mode], item) for mode in MODES},
        })
        logger.info("  [%d/%d] %s done", i + 1, len(items), item["id"])
    return records


# ── Main ──────────────────────────────────────────────────────


def build_meta(dataset_meta: dict[str, Any], started_at: str) -> dict[str, Any]:
    return {
        "started_at": started_at,
        "commit": {
            "sha": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        },
        "collection": {"name": settings.milvus_collection_name, **_collection_meta()},
        "dataset": dataset_meta,
        "config": {
            "modes": list(MODES),
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "rrf_k": RRF_K,
            "hnsw_ef": HNSW_EF,
            "bm25_k1": settings.bm25_k1,
            "bm25_b": settings.bm25_b,
            "enable_reranker_env": settings.enable_reranker,
            "reranker_used": False,
            "llm_used": False,
        },
        "bootstrap": {"n": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED, "comparisons": [f"{b}_vs_{a}" for a, b in COMPARISONS]},
        "embedding": {"name": EMBEDDING_MODEL_NAME, "path": settings.embedding_model_path},
        "versions": {
            "python": sys.version.split()[0],
            "pymilvus": _version("pymilvus"),
            "FlagEmbedding": _version("FlagEmbedding"),
            "torch": _version("torch"),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="qa_l0_v1.json")
    parser.add_argument("--reference", help="eval_rag result file to compare hybrid top-5 child_ids with")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    require_explicit_env()

    collection = settings.milvus_collection_name
    if collection not in COLLECTION_SCOPES:
        raise SystemExit(f"MILVUS_COLLECTION_NAME={collection!r} is not one of {sorted(COLLECTION_SCOPES)}")

    from pymilvus import utility
    from app.services.embedding import embedding_service
    from app.services.milvus_client import milvus_service

    milvus_service.connect()
    # init_collection() creates a missing collection; this script must only read.
    if not utility.has_collection(collection):
        raise SystemExit(f"collection {collection!r} does not exist")
    milvus_service.init_collection()
    embedding_service.warmup()

    t_start = datetime.now(EVAL_TZ)
    items, dataset_meta = load_eval_dataset(args.dataset)
    answerable = [i for i in items if i["answerable"]]
    logger.info("collection=%s answerable=%d modes=%s", collection, len(answerable), MODES)

    meta = build_meta(dataset_meta, t_start.isoformat())
    records = run_questions(answerable)
    summary = {mode: summarize_mode(records, mode) for mode in MODES}
    problems = check_hybrid(summary, collection)
    report: dict[str, Any] = {
        "meta": meta,
        "summary": summary,
        "comparisons": compare_modes(records),
        "self_check": {
            "expected_hybrid": EXPECTED_HYBRID.get(collection),
            "passed": not problems,
            "problems": problems,
        },
        "per_question": records,
    }
    if args.reference:
        with open(args.reference, encoding="utf-8") as f:
            report["self_check"]["reference"] = {
                "file": Path(args.reference).name,
                **compare_with_reference(records, json.load(f)),
            }

    RETRIEVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = RETRIEVAL_OUTPUT_DIR / f"retrieval_ablation_{collection}_{t_start:%Y%m%d_%H%M%S}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    for mode in MODES:
        s = summary[mode]
        logger.info(
            "%-6s Recall@5=%.3f MRR=%.3f | Section Recall@5=%.3f MRR=%.3f Hit@1=%.3f",
            mode, s["recall_at_k"], s["mrr"], s["section_recall_at_k"], s["section_mrr"], s["section_hit_at_1"],
        )
    logger.info("self-check %s %s", "PASSED" if not problems else "FAILED", problems or "")
    logger.info("Saved %s", out)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
