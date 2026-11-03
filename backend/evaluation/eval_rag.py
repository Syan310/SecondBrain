"""RAG Evaluation — RAGAS framework + classic IR metrics + refusal metrics.

Evaluates:
  1. RAGAS metrics: Context Precision, Context Recall, Faithfulness, Answer Relevancy
     (answerable questions only)
  2. Classic IR: Recall@K, MRR, Reranker Lift (answerable questions only)
  3. Refusal: the judge labels every answer refusal / partial / answer.
     Refusal accuracy = refusal questions labelled "refusal";
     false refusal rate = answerable questions labelled "refusal".
  4. Answer correctness: the judge labels every answerable answer
     correct (1.0) / partial (0.5) / incorrect (0.0) against the reference answer.

Modes (--mode):
  rag      retrieve from MILVUS_COLLECTION_NAME, then answer (all metrics above)
  longctx  no retrieval: the whole corpus (--longctx-corpus l0 | l0_l1) goes into
           the system prompt; retrieval and RAGAS metrics are null

Usage:
    cd backend
    MILVUS_HOST=localhost python -m evaluation.eval_rag
    MILVUS_HOST=localhost python -m evaluation.eval_rag --profile smoke --dataset qa_l0_v1.json
    MILVUS_HOST=localhost python -m evaluation.eval_rag --check-rerank-length --dataset qa_l0_v1.json
    MILVUS_HOST=localhost MILVUS_COLLECTION_NAME=secondbrain_docs ENABLE_RERANKER=false       RERANKER_USE_HEADING=false CITATION_MODE=off       python -m evaluation.eval_rag --profile full --dataset qa_l0_v1.json --mode longctx --longctx-corpus l0

Requires Milvus, embedding/reranker models, and LLM APIs to be available.
Report filename and meta timestamps use America/Los_Angeles local wall clock.
"""

from __future__ import annotations

import asyncio
import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.chunk_text import reranker_passages
from app.services.citations import is_not_found
from app.services.embedding import embedding_service
from app.services.milvus_client import milvus_service
from app.services.reranker import reranker_service
from app.services.llm_client import build_routing_extra_body, llm_client, usage_tracker
from app.agent.tools.rag_search import rag_search
from evaluation.eval_utils import EVAL_TZ, configure_run_logging, safe_p50, safe_p95, warmup_and_health_check
from evaluation.judge_client import judge_client
from evaluation.profiles import EvalProfile, build_balanced_subset, get_eval_profile
from evaluation.ragas_runner import (
    build_ragas_samples,
    compute_ragas_means,
    run_ragas_evaluation,
)

logger = logging.getLogger(__name__)

DATASET_DIR = Path(__file__).parent / "datasets"
DEFAULT_DATASET = "qa_dataset.json"
OUTPUT_DIR = Path(__file__).parent / "results"
RAG_OUTPUT_DIR = OUTPUT_DIR / "rag"
RAG_LOG_DIR = OUTPUT_DIR / "logs_rag"
REPO_ROOT = Path(__file__).resolve().parents[2]

TOP_K = 5
CANDIDATE_K = TOP_K * 3  # rag_search retrieves top_k * 3 candidates
RRF_K = 60
QA_TEMPERATURE = 0.1
QA_MAX_TOKENS = 1024
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# L0 corpus: the 4 notes ingested for this evaluation (hashes go into meta).
CORPUS_DIR = Path(os.environ.get("EVAL_CORPUS_DIR", r"C:\Users\Shaoyu Yan\Desktop\LLM_Learn"))
CORPUS_FILES = (
    "Agent 复习.md",
    "RAG复习.md",
    "Transformer 复习.md",
    "你不知道的Agent-原理架构与工程实践.md",
)

# Collection -> corpus scope. A RAG run on any other collection is refused, so
# every result file says which corpus it answered from.
COLLECTION_SCOPES = {"secondbrain_docs": "l0", "secondbrain_l1": "l0_l1"}
CORPUS_SCOPES = ("l0", "l0_l1")

# Refusal questions that the L1 notes can answer (docs/l1-refusal-review.md,
# confirmed by the user). On the l0_l1 corpus they do not count toward refusal accuracy.
L1_ANSWERABLE_REFUSALS = frozenset({"q053", "q054", "q055", "q056", "q060"})

# Long-context corpus l0_l1: the files listed in this ingest record, read from
# the paths in the (local, untracked) corpus manifest.
INGEST_RECORD_L1 = REPO_ROOT / "docs" / "ingest-record-l1.md"
CORPUS_MANIFEST = REPO_ROOT / "data" / "corpus" / "manifest.json"

# deepseek-flash (DeepSeek-V4.1-Flash-1017), official pricing page checked 2026-11-02:
# https://api-docs.deepseek.com/quick_start/pricing  — USD per 1M tokens.
DEEPSEEK_CONTEXT_LIMIT = 1_000_000
DEEPSEEK_PRICES = {
    "peak": {"cache_hit": 0.006, "cache_miss": 0.3, "output": 1.2},
    "off_peak": {"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.6},
}
# Peak: Mon–Fri 01:00–04:00 and 06:00–10:00 UTC (Chinese public holidays are
# off-peak; not modelled here).
DEEPSEEK_PEAK_HOURS_UTC = ((1, 4), (6, 10))

# RAGAS column name -> per-question field name
RAGAS_FIELD_NAMES = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
}

# ── 1. Dataset ────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_text_lf(path: Path) -> str:
    """SHA-256 after CRLF -> LF, so it matches the blob git stores (autocrlf checkouts)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def normalize_dataset(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return eval items with the qa_l0_v1 fields.

    The legacy ``qa_dataset.json`` has no ``answerable`` field. Its single-hop
    questions become ``q001``... in file order (multi-hop is evaluated at Agent level).
    """
    if raw and all("answerable" in item for item in raw):
        return [
            {
                **item,
                "type": item.get("type", "original"),
                "source_doc": item.get("source_doc") or "",
                "gold_sections": item.get("gold_sections") or [],
            }
            for item in raw
        ]

    single_hop = [q for q in raw if q.get("question_type") != "multi-hop"]
    return [
        {
            "id": f"q{i + 1:03d}",
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "type": "original",
            "original_type": item.get("question_type"),
            "answerable": True,
            "source_doc": item.get("source_doc", ""),
            "gold_sections": [],
            "origin": "original_v1",
        }
        for i, item in enumerate(single_hop)
    ]


def load_eval_dataset(name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(name) if Path(name).is_absolute() else DATASET_DIR / name
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = normalize_dataset(raw)
    type_counts: dict[str, int] = {}
    for item in items:
        type_counts[item["type"]] = type_counts.get(item["type"], 0) + 1
    return items, {
        "name": path.name,
        "sha256": _sha256_text_lf(path),
        "items": len(items),
        "answerable": sum(1 for i in items if i["answerable"]),
        "refusal": sum(1 for i in items if not i["answerable"]),
        "type_counts": type_counts,
    }


def select_eval_subset(
    items: list[dict[str, Any]], profile: EvalProfile
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Answerable questions (balanced across source docs) + refusal questions."""
    answerable_pool = [i for i in items if i["answerable"]]
    if profile.rag_answerable_types is not None:
        answerable_pool = [i for i in answerable_pool if i["type"] in profile.rag_answerable_types]
    answerable, answerable_meta = build_balanced_subset(
        answerable_pool,
        profile.rag_single_hop_limit,
        group_key=lambda item: item.get("source_doc") or "__unknown__",
    )
    refusal_pool = [i for i in items if not i["answerable"]]
    refusal = (
        refusal_pool
        if profile.rag_refusal_limit is None
        else refusal_pool[: profile.rag_refusal_limit]
    )
    return answerable + refusal, {
        "answerable_limit": profile.rag_single_hop_limit,
        "answerable_types": list(profile.rag_answerable_types) if profile.rag_answerable_types else None,
        "refusal_limit": profile.rag_refusal_limit,
        "answerable_selected": len(answerable),
        "refusal_selected": len(refusal),
        "selected": len(answerable) + len(refusal),
        "group_counts": answerable_meta["group_counts"],
        "ids": [i["id"] for i in answerable + refusal],
    }


# ── 2. Prompt and citations ───────────────────────────────────


QA_SYSTEM_PROMPT = """\
你是一个知识问答助手。根据以下检索到的文档片段回答用户问题。

## 回答要求

- 你的回答必须完全基于以下检索片段中的信息
- 允许对多个片段的内容进行归纳、对比和因果推理
- 禁止引入检索内容中未提及的概念、数据、案例或结论
- 如果检索片段不足以回答问题，明确指出信息不足，不要用自身知识补充
- 篇幅控制在 200 字以内，只保留核心结论和关键依据
{citation_rules}
检索到的文档片段：
{contexts}
"""

VALID_CITATION_MODES = ("off", "prompt", "filter", "fallback")


def build_qa_prompt_template(citation_mode: str) -> str:
    """QA system prompt template for one citation mode ({contexts} still unfilled).

    ``prompt`` / ``filter`` append CITATION_RULES and ``fallback`` appends FALLBACK_RULES
    from app.services.citations, so the eval pipeline and the QA sub-agent use the same wording.
    """
    if citation_mode not in VALID_CITATION_MODES:
        raise ValueError(f"unknown citation_mode: {citation_mode!r}")
    rules = ""
    if citation_mode != "off":
        from app.services.citations import rules_for_mode

        rules = rules_for_mode(citation_mode).strip("\n") + "\n"
    return QA_SYSTEM_PROMPT.replace("{citation_rules}", rules)


def apply_citation_mode(answer: str, n_contexts: int, citation_mode: str) -> dict[str, Any]:
    """Post-process one answer. Returns clean_answer plus citation stats (None when off).

    ``off`` still strips [n] markers (prompt-mode cleanup): the contexts are numbered,
    so the model may cite on its own, and RAGAS / refusal judging must see clean text.
    """
    from app.services.citations import apply_citations

    valid_ids = set(range(1, n_contexts + 1))
    if citation_mode == "off":
        return {"clean_answer": apply_citations(answer, valid_ids, "prompt").clean_answer, "citation": None}

    result = apply_citations(answer, valid_ids, citation_mode)
    return {
        "clean_answer": result.clean_answer,
        "citation": {
            "units_total": result.units_total,
            "units_kept": result.units_kept,
            "invalid_citations": result.invalid_citations,
            "fully_refused": result.fully_refused,
        },
    }


# ── 3. Run RAG pipeline per question ──────────────────────────


async def run_rag_pipeline(
    question: str,
    prompt_template: str,
    citation_mode: str = "off",
    top_k: int = TOP_K,
) -> dict:
    """Run full RAG pipeline: retrieve + generate answer + citation post-processing.

    Returns: {answer (clean), raw_answer, citation, contexts, context_texts, *_ms}
    """
    t0 = time.perf_counter()

    # Retrieve
    ranked_chunks = await asyncio.to_thread(rag_search, query=question, top_k=top_k)
    t_retrieve = time.perf_counter()

    context_texts = [c.text for c in ranked_chunks]
    contexts_str = "\n\n".join(
        f"[{i+1}] (来源: {c.source_doc} | {c.heading_path})\n{c.text}"
        for i, c in enumerate(ranked_chunks)
    )

    # Generate answer
    messages = [
        {"role": "system", "content": prompt_template.replace("{contexts}", contexts_str)},
        {"role": "user", "content": question},
    ]
    llm_started_at = datetime.now(timezone.utc)
    resp = await llm_client.chat(
        messages=messages, temperature=QA_TEMPERATURE, max_tokens=QA_MAX_TOKENS
    )
    raw_answer = resp.get("content") or ""
    post = apply_citation_mode(raw_answer, len(ranked_chunks), citation_mode)
    t_generate = time.perf_counter()

    return {
        "answer": post["clean_answer"],
        "raw_answer": raw_answer,
        "citation": post["citation"],
        "contexts": ranked_chunks,
        "context_texts": context_texts,
        "usage": resp.get("usage"),
        "llm_started_at": llm_started_at.isoformat(),
        "retrieve_ms": (t_retrieve - t0) * 1000,
        "generate_ms": (t_generate - t_retrieve) * 1000,
        "total_ms": (t_generate - t0) * 1000,
    }


# ── 3b. Long-context mode (task doc §4.2) ─────────────────────


_INGEST_ROW_RE = re.compile(r"^\| L[01] \| `([^`]+)` \| [\d,]+ \| `([0-9a-f]{64})` \|")


def parse_ingest_record(text: str) -> list[tuple[str, str]]:
    """(name, sha256) of every row in the per-file table of ingest-record-l1.md."""
    rows = []
    for line in text.splitlines():
        match = _INGEST_ROW_RE.match(line)
        if match:
            rows.append((match.group(1), match.group(2)))
    return rows


def longctx_files(corpus: str) -> list[dict[str, Any]]:
    """Files of one long-context corpus, sorted by name: name, path, sha256.

    ``l0``: the 4 L0 notes. ``l0_l1``: every file in the ingest record, located
    through the corpus manifest by SHA-256; a file whose content changed is refused.
    """
    if corpus == "l0":
        files = [{"name": name, "path": CORPUS_DIR / name, "expected_sha256": None} for name in CORPUS_FILES]
    elif corpus == "l0_l1":
        manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
        path_by_sha = {e["sha256"]: Path(e["path"]) for e in manifest["included"]}
        files = []
        for name, sha in parse_ingest_record(INGEST_RECORD_L1.read_text(encoding="utf-8")):
            if sha not in path_by_sha:
                raise ValueError(f"{name}: sha256 {sha[:12]} is not in {CORPUS_MANIFEST}")
            files.append({"name": name, "path": path_by_sha[sha], "expected_sha256": sha})
        if not files:
            raise ValueError(f"no files found in {INGEST_RECORD_L1}")
    else:
        raise ValueError(f"unknown long-context corpus: {corpus!r}")
    files.sort(key=lambda f: f["name"])
    for f in files:
        f["sha256"] = _sha256_file(f["path"])
        if f["expected_sha256"] and f["expected_sha256"] != f["sha256"]:
            raise ValueError(f"{f['name']}: file changed since ingestion (sha256 {f['sha256'][:12]})")
    return files


def build_longctx_context(files: list[dict[str, Any]]) -> str:
    """All files in one block, each preceded by a [来源: <name>] line."""
    parts = []
    for f in files:
        text = Path(f["path"]).read_text(encoding="utf-8").replace("\r\n", "\n").strip()
        parts.append(f"[来源: {f['name']}]\n{text}")
    return "\n\n".join(parts)


async def run_longctx_pipeline(question: str, system_prompt: str) -> dict:
    """Answer one question with the whole corpus in the system prompt (no retrieval)."""
    t0 = time.perf_counter()
    llm_started_at = datetime.now(timezone.utc)
    resp = await llm_client.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=QA_TEMPERATURE,
        max_tokens=QA_MAX_TOKENS,
    )
    raw_answer = resp.get("content") or ""
    post = apply_citation_mode(raw_answer, 0, "off")
    t_generate = time.perf_counter()
    return {
        "answer": post["clean_answer"],
        "raw_answer": raw_answer,
        "citation": None,
        "contexts": None,
        "context_texts": None,
        "usage": resp.get("usage"),
        "llm_started_at": llm_started_at.isoformat(),
        "retrieve_ms": None,
        "generate_ms": (t_generate - t0) * 1000,
        "total_ms": (t_generate - t0) * 1000,
    }


async def count_longctx_tokens(system_prompt: str) -> dict[str, Any]:
    """Prompt tokens of the long-context system prompt, counted by the DeepSeek API.

    One request with max_tokens=1. Its usage and cost go into meta. It also warms
    the prefix cache, as it would be in a deployed long-context system.
    """
    llm_started_at = datetime.now(timezone.utc)
    resp = await llm_client.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "你好"},
        ],
        temperature=QA_TEMPERATURE,
        max_tokens=1,
    )
    return {"usage": resp.get("usage"), **llm_cost(resp.get("usage"), llm_started_at)}


# ── 3c. Token cost (task doc §4.3) ────────────────────────────


def is_deepseek_peak(ts: datetime) -> bool:
    ts = ts.astimezone(timezone.utc)
    return ts.weekday() < 5 and any(lo <= ts.hour < hi for lo, hi in DEEPSEEK_PEAK_HOURS_UTC)


def llm_cost(usage: dict[str, Any] | None, started_at: datetime) -> dict[str, Any]:
    """Tokens and cost of one DeepSeek request, at the price of the hour it started.

    ``cost_usd_actual`` uses the reported cache hits; ``cost_usd_no_cache`` prices
    every prompt token as a cache miss.
    """
    if not usage:
        return {
            "input_tokens": None, "output_tokens": None, "cache_hit_tokens": None,
            "cache_miss_tokens": None, "price_period": None,
            "cost_usd_actual": None, "cost_usd_no_cache": None,
        }
    prompt = int(usage.get("prompt_tokens") or 0)
    hit = int(usage.get("prompt_cache_hit_tokens") or 0)
    miss = usage.get("prompt_cache_miss_tokens")
    miss = prompt - hit if miss is None else int(miss)
    out = int(usage.get("completion_tokens") or 0)
    period = "peak" if is_deepseek_peak(started_at) else "off_peak"
    price = DEEPSEEK_PRICES[period]
    return {
        "input_tokens": prompt,
        "output_tokens": out,
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "price_period": period,
        "cost_usd_actual": (hit * price["cache_hit"] + miss * price["cache_miss"] + out * price["output"]) / 1e6,
        "cost_usd_no_cache": ((hit + miss) * price["cache_miss"] + out * price["output"]) / 1e6,
    }


def summarize_costs(recs: list[dict[str, Any]], preflight: dict[str, Any] | None = None) -> dict[str, Any]:
    """Per-question means and run totals of tokens and DeepSeek cost (both pricings)."""
    priced = [r for r in recs if r.get("input_tokens") is not None]

    def _sum(key: str) -> float:
        return sum(r[key] for r in priced)

    n = len(priced)
    prompt_total = _sum("input_tokens")
    out: dict[str, Any] = {
        "n": n,
        "unpriced": len(recs) - n,
        "input_tokens_mean": prompt_total / n if n else None,
        "output_tokens_mean": _sum("output_tokens") / n if n else None,
        "cache_hit_ratio": _sum("cache_hit_tokens") / prompt_total if prompt_total else None,
        "cost_usd_actual_mean": _sum("cost_usd_actual") / n if n else None,
        "cost_usd_no_cache_mean": _sum("cost_usd_no_cache") / n if n else None,
        "cost_usd_actual_total": _sum("cost_usd_actual"),
        "cost_usd_no_cache_total": _sum("cost_usd_no_cache"),
        "price_periods": {
            period: sum(1 for r in priced if r["price_period"] == period) for period in DEEPSEEK_PRICES
        },
        "prices_usd_per_1m": DEEPSEEK_PRICES,
        "prices_checked": "2026-11-02, https://api-docs.deepseek.com/quick_start/pricing",
    }
    if preflight is not None:
        out["preflight_cost_usd_actual"] = preflight.get("cost_usd_actual")
        out["preflight_cost_usd_no_cache"] = preflight.get("cost_usd_no_cache")
    return out


# ── 4. Classic IR metrics ─────────────────────────────────────


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


def compute_reranker_lift(question: str, top_k: int = TOP_K) -> dict:
    """Compare MRR with/without reranker for one question.

    Returns raw search sources and reranked sources for external MRR comparison.
    """
    # Encode query (dense only; the raw question goes to the BM25 path)
    dense = embedding_service.encode_dense(question)[0].tolist()

    # Raw hybrid search (no rerank)
    raw_candidates = milvus_service.hybrid_search(
        query_vector=dense, query_text=question, top_k=top_k
    )
    raw_sources = [c.source_doc for c in raw_candidates]

    # Reranked
    candidate_k = top_k * 3
    candidates = milvus_service.hybrid_search(
        query_vector=dense, query_text=question, top_k=candidate_k
    )
    if candidates:
        # Same reranker input as rag_search (follows RERANKER_USE_HEADING)
        passages = reranker_passages(candidates, settings.reranker_use_heading)
        reranked = reranker_service.rerank(query=question, passages=passages, top_n=top_k)
        reranked_sources = [candidates[r.original_index].source_doc for r in reranked]
    else:
        reranked_sources = []

    return {"raw_sources": raw_sources, "reranked_sources": reranked_sources}


# ── 5. Refusal judging ────────────────────────────────────────


REFUSAL_LABELS = ("refusal", "partial", "answer")

REFUSAL_JUDGE_PROMPT = """\
你是评估员。判断下面这个回答属于哪一类：

- refusal：回答说明资料或知识库里没有足够的信息，没有给出针对问题的实质内容
- partial：回答说明信息不足，但仍然给出了一部分针对问题的实质内容
- answer：回答直接给出了针对问题的实质内容

只看回答本身是否给出了实质内容，不要判断内容对不对。
只输出一个 JSON 对象，不要输出其他文字：
{{"label": "refusal" 或 "partial" 或 "answer", "reason": "一句话理由"}}

问题：
{question}

回答：
{answer}
"""


def parse_refusal_label(text: str) -> tuple[str | None, str]:
    """Extract (label, reason) from the judge output; label is None if unparseable."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None, ""
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None, ""
    label = str(data.get("label", "")).strip().lower()
    return (label if label in REFUSAL_LABELS else None), str(data.get("reason", ""))


async def judge_refusal(question: str, answer: str, attempts: int = 2) -> dict[str, Any]:
    prompt = REFUSAL_JUDGE_PROMPT.format(question=question, answer=answer)
    last_output, last_error = "", None
    for _ in range(attempts):
        try:
            last_output = await judge_client.judge(
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1024,
            )
            last_error = None
        except Exception as exc:  # recorded per question, run continues
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        label, reason = parse_refusal_label(last_output)
        if label:
            return {"refusal_label": label, "refusal_reason": reason}
    return {
        "refusal_label": None,
        "refusal_reason": "",
        "refusal_error": last_error or f"unparseable judge output: {last_output[:200]}",
    }


# ── 5b. Answer correctness judging (task doc §5.2) ────────────


CORRECTNESS_SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

CORRECTNESS_JUDGE_PROMPT = """\
你是评估员。对照参考答案，判断下面这个回答属于哪一类：

- correct：回答覆盖了参考答案的全部要点，没有和参考答案矛盾的内容
- partial：回答覆盖了参考答案的部分要点，或者有小错
- incorrect：回答没有覆盖参考答案的要点、和参考答案矛盾，或者拒绝回答

只比较回答和参考答案的要点。回答里多出的内容，只要不和参考答案矛盾，就不扣分。
只输出一个 JSON 对象，不要输出其他文字：
{{"label": "correct" 或 "partial" 或 "incorrect", "reason": "一句话理由"}}

问题：
{question}

参考答案：
{ground_truth}

回答：
{answer}
"""


def parse_correctness_label(text: str) -> tuple[str | None, str]:
    """Extract (label, reason) from the judge output; label is None if unparseable."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None, ""
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None, ""
    label = str(data.get("label", "")).strip().lower()
    return (label if label in CORRECTNESS_SCORES else None), str(data.get("reason", ""))


async def judge_correctness(
    question: str, ground_truth: str, answer: str, attempts: int = 2
) -> dict[str, Any]:
    """Label one answer against the reference answer. No retrieved contexts are shown,
    so RAG and long-context answers are judged the same way."""
    prompt = CORRECTNESS_JUDGE_PROMPT.format(question=question, ground_truth=ground_truth, answer=answer)
    last_output, last_error = "", None
    for _ in range(attempts):
        try:
            last_output = await judge_client.judge(
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1024,
            )
            last_error = None
        except Exception as exc:  # recorded per question, run continues
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        label, reason = parse_correctness_label(last_output)
        if label:
            return {
                "correctness_label": label,
                "correctness_score": CORRECTNESS_SCORES[label],
                "correctness_reason": reason,
            }
    return {
        "correctness_label": "unjudged",
        "correctness_score": None,
        "correctness_reason": "",
        "correctness_error": last_error or f"unparseable judge output: {last_output[:200]}",
    }


def summarize_correctness(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean correctness score over judged answerable questions, and label counts."""
    answerable = [r for r in recs if r["answerable"]]
    scores = [r["correctness_score"] for r in answerable if r.get("correctness_score") is not None]
    return {
        "n": len(answerable),
        "judged": len(scores),
        "unjudged": sum(1 for r in answerable if r.get("correctness_label") == "unjudged"),
        "correctness": _mean(scores) if scores else None,
        "label_counts": {
            label: sum(1 for r in answerable if r.get("correctness_label") == label)
            for label in (*CORRECTNESS_SCORES, "unjudged")
        },
    }


# ── 6. RAGAS evaluation ───────────────────────────────────────


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


# ── 7. Run meta (M1) ──────────────────────────────────────────


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _milvus_server_version() -> str | None:
    try:
        from pymilvus import utility

        return utility.get_server_version()
    except Exception as exc:
        return f"unknown ({type(exc).__name__})"


def _collection_meta() -> dict[str, Any]:
    """Total rows and distinct source_doc count of the current Milvus collection."""
    try:
        col = milvus_service.collection
        total = col.query(expr="", output_fields=["count(*)"])[0]["count(*)"]
        if total > 16384:  # Milvus caps one query at 16384 rows
            return {"total_rows": total, "source_docs": "unknown (> 16384 rows)"}
        rows = col.query(expr="chunk_index >= 0", output_fields=["source_doc"], limit=16384)
        return {"total_rows": total, "source_docs": len({r["source_doc"] for r in rows})}
    except Exception as exc:
        return {"total_rows": f"unknown ({type(exc).__name__})", "source_docs": f"unknown ({type(exc).__name__})"}


def _corpus_meta(mode: str = "rag", scope: str | None = None) -> dict[str, Any]:
    """SHA-256 of each L0 note and its chunk count in Milvus, plus the corpus that
    answered the questions: the collection (rag) or None (longctx, see meta.longctx)."""
    files = {}
    for name in CORPUS_FILES:
        path = CORPUS_DIR / name
        entry: dict[str, Any] = {"sha256": _sha256_file(path) if path.exists() else None}
        try:
            rows = milvus_service.collection.query(
                expr=f'source_doc == "{name}"', output_fields=["count(*)"]
            )
            entry["chunks"] = rows[0]["count(*)"] if rows else 0
        except Exception as exc:
            entry["chunks"] = f"unknown ({type(exc).__name__})"
        files[name] = entry
    out: dict[str, Any] = {"dir": str(CORPUS_DIR), "files": files, "scope": scope}
    if mode == "rag":
        out["collection"] = settings.milvus_collection_name
        out.update(_collection_meta())
    else:
        out.update({"collection": None, "total_rows": None, "source_docs": None})
    return out


def _expected_providers(base_url: str, provider_order: list[str]) -> list[str]:
    if "openrouter.ai" in base_url:
        return [p.lower() for p in provider_order]
    from httpx import URL

    return [(URL(base_url).host or base_url).lower()]


REQUIRED_PHASES = {
    "rag": ("ir", "refusal", "correctness", "ragas"),
    "longctx": ("generate", "refusal", "correctness"),
}


def check_run_completeness(report: dict[str, Any]) -> list[str]:
    """List reasons a run is incomplete: a phase did not finish, a RAGAS metric
    errored, or an answer could not be judged for correctness."""
    mode = report["meta"].get("mode", "rag")
    problems = [
        f"phase {phase!r} did not complete"
        for phase in REQUIRED_PHASES[mode]
        if phase not in report["meta"].get("completed_phases", [])
    ]
    for metric, status in (report.get("ragas_metric_status") or {}).items():
        if status.get("status") == "error":
            problems.append(f"RAGAS {metric} failed: {status.get('error_type')}")
    unjudged = (report.get("correctness_metrics") or {}).get("unjudged")
    if unjudged:
        problems.append(f"correctness: {unjudged} answer(s) unjudged")
    return problems


def check_providers(usage: dict[str, dict[str, Any]]) -> list[str]:
    """Return problems if any request was served by an unexpected provider."""
    expected = {
        "llm": _expected_providers(settings.llm_base_url, settings.llm_provider_order),
        "judge": _expected_providers(settings.judge_base_url, settings.judge_provider_order),
    }
    problems = []
    for role, allowed in expected.items():
        if not allowed:
            continue
        for provider, n in usage.get(role, {}).get("providers", {}).items():
            if provider.lower() not in allowed:
                problems.append(f"{role}: {n} request(s) served by {provider!r}, expected {allowed}")
    return problems


def build_static_meta(prompt_template: str, mode: str = "rag", scope: str | None = None) -> dict[str, Any]:
    """Meta known before the run starts (config, versions, hashes)."""
    head = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    llm_extra = build_routing_extra_body(settings.llm_base_url, settings.llm_provider_order)
    judge_extra = build_routing_extra_body(settings.judge_base_url, settings.judge_provider_order)
    return {
        "commit": {"sha": head, "dirty": dirty},
        "config": {
            "enable_reranker": settings.enable_reranker,
            "reranker_use_heading": settings.reranker_use_heading,
            "citation_mode": settings.citation_mode,
            "reranker_max_length": reranker_service.max_length,
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "rrf_k": RRF_K,
            "bm25_k1": settings.bm25_k1,
            "bm25_b": settings.bm25_b,
            "qa_temperature": QA_TEMPERATURE,
            "qa_max_tokens": QA_MAX_TOKENS,
        },
        "prompt_sha256": _sha12(prompt_template),
        "refusal_judge_prompt_sha256": _sha12(REFUSAL_JUDGE_PROMPT),
        "correctness_judge_prompt_sha256": _sha12(CORRECTNESS_JUDGE_PROMPT),
        "corpus": _corpus_meta(mode, scope),
        "models": {
            "llm": {
                "model": settings.llm_model,
                "base_url": settings.llm_base_url,
                "provider_order": settings.llm_provider_order,
                "extra_body": llm_extra,
            },
            "judge": {
                "model": settings.judge_model,
                "base_url": settings.judge_base_url,
                "provider_order": settings.judge_provider_order,
                "extra_body": judge_extra,
                "sampling": "temperature/top_p not sent (provider default)",
                "reasoning": "not sent (OpenRouter default)",
            },
            "embedding": {"name": EMBEDDING_MODEL_NAME, "path": settings.embedding_model_path},
            "reranker": {"name": RERANKER_MODEL_NAME, "path": settings.reranker_model_path},
        },
        "versions": {
            "python": sys.version.split()[0],
            "pymilvus": _version("pymilvus"),
            "milvus_server": _milvus_server_version(),
            "ragas": _version("ragas"),
            "langgraph": _version("langgraph"),
            "openai": _version("openai"),
            "FlagEmbedding": _version("FlagEmbedding"),
            "transformers": _version("transformers"),
            "torch": _version("torch"),
        },
    }


# ── 8. Rerank input length check (task doc §4.3) ──────────────


def check_rerank_lengths(items: list[dict[str, Any]], limit: int = 512) -> dict[str, Any]:
    """Token length of every [query, passage] pair the reranker would see.

    Counts both the text-only and the heading-augmented passages for the
    CANDIDATE_K candidates of each question.
    """
    tokenizer = reranker_service._load_model().tokenizer
    lengths: dict[str, list[int]] = {"text": [], "heading": []}
    for item in items:
        question = item["question"]
        dense = embedding_service.encode_dense(question)[0].tolist()
        candidates = milvus_service.hybrid_search(
            query_vector=dense, query_text=question, top_k=CANDIDATE_K
        )
        for key, use_heading in (("text", False), ("heading", True)):
            for passage in reranker_passages(candidates, use_heading):
                ids = tokenizer(question, passage, truncation=False)["input_ids"]
                lengths[key].append(len(ids))
    summary = {}
    for key, values in lengths.items():
        over = sum(1 for v in values if v > limit)
        summary[key] = {
            "pairs": len(values),
            "max": max(values) if values else 0,
            "over_limit": over,
            "over_ratio": over / len(values) if values else 0.0,
        }
    return {"limit": limit, "questions": len(items), "summary": summary}


# ── 9. Main ───────────────────────────────────────────────────


def _rag_report_path(run_ts: str) -> Path:
    """Single JSON report path for one eval run (under results/rag/)."""
    return RAG_OUTPUT_DIR / f"rag_eval_{run_ts}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG evaluation with a preset profile.")
    parser.add_argument(
        "--profile",
        choices=("smoke", "standard", "full"),
        default=os.environ.get("EVAL_PROFILE", "standard"),
        help="Evaluation profile preset.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Dataset file name under evaluation/datasets/ (or an absolute path).",
    )
    parser.add_argument(
        "--mode",
        choices=("rag", "longctx"),
        default="rag",
        help="rag: retrieve then answer; longctx: the whole corpus in the prompt, no retrieval.",
    )
    parser.add_argument(
        "--longctx-corpus",
        choices=CORPUS_SCOPES,
        default=None,
        help="Corpus for --mode longctx: l0 (4 notes) or l0_l1 (every file in docs/ingest-record-l1.md).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Parallel questions (default: the profile's value).",
    )
    parser.add_argument(
        "--check-rerank-length",
        action="store_true",
        help="Only count reranker [query, passage] token lengths (no LLM calls).",
    )
    args = parser.parse_args()
    if args.mode == "longctx" and args.longctx_corpus is None:
        parser.error("--mode longctx needs --longctx-corpus")
    if args.mode == "rag" and args.longctx_corpus is not None:
        parser.error("--longctx-corpus only applies to --mode longctx")
    return args


def _save_report(report: dict, output_path: Path) -> None:
    """Atomically write partial/final report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(output_path)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def refusal_scored(item: dict[str, Any], scope: str | None) -> bool:
    """False for a refusal question that the L1 notes answer, on the l0_l1 corpus (§5.3)."""
    return item["answerable"] or not (scope == "l0_l1" and item["id"] in L1_ANSWERABLE_REFUSALS)


def _per_question_record(
    item: dict[str, Any], rag_result: dict[str, Any], scope: str | None = None
) -> dict[str, Any]:
    started = rag_result.get("llm_started_at")
    cost = llm_cost(rag_result.get("usage"), datetime.fromisoformat(started) if started else datetime.now(timezone.utc))
    contexts = rag_result["contexts"]
    return {
        "id": item["id"],
        "type": item["type"],
        "answerable": item["answerable"],
        "origin": item.get("origin"),
        "question": item["question"],
        "source_doc": item.get("source_doc", ""),
        "ground_truth": item.get("ground_truth", ""),
        "refusal_scored": refusal_scored(item, scope),
        "raw_answer": rag_result["raw_answer"],
        "clean_answer": rag_result["answer"],
        "not_found": is_not_found(rag_result["answer"]),
        "citation": rag_result["citation"],
        "retrieved": None if contexts is None else [
            {"source_doc": c.source_doc, "heading_path": c.heading_path, "child_id": c.child_id}
            for c in contexts
        ],
        "latency_ms": rag_result["total_ms"],
        "llm_started_at": started,
        "llm_usage": rag_result.get("usage"),
        **cost,
    }


def summarize_refusals(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Refusal accuracy on refusal questions and false refusals on answerable ones.

    Refusal questions with ``refusal_scored`` False (answered by the L1 notes) are
    left out of refusal accuracy and listed in ``refusal_excluded``.
    """
    refusal_recs = [r for r in recs if not r["answerable"] and r.get("refusal_scored", True)]
    answerable_recs = [r for r in recs if r["answerable"]]

    def _refused(rs: list[dict[str, Any]]) -> int:
        return sum(1 for r in rs if r.get("refusal_label") == "refusal")

    return {
        "refusal_correct": _refused(refusal_recs),
        "refusal_total": len(refusal_recs),
        "refusal_accuracy": _refused(refusal_recs) / len(refusal_recs) if refusal_recs else None,
        "false_refusals": _refused(answerable_recs),
        "answerable_total": len(answerable_recs),
        "false_refusal_rate": (
            _refused(answerable_recs) / len(answerable_recs) if answerable_recs else None
        ),
        "unjudged": sum(1 for r in recs if r.get("refusal_label") is None),
        "refusal_excluded": [r["id"] for r in recs if not r["answerable"] and not r.get("refusal_scored", True)],
        "label_counts": {
            label: sum(1 for r in recs if r.get("refusal_label") == label)
            for label in REFUSAL_LABELS
        },
    }


def summarize_not_found(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Answers starting with NOT_FOUND_SENTENCE: on scored refusal questions (wanted)
    and on answerable questions (false "not found"); the latter split by correctness."""
    refusal_recs = [r for r in recs if not r["answerable"] and r.get("refusal_scored", True)]
    answerable_nf = [r for r in recs if r["answerable"] and r.get("not_found")]
    return {
        "refusal_not_found": sum(1 for r in refusal_recs if r.get("not_found")),
        "refusal_total": len(refusal_recs),
        "answerable_not_found": len(answerable_nf),
        "answerable_total": sum(1 for r in recs if r["answerable"]),
        "answerable_not_found_ids": [r["id"] for r in answerable_nf],
        "answerable_not_found_correctness": {
            label: sum(1 for r in answerable_nf if r.get("correctness_label") == label)
            for label in (*CORRECTNESS_SCORES, "unjudged")
        },
        "excluded_not_found_ids": [
            r["id"] for r in recs if not r["answerable"] and not r.get("refusal_scored", True) and r.get("not_found")
        ],
    }


def summarize_citations(recs: list[dict[str, Any]]) -> dict[str, Any]:
    cites = [r["citation"] for r in recs if r.get("citation")]
    total = sum(c["units_total"] for c in cites)
    kept = sum(c["units_kept"] for c in cites)
    return {
        "units_total": total,
        "units_kept": kept,
        "drop_ratio": (total - kept) / total if total else 0.0,
        "invalid_citations": sum(c["invalid_citations"] for c in cites),
        "fully_refused": [r["id"] for r in recs if (r.get("citation") or {}).get("fully_refused")],
    }


async def run_check_rerank_length(dataset: str) -> None:
    milvus_service.connect()
    milvus_service.init_collection()
    items, dataset_meta = load_eval_dataset(dataset)
    result = check_rerank_lengths(items)
    result["dataset"] = dataset_meta
    out = RAG_OUTPUT_DIR / f"rerank_length_{datetime.now(EVAL_TZ).strftime('%Y%m%d_%H%M%S')}.json"
    _save_report(result, out)
    logger.info("Rerank length check: %s", json.dumps(result["summary"], indent=2))
    logger.info("Saved to %s", out)


def require_citation_mode_env(environ: dict[str, str] | None = None) -> str:
    """Return CITATION_MODE from the environment; exit if unset or unknown.

    The eval must not fall back to the online default (config.py: "filter"),
    so every run states its mode on the command line.
    """
    env = os.environ if environ is None else environ
    mode = env.get("CITATION_MODE")
    if mode is None:
        raise SystemExit(
            "CITATION_MODE is not set. Set it explicitly for eval runs: "
            f"one of {', '.join(VALID_CITATION_MODES)}."
        )
    if mode not in VALID_CITATION_MODES:
        raise SystemExit(f"CITATION_MODE={mode!r} is invalid; use one of {', '.join(VALID_CITATION_MODES)}.")
    return mode


EXPLICIT_ENV_VARS = ("MILVUS_COLLECTION_NAME", "ENABLE_RERANKER", "RERANKER_USE_HEADING", "CITATION_MODE")


def require_explicit_env(environ: dict[str, str] | None = None) -> None:
    """Exit unless every retrieval switch is set on the command line (task doc §7.3)."""
    env = os.environ if environ is None else environ
    missing = [name for name in EXPLICIT_ENV_VARS if name not in env]
    if missing:
        raise SystemExit(f"Set these explicitly for eval runs: {', '.join(missing)}.")


async def run_longctx_phase(
    qa_subset: list[dict], system_prompt: str, concurrency: int, scope: str
) -> list[dict]:
    """Phase 1 for --mode longctx: answer every question; no retrieval metrics."""
    sem = asyncio.Semaphore(concurrency)
    total_items = len(qa_subset)

    async def _one(i: int, item: dict) -> dict:
        async with sem:
            logger.info("  [longctx %d/%d] %s %s", i + 1, total_items, item["id"], item["question"][:60])
            return await run_longctx_pipeline(item["question"], system_prompt)

    results = await asyncio.gather(*[_one(i, item) for i, item in enumerate(qa_subset)])
    outcomes = []
    for item, result in zip(qa_subset, results):
        rec = _per_question_record(item, result, scope)
        if item["answerable"]:
            rec.update({"recall_at_k": None, "rr": None, "rr_raw": None, "rr_reranked": None})
        outcomes.append({"item": item, "rag_result": result, "record": rec})
    return outcomes


async def run_qa_then_lift(
    qa_subset: list[dict],
    prompt_template: str,
    citation_mode: str,
    concurrency: int,
    scope: str | None = None,
) -> list[dict]:
    """Phase 1 in two stages that never overlap (§4.4 latency).

    Stage 1 answers every question and records latency. Stage 2 then computes
    reranker lift. compute_reranker_lift always calls the reranker, so running it
    alongside QA would make QA wait on the reranker lock and inflate P95.
    """
    sem = asyncio.Semaphore(concurrency)
    total_items = len(qa_subset)

    async def _qa_one(i: int, item: dict) -> dict:
        async with sem:
            logger.info("  [rag %d/%d] %s %s", i + 1, total_items, item["id"], item["question"][:60])
            return await run_rag_pipeline(item["question"], prompt_template, citation_mode)

    results = await asyncio.gather(*[_qa_one(i, item) for i, item in enumerate(qa_subset)])

    async def _lift_one(item: dict) -> dict | None:
        if not item["answerable"]:
            return None
        async with sem:
            return await asyncio.to_thread(compute_reranker_lift, item["question"])

    logger.info("  QA stage done; computing reranker lift")
    lifts = await asyncio.gather(*[_lift_one(item) for item in qa_subset])

    outcomes = []
    for item, result, lift_data in zip(qa_subset, results, lifts):
        rec = _per_question_record(item, result, scope)
        if item["answerable"]:
            source = item.get("source_doc", "")
            retrieved_sources = [c.source_doc for c in result["contexts"]]
            rec.update(
                {
                    "recall_at_k": compute_recall_at_k(retrieved_sources, source),
                    "rr": compute_reciprocal_rank(retrieved_sources, source),
                    "rr_raw": compute_reciprocal_rank(lift_data["raw_sources"], source),
                    "rr_reranked": compute_reciprocal_rank(lift_data["reranked_sources"], source),
                }
            )
        outcomes.append({"item": item, "rag_result": result, "record": rec})
    return outcomes


async def main(
    run_log_file: Path,
    profile_name: str,
    dataset: str = DEFAULT_DATASET,
    mode: str = "rag",
    longctx_corpus: str | None = None,
    concurrency: int | None = None,
):
    t_start = datetime.now(EVAL_TZ)
    started_at = t_start.isoformat()
    run_ts = t_start.strftime("%Y%m%d_%H%M%S")
    profile = get_eval_profile(profile_name)
    concurrency = concurrency or profile.concurrency
    citation_mode = settings.citation_mode
    logger.info("=== eval_rag run start ===")
    logger.info("Log file: %s", run_log_file.resolve())
    logger.info("cwd=%s python=%s", os.getcwd(), sys.version.split()[0])
    logger.info("MILVUS_HOST=%s", os.environ.get("MILVUS_HOST", "(unset)"))
    logger.info(
        "EVAL_PROFILE=%s dataset=%s mode=%s longctx_corpus=%s concurrency=%d MILVUS_COLLECTION_NAME=%s "
        "ENABLE_RERANKER=%s RERANKER_USE_HEADING=%s CITATION_MODE=%s",
        profile.name, dataset, mode, longctx_corpus, concurrency, settings.milvus_collection_name,
        settings.enable_reranker, settings.reranker_use_heading, citation_mode,
    )
    usage_tracker.reset()

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

    if mode == "rag":
        scope = COLLECTION_SCOPES.get(settings.milvus_collection_name)
        if scope is None:
            issues.append(
                f"MILVUS_COLLECTION_NAME={settings.milvus_collection_name!r} has no corpus scope; "
                f"known: {sorted(COLLECTION_SCOPES)}"
            )
    else:
        scope = longctx_corpus
        if citation_mode != "off":
            issues.append(f"--mode longctx needs CITATION_MODE=off, got {citation_mode!r}")

    prompt_template = ""
    try:
        prompt_template = build_qa_prompt_template(citation_mode)
        import app.services.citations  # noqa: F401 — every mode, off included, post-processes with it
    except (ValueError, ImportError) as e:
        issues.append(f"citation_mode={citation_mode!r} unavailable: {type(e).__name__}: {e}")

    # Long-context system prompt: the same QA prompt, {contexts} = the whole corpus
    longctx_meta: dict[str, Any] | None = None
    longctx_prompt = ""
    if mode == "longctx" and not issues:
        try:
            files = longctx_files(longctx_corpus)
            context = build_longctx_context(files)
            longctx_prompt = prompt_template.replace("{contexts}", context)
            longctx_meta = {
                "corpus": longctx_corpus,
                "files": [{"name": f["name"], "sha256": f["sha256"]} for f in files],
                "file_count": len(files),
                "context_chars": len(context),
                "context_sha256": _sha12(context),
                "context_limit_tokens": DEEPSEEK_CONTEXT_LIMIT,
            }
            preflight = await count_longctx_tokens(longctx_prompt)
            longctx_meta["preflight"] = preflight
            longctx_meta["system_prompt_tokens"] = preflight["input_tokens"]
            logger.info(
                "Long-context prompt: %d files, %d chars, %s tokens (limit %d)",
                len(files), len(context), preflight["input_tokens"], DEEPSEEK_CONTEXT_LIMIT,
            )
            if preflight["input_tokens"] is None:
                issues.append("long-context preflight returned no usage")
            elif preflight["input_tokens"] + QA_MAX_TOKENS > DEEPSEEK_CONTEXT_LIMIT:
                issues.append(
                    f"long-context prompt has {preflight['input_tokens']} tokens, over the "
                    f"{DEEPSEEK_CONTEXT_LIMIT} limit; config not run (§4.2 step 4)"
                )
        except Exception as e:  # e.g. the API rejects an over-long prompt
            issues.append(f"long-context setup failed: {type(e).__name__}: {e}")

    qa_items, dataset_meta = load_eval_dataset(dataset)

    report: dict[str, Any] = {
        "meta": {
            "run_id": f"rag_{run_ts}",
            "run_log_file": str(run_log_file.resolve()),
            "timezone": "America/Los_Angeles",
            "started_at": started_at,
            "finished_at": None,
            "profile": profile.name,
            "mode": mode,
            "concurrency": concurrency,
            "dataset": dataset_meta,
            "output_file": str(output_path.resolve()),
            "status": "invalid" if issues else "running",
            "health_issues": issues,
            "completed_phases": [],
            **build_static_meta(prompt_template, mode, scope),
            "longctx": longctx_meta,
        },
    }

    if issues:
        report["meta"]["usage"] = usage_tracker.snapshot()
        report["meta"]["finished_at"] = datetime.now(EVAL_TZ).isoformat()
        _save_report(report, output_path)
        logger.error("Run marked invalid — aborting: %s. Report: %s", issues, output_path)
        return

    qa_subset, qa_subset_meta = select_eval_subset(qa_items, profile)
    answerable_subset = [i for i in qa_subset if i["answerable"]]
    logger.info(
        "Loaded %d items from %s; selected %d answerable + %d refusal",
        len(qa_items), dataset_meta["name"],
        qa_subset_meta["answerable_selected"], qa_subset_meta["refusal_selected"],
    )
    report["subset"] = qa_subset_meta

    # Intermediate results keyed by question text to avoid mutating dataset items
    rag_results_map: dict[str, dict] = {}
    per_question: dict[str, dict[str, Any]] = {}
    ir_metrics: dict[str, Any] = {}

    # ── Phase 1: answers (+ classic IR metrics in rag mode) ──
    phase1 = "ir" if mode == "rag" else "generate"
    logger.info("=== Phase 1: %s answers (concurrency=%d) ===", mode, concurrency)
    try:
        if mode == "rag":
            outcomes = await run_qa_then_lift(qa_subset, prompt_template, citation_mode, concurrency, scope)
        else:
            outcomes = await run_longctx_phase(qa_subset, longctx_prompt, concurrency, scope)
        for o in outcomes:
            rag_results_map[o["item"]["question"]] = o["rag_result"]
            per_question[o["item"]["id"]] = o["record"]

        answerable_recs = [per_question[i["id"]] for i in answerable_subset]
        latencies = [r["latency_ms"] for r in per_question.values()]
        if mode == "rag":
            mrr_raw = _mean([r["rr_raw"] for r in answerable_recs])
            mrr_reranked = _mean([r["rr_reranked"] for r in answerable_recs])
            retrieval = {
                "recall_at_k": _mean([r["recall_at_k"] for r in answerable_recs]),
                "mrr": _mean([r["rr"] for r in answerable_recs]),
                "mrr_raw": mrr_raw,
                "mrr_reranked": mrr_reranked,
                "reranker_lift": mrr_reranked - mrr_raw,
            }
        else:  # no retrieval: null, not 0 (§4.2 step 6)
            retrieval = dict.fromkeys(("recall_at_k", "mrr", "mrr_raw", "mrr_reranked", "reranker_lift"))
        ir_metrics = {
            "n": len(answerable_recs),
            **retrieval,
            "latency_p50_ms": safe_p50(latencies),
            "latency_p95_ms": safe_p95(latencies),
            "latency_n": len(latencies),
        }
        logger.info("IR Metrics: %s", json.dumps(ir_metrics, indent=2))
        report["ir_metrics"] = ir_metrics
        report["meta"]["completed_phases"].append(phase1)
    except Exception as e:
        logger.exception("Phase 1 (%s) failed: %s", phase1, e)
        report["ir_metrics"] = {"error": f"{type(e).__name__}: {e}"}
    report["per_question"] = list(per_question.values())
    _save_report(report, output_path)

    # ── Phase 2: Refusal judging (all questions) ──
    if per_question:
        logger.info("=== Phase 2: Refusal judging ===")
        try:
            sem = asyncio.Semaphore(concurrency)

            async def _judge_one(rec: dict) -> None:
                async with sem:
                    rec.update(await judge_refusal(rec["question"], rec["clean_answer"]))

            await asyncio.gather(*[_judge_one(rec) for rec in per_question.values()])
            recs = list(per_question.values())
            report["refusal_metrics"] = summarize_refusals(recs)
            if citation_mode != "off":
                report["citation_metrics"] = summarize_citations(recs)
            logger.info("Refusal metrics: %s", json.dumps(report["refusal_metrics"], indent=2))
            report["meta"]["completed_phases"].append("refusal")
        except Exception as e:
            logger.exception("Phase 2 (refusal) failed: %s", e)
            report["refusal_metrics"] = {"error": f"{type(e).__name__}: {e}"}
        report["per_question"] = list(per_question.values())
        _save_report(report, output_path)

    # ── Phase 2b: Answer correctness (answerable questions) ──
    if per_question:
        logger.info("=== Phase 2b: Answer correctness judging ===")
        try:
            sem = asyncio.Semaphore(concurrency)

            async def _correct_one(rec: dict) -> None:
                async with sem:
                    rec.update(await judge_correctness(rec["question"], rec["ground_truth"], rec["clean_answer"]))

            await asyncio.gather(*[_correct_one(r) for r in per_question.values() if r["answerable"]])
            report["correctness_metrics"] = summarize_correctness(list(per_question.values()))
            report["not_found_metrics"] = summarize_not_found(list(per_question.values()))
            logger.info("Correctness metrics: %s", json.dumps(report["correctness_metrics"], indent=2))
            report["meta"]["completed_phases"].append("correctness")
        except Exception as e:
            logger.exception("Phase 2b (correctness) failed: %s", e)
            report["correctness_metrics"] = {"error": f"{type(e).__name__}: {e}"}
        report["per_question"] = list(per_question.values())
        _save_report(report, output_path)

    # ── Phase 3: RAGAS evaluation (rag mode, answerable questions only) ──
    ragas_means: dict[str, Any] = {}
    ragas_metric_status: dict[str, Any] = {}

    if mode == "longctx":
        # No retrieved contexts, so no context or faithfulness metrics: null, not 0
        report["ragas_means"] = {field: None for field in RAGAS_FIELD_NAMES.values()}
        for rec in per_question.values():
            if rec["answerable"]:
                rec.update(dict.fromkeys(RAGAS_FIELD_NAMES.values()))
    elif not rag_results_map:
        logger.warning("Phase 1 produced no results — skipping Phase 3 (RAGAS).")
        report["ragas_means"] = {"skipped": "Phase 1 failed or returned no results"}
    else:
        logger.info("=== Phase 3: RAGAS Evaluation (answerable only) ===")
        try:
            ragas_samples = build_ragas_samples(answerable_subset, rag_results_map)
            ragas_eval = await run_ragas_evaluation(
                ragas_samples,
                judge_model=settings.judge_model,
                judge_api_key=settings.judge_api_key,
                judge_base_url=settings.judge_base_url,
                embeddings=_build_bge_ragas_embeddings(),
                judge_extra_body=build_routing_extra_body(
                    settings.judge_base_url, settings.judge_provider_order
                ),
            )
            ragas_results = ragas_eval["results"]
            ragas_metric_status = ragas_eval["metric_status"]
            ragas_means = compute_ragas_means(ragas_results)
            # Samples follow answerable_subset order (build_ragas_samples skips missing results)
            sample_ids = [i["id"] for i in answerable_subset if i["question"] in rag_results_map]
            for column, field in RAGAS_FIELD_NAMES.items():
                for qid, value in zip(sample_ids, ragas_results.get(column, [])):
                    per_question[qid][field] = value
            logger.info("RAGAS Means: %s", json.dumps(ragas_means, indent=2))
            logger.info("RAGAS Metric Status: %s", json.dumps(ragas_metric_status, ensure_ascii=False, indent=2))
            report["ragas_means"] = ragas_means
            report["ragas_metric_status"] = ragas_metric_status
            report["sample_count"] = len(ragas_samples)
            report["meta"]["completed_phases"].append("ragas")
        except Exception as e:
            logger.exception("Phase 3 (RAGAS) failed: %s", e)
            report["ragas_means"] = {"error": f"{type(e).__name__}: {e}"}

    # ── Final save ──
    report["cost_metrics"] = summarize_costs(
        list(per_question.values()), (longctx_meta or {}).get("preflight")
    )
    usage = usage_tracker.snapshot()
    provider_problems = check_providers(usage)
    report["per_question"] = list(per_question.values())
    report["meta"]["usage"] = usage
    report["meta"]["provider_problems"] = provider_problems
    run_problems = check_run_completeness(report)
    report["meta"]["run_problems"] = run_problems
    report["meta"]["status"] = "invalid" if provider_problems or run_problems else "completed"
    report["meta"]["finished_at"] = datetime.now(EVAL_TZ).isoformat()
    _save_report(report, output_path)
    logger.info("RAG report saved to %s", output_path)

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("RAG EVALUATION SUMMARY  (mode: %s, phases: %s)", mode, report["meta"]["completed_phases"])
    logger.info("=" * 60)
    if ir_metrics and "error" not in ir_metrics:
        if mode == "rag":
            logger.info("  Recall@K:          %.3f  (target >= 0.85)", ir_metrics["recall_at_k"])
            logger.info("  MRR:               %.3f  (target >= 0.70)", ir_metrics["mrr"])
            logger.info("  Reranker Lift:     %.3f  (target > 0)", ir_metrics["reranker_lift"])
        logger.info("  Latency P95:       %.0f ms", ir_metrics["latency_p95_ms"])
    refusal_metrics = report.get("refusal_metrics") or {}
    if refusal_metrics and "error" not in refusal_metrics:
        logger.info(
            "  Refusal accuracy:  %s/%s   False refusals: %s/%s   Unjudged: %s   Excluded: %s",
            refusal_metrics["refusal_correct"], refusal_metrics["refusal_total"],
            refusal_metrics["false_refusals"], refusal_metrics["answerable_total"],
            refusal_metrics["unjudged"], refusal_metrics["refusal_excluded"],
        )
    correctness_metrics = report.get("correctness_metrics") or {}
    if correctness_metrics and "error" not in correctness_metrics:
        logger.info(
            "  Correctness:       %s   labels: %s",
            _fmt_opt(correctness_metrics["correctness"]), correctness_metrics["label_counts"],
        )
    not_found = report.get("not_found_metrics") or {}
    if not_found:
        logger.info(
            "  Not found:         refusal %s/%s   answerable %s/%s",
            not_found["refusal_not_found"], not_found["refusal_total"],
            not_found["answerable_not_found"], not_found["answerable_total"],
        )
    cost = report["cost_metrics"]
    logger.info(
        "  Input tokens/q:    %s   cache hit ratio: %s   cost/q actual $%s, no cache $%s",
        _fmt_opt(cost["input_tokens_mean"], 0), _fmt_opt(cost["cache_hit_ratio"]),
        _fmt_opt(cost["cost_usd_actual_mean"], 6), _fmt_opt(cost["cost_usd_no_cache_mean"], 6),
    )
    if ragas_means and "error" not in ragas_means:
        for k, v in ragas_means.items():
            logger.info("  %-20s %s", k + ":", _fmt_opt(v))
    for metric_name, status in ragas_metric_status.items():
        if status.get("status") == "error":
            logger.warning(
                "  RAGAS %-12s failed: %s: %s",
                metric_name,
                status.get("error_type"),
                status.get("error"),
            )
    for role, u in usage.items():
        logger.info(
            "  Usage %-5s requests=%d prompt=%d completion=%d cost=$%.4f providers=%s",
            role, u["requests"], u["prompt_tokens"], u["completion_tokens"], u["cost_usd"], u["providers"],
        )
    if provider_problems:
        logger.error("  Run INVALID — unexpected providers: %s", provider_problems)
    if run_problems:
        logger.error("  Run INVALID — %s", run_problems)
    logger.info("=" * 60)
    logger.info("=== eval_rag run end; full log: %s ===", run_log_file.resolve())


def _fmt_opt(v: float | None, digits: int = 3) -> str:
    return "null" if v is None else f"{v:.{digits}f}"


if __name__ == "__main__":
    args = parse_args()
    if not args.check_rerank_length:
        require_explicit_env()
        require_citation_mode_env()
    _ts = datetime.now(EVAL_TZ).strftime("%Y%m%d_%H%M%S")
    _log_path = RAG_LOG_DIR / f"eval_rag_{_ts}.log"
    configure_run_logging(_log_path)
    try:
        if args.check_rerank_length:
            asyncio.run(run_check_rerank_length(args.dataset))
        else:
            asyncio.run(
                main(_log_path, args.profile, args.dataset, args.mode, args.longctx_corpus, args.concurrency)
            )
    except Exception:
        logging.exception("eval_rag aborted with exception (see log file above)")
        raise
