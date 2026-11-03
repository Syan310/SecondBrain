"""Tests for eval_rag dataset handling, refusal parsing and meta checks (no external services)."""

import json

import pytest

from evaluation import eval_rag
from evaluation.profiles import get_eval_profile


def _fake_l0_dataset():
    """A few items with the qa_l0_v1 fields: 6 original, 2 term, 3 refusal."""
    items = []
    for i in range(6):
        items.append({
            "id": f"q{i + 1:03d}",
            "question": f"original {i}",
            "ground_truth": f"gt {i}",
            "type": "original",
            "original_type": "factual",
            "answerable": True,
            "source_doc": f"doc-{i % 3}.md",
            "gold_sections": [{"source_doc": f"doc-{i % 3}.md", "heading_path": "H"}],
            "origin": "original_v1",
        })
    for i in range(2):
        items.append({
            "id": f"q{21 + i:03d}",
            "question": f"term {i}",
            "ground_truth": f"gt term {i}",
            "type": "term",
            "answerable": True,
            "source_doc": "doc-0.md",
            "gold_sections": [{"source_doc": "doc-0.md", "heading_path": "H"}],
            "origin": "draft_reviewed",
        })
    for i in range(3):
        items.append({
            "id": f"q{51 + i:03d}",
            "question": f"refusal {i}",
            "ground_truth": "知识库中没有这个问题的答案",
            "type": "refusal",
            "answerable": False,
            "source_doc": "",
            "gold_sections": [],
            "origin": "draft_reviewed",
        })
    return items


def test_legacy_dataset_gets_ids_in_file_order_and_drops_multi_hop():
    raw = [
        {"question": "a", "ground_truth": "x", "source_doc": "d1", "question_type": "factual"},
        {"question": "m", "ground_truth": "y", "source_docs": ["d1", "d2"], "question_type": "multi-hop"},
        {"question": "b", "ground_truth": "z", "source_doc": "d2", "question_type": "comparison"},
    ]
    items = eval_rag.normalize_dataset(raw)
    assert [i["id"] for i in items] == ["q001", "q002"]
    assert [i["question"] for i in items] == ["a", "b"]
    assert all(i["answerable"] and i["type"] == "original" for i in items)
    assert items[1]["original_type"] == "comparison"


def test_load_eval_dataset_records_sha_and_type_counts(tmp_path):
    path = tmp_path / "qa_fake.json"
    path.write_text(json.dumps(_fake_l0_dataset(), ensure_ascii=False), encoding="utf-8")
    items, meta = eval_rag.load_eval_dataset(str(path))
    assert len(items) == 11
    assert meta["answerable"] == 8 and meta["refusal"] == 3
    assert meta["type_counts"] == {"original": 6, "term": 2, "refusal": 3}
    assert len(meta["sha256"]) == 64


def test_smoke_subset_is_4_originals_plus_2_refusals():
    subset, meta = eval_rag.select_eval_subset(_fake_l0_dataset(), get_eval_profile("smoke"))
    answerable = [i for i in subset if i["answerable"]]
    assert len(answerable) == 4 and all(i["type"] == "original" for i in answerable)
    assert len({i["source_doc"] for i in answerable}) == 3  # balanced across docs
    assert [i["id"] for i in subset if not i["answerable"]] == ["q051", "q052"]
    assert meta["answerable_selected"] == 4 and meta["refusal_selected"] == 2


def test_full_subset_takes_everything():
    subset, meta = eval_rag.select_eval_subset(_fake_l0_dataset(), get_eval_profile("full"))
    assert len(subset) == 11
    assert meta["answerable_selected"] == 8 and meta["refusal_selected"] == 3


def test_qa_prompt_off_mode_keeps_existing_rules_and_no_citation_rules():
    template = eval_rag.build_qa_prompt_template("off")
    assert "篇幅控制在 200 字以内" in template
    assert "{citation_rules}" not in template
    assert "{contexts}" in template


def test_qa_prompt_fallback_mode_uses_fallback_rules():
    from app.services.citations import FALLBACK_RULES, REFUSAL_SENTENCE

    template = eval_rag.build_qa_prompt_template("fallback")
    assert FALLBACK_RULES in template
    assert REFUSAL_SENTENCE not in template
    assert "{contexts}" in template


def test_summarize_not_found_counts_scored_refusals_and_answerable():
    recs = [
        {"id": "q1", "answerable": True, "not_found": True, "correctness_label": "correct"},
        {"id": "q2", "answerable": True, "not_found": True, "correctness_label": "incorrect"},
        {"id": "q3", "answerable": True, "not_found": False, "correctness_label": "correct"},
        {"id": "q051", "answerable": False, "not_found": True},
        {"id": "q052", "answerable": False, "not_found": False},
        {"id": "q053", "answerable": False, "not_found": True, "refusal_scored": False},
    ]
    m = eval_rag.summarize_not_found(recs)
    assert (m["refusal_not_found"], m["refusal_total"]) == (1, 2)
    assert (m["answerable_not_found"], m["answerable_total"]) == (2, 3)
    assert m["answerable_not_found_ids"] == ["q1", "q2"]
    assert m["answerable_not_found_correctness"]["correct"] == 1
    assert m["excluded_not_found_ids"] == ["q053"]


def test_per_question_record_flags_not_found():
    item = {"id": "q051", "type": "refusal", "answerable": False, "question": "Q"}
    rag_result = {"raw_answer": "x", "answer": "笔记里没有找到相关内容。根据我的知识……",
                  "citation": None, "contexts": None, "total_ms": 1.0}
    assert eval_rag._per_question_record(item, rag_result)["not_found"] is True
    rag_result["answer"] = "A 是 B。"
    assert eval_rag._per_question_record(item, rag_result)["not_found"] is False


def test_qa_prompt_rejects_unknown_mode():
    with pytest.raises(ValueError):
        eval_rag.build_qa_prompt_template("strict")


@pytest.mark.parametrize(
    "text, label",
    [
        ('{"label": "refusal", "reason": "说没有信息"}', "refusal"),
        ('结果如下：\n```json\n{"label": "Partial", "reason": "r"}\n```', "partial"),
        ('{"label": "answer"}', "answer"),
        ('{"label": "maybe"}', None),
        ("not json", None),
    ],
)
def test_parse_refusal_label(text, label):
    assert eval_rag.parse_refusal_label(text)[0] == label


def test_summarize_refusals_counts_both_sides():
    recs = [
        {"id": "q1", "answerable": True, "refusal_label": "answer"},
        {"id": "q2", "answerable": True, "refusal_label": "refusal"},
        {"id": "q3", "answerable": False, "refusal_label": "refusal"},
        {"id": "q4", "answerable": False, "refusal_label": "partial"},
        {"id": "q5", "answerable": False, "refusal_label": None},
    ]
    m = eval_rag.summarize_refusals(recs)
    assert (m["refusal_correct"], m["refusal_total"]) == (1, 3)
    assert (m["false_refusals"], m["answerable_total"]) == (1, 2)
    assert m["unjudged"] == 1


def test_check_providers_flags_unexpected_provider(monkeypatch):
    monkeypatch.setattr(eval_rag.settings, "llm_base_url", "https://api.deepseek.com")
    monkeypatch.setattr(eval_rag.settings, "judge_base_url", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(eval_rag.settings, "judge_provider_order", ["anthropic"])
    ok = {
        "llm": {"providers": {"api.deepseek.com": 10}},
        "judge": {"providers": {"Anthropic": 30}},
    }
    assert eval_rag.check_providers(ok) == []
    bad = {"judge": {"providers": {"Anthropic": 29, "Google": 1}}}
    problems = eval_rag.check_providers(bad)
    assert len(problems) == 1 and "Google" in problems[0]


# ── Phase 1: QA stage finishes before lift stage (§4.4) ──


def _chunk(source):
    from types import SimpleNamespace

    return SimpleNamespace(source_doc=source, heading_path="h", child_id="c")


@pytest.mark.asyncio
async def test_lift_stage_starts_only_after_all_qa_done(monkeypatch):
    import asyncio

    events: list[str] = []

    async def fake_pipeline(question, prompt_template, citation_mode):
        events.append(f"qa_start:{question}")
        await asyncio.sleep(0.01)
        events.append(f"qa_end:{question}")
        return {"answer": "a", "raw_answer": "a", "citation": None,
                "contexts": [_chunk("doc.md")], "total_ms": 1.0}

    def fake_lift(question):
        events.append(f"lift:{question}")
        return {"raw_sources": ["doc.md"], "reranked_sources": ["doc.md"]}

    monkeypatch.setattr(eval_rag, "run_rag_pipeline", fake_pipeline)
    monkeypatch.setattr(eval_rag, "compute_reranker_lift", fake_lift)
    items = [
        {"id": f"q{i}", "type": "original", "answerable": i != 2, "question": f"Q{i}", "source_doc": "doc.md"}
        for i in range(4)
    ]

    outcomes = await eval_rag.run_qa_then_lift(items, "tmpl", "off", concurrency=2)

    first_lift = min(i for i, e in enumerate(events) if e.startswith("lift:"))
    last_qa_end = max(i for i, e in enumerate(events) if e.startswith("qa_end:"))
    assert last_qa_end < first_lift
    assert sum(e.startswith("lift:") for e in events) == 3  # refusal question q2 has no lift
    assert [o["item"]["id"] for o in outcomes] == ["q0", "q1", "q2", "q3"]
    assert outcomes[0]["record"]["rr_reranked"] == 1.0
    assert "rr" not in outcomes[2]["record"]


# ── CITATION_MODE must be set explicitly (§6.3 step 2) ──


def test_citation_mode_env_missing_exits():
    with pytest.raises(SystemExit, match="CITATION_MODE is not set"):
        eval_rag.require_citation_mode_env({})


def test_citation_mode_env_invalid_exits():
    with pytest.raises(SystemExit, match="invalid"):
        eval_rag.require_citation_mode_env({"CITATION_MODE": "on"})


@pytest.mark.parametrize("mode", ["off", "prompt", "filter"])
def test_citation_mode_env_accepts_valid_modes(mode):
    assert eval_rag.require_citation_mode_env({"CITATION_MODE": mode}) == mode


# ── off mode strips [n] markers before RAGAS / refusal judging (§5.4 item 3) ──


def test_off_mode_uses_prompt_cleanup_and_no_citation_stats(monkeypatch):
    import re
    import sys
    from types import ModuleType, SimpleNamespace

    calls = []

    def fake_apply_citations(answer, valid_ids, mode):
        calls.append((valid_ids, mode))
        return SimpleNamespace(clean_answer=re.sub(r"\[\d+\]", "", answer))

    fake = ModuleType("app.services.citations")
    fake.apply_citations = fake_apply_citations
    monkeypatch.setitem(sys.modules, "app.services.citations", fake)

    post = eval_rag.apply_citation_mode("A 是 B[1]。", 5, "off")

    assert post == {"clean_answer": "A 是 B。", "citation": None}
    assert calls == [({1, 2, 3, 4, 5}, "prompt")]


def test_off_mode_strips_markers_with_real_citations_module():
    pytest.importorskip("app.services.citations")  # lands with feat/citation-b at P5
    post = eval_rag.apply_citation_mode("A 是 B[1]。", 5, "off")
    assert post == {"clean_answer": "A 是 B。", "citation": None}


# ── Dataset SHA-256 ignores CRLF checkouts ──


def test_dataset_sha_is_same_for_crlf_and_lf(tmp_path):
    lf, crlf = tmp_path / "lf.json", tmp_path / "crlf.json"
    lf.write_bytes(b'{\n  "a": 1\n}\n')
    crlf.write_bytes(b'{\r\n  "a": 1\r\n}\r\n')
    assert eval_rag._sha256_text_lf(crlf) == eval_rag._sha256_text_lf(lf) == eval_rag._sha256_file(lf)


def test_frozen_dataset_sha_matches_git_blob():
    _, meta = eval_rag.load_eval_dataset("qa_l0_v1.json")
    assert meta["sha256"] == "ee47c4ecc71352f628a4ca46fe3d73d6f12be187aefc70bdc871d6fcf6c2d0db"


# ── A run with a failed phase or RAGAS metric is invalid ──


def _report(phases, metric_status=None):
    return {"meta": {"completed_phases": phases}, "ragas_metric_status": metric_status or {}}


def test_run_completeness_ok_when_all_phases_and_metrics_ok():
    report = _report(["ir", "refusal", "correctness", "ragas"], {"faithfulness": {"status": "ok"}})
    assert eval_rag.check_run_completeness(report) == []


def test_run_completeness_flags_ragas_metric_error():
    report = _report(
        ["ir", "refusal", "correctness", "ragas"],
        {"faithfulness": {"status": "ok"}, "context_recall": {"status": "error", "error_type": "APIStatusError"}},
    )
    assert eval_rag.check_run_completeness(report) == ["RAGAS context_recall failed: APIStatusError"]


def test_run_completeness_flags_missing_phase():
    assert eval_rag.check_run_completeness(_report(["ir"])) == [
        "phase 'refusal' did not complete",
        "phase 'correctness' did not complete",
        "phase 'ragas' did not complete",
    ]


def test_run_completeness_longctx_needs_no_ir_or_ragas():
    report = _report(["generate", "refusal", "correctness"])
    report["meta"]["mode"] = "longctx"
    assert eval_rag.check_run_completeness(report) == []


def test_run_completeness_flags_unjudged_correctness():
    report = _report(["ir", "refusal", "correctness", "ragas"])
    report["correctness_metrics"] = {"unjudged": 2}
    assert eval_rag.check_run_completeness(report) == ["correctness: 2 answer(s) unjudged"]


# ── Answer correctness (§5.2) ──


@pytest.mark.parametrize(
    "text,label",
    [
        ('{"label": "correct", "reason": "ok"}', "correct"),
        ('结果：{"label": "PARTIAL", "reason": "少一点"}', "partial"),
        ('{"label": "incorrect"}', "incorrect"),
        ('{"label": "refusal"}', None),
        ("not json", None),
    ],
)
def test_parse_correctness_label(text, label):
    assert eval_rag.parse_correctness_label(text)[0] == label


@pytest.mark.asyncio
async def test_judge_correctness_retries_once_then_unjudged(monkeypatch):
    calls = []

    async def fake_judge(messages, max_completion_tokens=None):
        calls.append(messages[0]["content"])
        return "no json here"

    monkeypatch.setattr(eval_rag.judge_client, "judge", fake_judge)
    result = await eval_rag.judge_correctness("Q", "GT", "A")
    assert len(calls) == 2
    assert result["correctness_label"] == "unjudged"
    assert result["correctness_score"] is None
    # the judge sees question, reference answer and answer, but no retrieved contexts
    assert "GT" in calls[0] and "A" in calls[0] and "检索" not in calls[0]


@pytest.mark.asyncio
async def test_judge_correctness_scores(monkeypatch):
    async def fake_judge(messages, max_completion_tokens=None):
        return '{"label": "partial", "reason": "r"}'

    monkeypatch.setattr(eval_rag.judge_client, "judge", fake_judge)
    result = await eval_rag.judge_correctness("Q", "GT", "A")
    assert (result["correctness_label"], result["correctness_score"]) == ("partial", 0.5)


def test_summarize_correctness_ignores_refusal_questions():
    recs = [
        {"answerable": True, "correctness_label": "correct", "correctness_score": 1.0},
        {"answerable": True, "correctness_label": "partial", "correctness_score": 0.5},
        {"answerable": True, "correctness_label": "unjudged", "correctness_score": None},
        {"answerable": False},
    ]
    summary = eval_rag.summarize_correctness(recs)
    assert summary["n"] == 3
    assert summary["judged"] == 2
    assert summary["unjudged"] == 1
    assert summary["correctness"] == pytest.approx(0.75)
    assert summary["label_counts"] == {"correct": 1, "partial": 1, "incorrect": 0, "unjudged": 1}


# ── Refusal questions answered by L1 (§5.3) ──


def test_refusal_scored_only_drops_l1_answerable_on_l0_l1():
    q053 = {"id": "q053", "answerable": False}
    q051 = {"id": "q051", "answerable": False}
    assert eval_rag.refusal_scored(q053, "l0") is True
    assert eval_rag.refusal_scored(q053, "l0_l1") is False
    assert eval_rag.refusal_scored(q051, "l0_l1") is True
    assert eval_rag.refusal_scored({"id": "q001", "answerable": True}, "l0_l1") is True


def test_summarize_refusals_skips_unscored_refusal_questions():
    recs = [
        {"id": "q051", "answerable": False, "refusal_scored": True, "refusal_label": "refusal"},
        {"id": "q053", "answerable": False, "refusal_scored": False, "refusal_label": "answer"},
        {"id": "q001", "answerable": True, "refusal_scored": True, "refusal_label": "answer"},
    ]
    summary = eval_rag.summarize_refusals(recs)
    assert (summary["refusal_correct"], summary["refusal_total"]) == (1, 1)
    assert summary["refusal_excluded"] == ["q053"]


# ── Cost (§4.3) ──


def test_deepseek_peak_hours():
    from datetime import datetime, timezone

    monday = lambda h: datetime(2026, 11, 2, h, 30, tzinfo=timezone.utc)  # noqa: E731
    assert eval_rag.is_deepseek_peak(monday(1)) is True
    assert eval_rag.is_deepseek_peak(monday(4)) is False
    assert eval_rag.is_deepseek_peak(monday(9)) is True
    assert eval_rag.is_deepseek_peak(monday(10)) is False
    saturday = datetime(2026, 11, 7, 2, 0, tzinfo=timezone.utc)
    assert eval_rag.is_deepseek_peak(saturday) is False


def test_llm_cost_actual_and_no_cache():
    from datetime import datetime, timezone

    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000,
             "prompt_cache_hit_tokens": 800_000, "prompt_cache_miss_tokens": 200_000}
    off_peak = datetime(2026, 11, 2, 20, 0, tzinfo=timezone.utc)
    cost = eval_rag.llm_cost(usage, off_peak)
    assert cost["price_period"] == "off_peak"
    assert cost["cost_usd_actual"] == pytest.approx(0.8 * 0.003 + 0.2 * 0.15 + 0.6)
    assert cost["cost_usd_no_cache"] == pytest.approx(0.15 + 0.6)
    assert eval_rag.llm_cost(None, off_peak)["cost_usd_actual"] is None


def test_llm_cost_without_cache_fields_counts_all_as_miss():
    from datetime import datetime, timezone

    peak = datetime(2026, 11, 2, 2, 0, tzinfo=timezone.utc)
    cost = eval_rag.llm_cost({"prompt_tokens": 100, "completion_tokens": 0}, peak)
    assert (cost["cache_hit_tokens"], cost["cache_miss_tokens"]) == (0, 100)
    assert cost["cost_usd_actual"] == pytest.approx(cost["cost_usd_no_cache"])


def test_summarize_costs_means_and_totals():
    recs = [
        {"input_tokens": 100, "output_tokens": 10, "cache_hit_tokens": 50, "cost_usd_actual": 0.1,
         "cost_usd_no_cache": 0.2, "price_period": "peak"},
        {"input_tokens": 300, "output_tokens": 30, "cache_hit_tokens": 150, "cost_usd_actual": 0.3,
         "cost_usd_no_cache": 0.4, "price_period": "off_peak"},
        {"input_tokens": None},
    ]
    s = eval_rag.summarize_costs(recs, {"cost_usd_actual": 0.05, "cost_usd_no_cache": 0.06})
    assert (s["n"], s["unpriced"]) == (2, 1)
    assert s["input_tokens_mean"] == 200
    assert s["cache_hit_ratio"] == pytest.approx(0.5)
    assert s["cost_usd_actual_total"] == pytest.approx(0.4)
    assert s["cost_usd_no_cache_mean"] == pytest.approx(0.3)
    assert s["price_periods"] == {"peak": 1, "off_peak": 1}
    assert s["preflight_cost_usd_actual"] == 0.05


# ── Long-context mode (§4.2) ──


def test_parse_ingest_record_reads_file_table():
    sha = "a" * 64
    text = "\n".join([
        "| 层 | 名字 | 字节 | SHA-256 | 父块 | 子块 | 概念 |",
        "|---|---|---|---|---|---|---|",
        f"| L0 | `RAG复习.md` | 1,234 | `{sha}` | 3 | 8 | 18 |",
        f"| L1 | `L1笔记__a__b.md` | 99 | `{'b' * 64}` | 1 | 2 | 0 |",
        "| 文件 | 解析失败的父块数 |",
        "| `L1笔记__a__b.md` | 1 |",
    ])
    assert eval_rag.parse_ingest_record(text) == [("RAG复习.md", sha), ("L1笔记__a__b.md", "b" * 64)]


def _write_corpus(tmp_path, monkeypatch, files):
    """files: {name: text}. Writes notes, a manifest and an ingest record."""
    import hashlib

    manifest = {"included": []}
    rows = []
    for name, text in files.items():
        path = tmp_path / ("src_" + name)
        path.write_bytes(text.encode("utf-8"))
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        manifest["included"].append({"path": str(path), "sha256": sha})
        rows.append(f"| L1 | `{name}` | {len(text)} | `{sha}` | 1 | 1 | 0 |")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "record.md").write_text("\n".join(rows), encoding="utf-8")
    monkeypatch.setattr(eval_rag, "CORPUS_MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(eval_rag, "INGEST_RECORD_L1", tmp_path / "record.md")


def test_longctx_files_sorted_and_context_has_source_lines(tmp_path, monkeypatch):
    _write_corpus(tmp_path, monkeypatch, {"b.md": "第二\r\n行", "a.md": "第一\n"})
    files = eval_rag.longctx_files("l0_l1")
    assert [f["name"] for f in files] == ["a.md", "b.md"]
    assert eval_rag.build_longctx_context(files) == "[来源: a.md]\n第一\n\n[来源: b.md]\n第二\n行"


def test_longctx_files_refuses_changed_file(tmp_path, monkeypatch):
    _write_corpus(tmp_path, monkeypatch, {"a.md": "原文"})
    (tmp_path / "src_a.md").write_text("改过", encoding="utf-8")
    with pytest.raises(ValueError, match="changed since ingestion"):
        eval_rag.longctx_files("l0_l1")


@pytest.mark.asyncio
async def test_longctx_phase_records_null_retrieval_metrics(monkeypatch):
    seen = []

    async def fake_pipeline(question, system_prompt):
        seen.append(system_prompt)
        return {"answer": "a", "raw_answer": "a", "citation": None, "contexts": None,
                "context_texts": None, "total_ms": 5.0, "llm_started_at": "2026-11-02T20:00:00+00:00",
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "prompt_cache_hit_tokens": 8,
                          "prompt_cache_miss_tokens": 2}}

    monkeypatch.setattr(eval_rag, "run_longctx_pipeline", fake_pipeline)
    items = [
        {"id": "q001", "type": "original", "answerable": True, "question": "Q1", "source_doc": "d.md"},
        {"id": "q053", "type": "refusal", "answerable": False, "question": "Q2", "source_doc": ""},
    ]
    outcomes = await eval_rag.run_longctx_phase(items, "SYS", concurrency=2, scope="l0_l1")
    rec, refusal = outcomes[0]["record"], outcomes[1]["record"]
    assert seen == ["SYS", "SYS"]
    assert rec["recall_at_k"] is None and rec["rr"] is None
    assert rec["retrieved"] is None
    assert (rec["input_tokens"], rec["cache_hit_tokens"]) == (10, 8)
    assert refusal["refusal_scored"] is False


# ── Explicit retrieval switches (§7.3 step 2) ──


def test_require_explicit_env_lists_missing():
    with pytest.raises(SystemExit, match="MILVUS_COLLECTION_NAME, RERANKER_USE_HEADING"):
        eval_rag.require_explicit_env({"ENABLE_RERANKER": "false", "CITATION_MODE": "off"})
    eval_rag.require_explicit_env(
        {"MILVUS_COLLECTION_NAME": "x", "ENABLE_RERANKER": "false",
         "RERANKER_USE_HEADING": "false", "CITATION_MODE": "off"}
    )
