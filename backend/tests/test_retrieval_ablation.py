"""Tests for the retrieval ablation metrics and self-check (no Milvus, no models)."""

import pytest

from evaluation import retrieval_ablation as ra


def _chunk(doc, path, cid="c"):
    return {"source_doc": doc, "heading_path": path, "child_id": cid, "score": 0.0}


GOLD = [{"source_doc": "a.md", "heading_path": "3. PE"}]


def test_in_section_matches_the_heading_and_its_sub_headings_only():
    assert ra.in_section("a.md", "3. PE", GOLD[0])
    assert ra.in_section("a.md", "3. PE/why sin", GOLD[0])
    assert not ra.in_section("a.md", "3. PE2", GOLD[0])  # prefix but not a sub-heading
    assert not ra.in_section("b.md", "3. PE", GOLD[0])
    assert not ra.in_section("a.md", "", GOLD[0])


def test_section_metrics_rank_recall_and_hit_at_1():
    retrieved = [_chunk("a.md", "2. Emb"), _chunk("a.md", "3. PE/key"), _chunk("b.md", "3. PE")]
    m = ra.section_metrics(retrieved, GOLD)
    assert m == {"section_recall_at_k": 1.0, "section_rr": 0.5, "section_hit_at_1": 0.0}


def test_section_recall_is_the_share_of_gold_sections_covered():
    gold = GOLD + [{"source_doc": "b.md", "heading_path": "X"}]
    m = ra.section_metrics([_chunk("a.md", "3. PE")], gold)
    assert m["section_recall_at_k"] == 0.5
    assert m["section_rr"] == 1.0
    assert m["section_hit_at_1"] == 1.0


def test_section_metrics_are_none_without_gold_and_zero_on_miss():
    assert ra.section_metrics([_chunk("a.md", "3. PE")], [])["section_rr"] is None
    assert ra.section_metrics([], GOLD) == {
        "section_recall_at_k": 0.0, "section_rr": 0.0, "section_hit_at_1": 0.0,
    }


def test_score_question_uses_eval_rag_document_metrics():
    item = {"source_doc": "a.md", "gold_sections": GOLD}
    s = ra.score_question([_chunk("b.md", "X"), _chunk("a.md", "3. PE")], item)
    assert s["recall_at_k"] == 1.0
    assert s["rr"] == 0.5


def _records(values_by_mode):
    """values_by_mode: {mode: [rr per question]} -> records with every metric = that value."""
    n = len(next(iter(values_by_mode.values())))
    return [
        {
            "id": f"q{i:03d}",
            "modes": {mode: {m: vals[i] for m in ra.METRICS} for mode, vals in values_by_mode.items()},
        }
        for i in range(n)
    ]


def test_summarize_and_compare_modes():
    records = _records({"dense": [0.0, 0.5, 1.0, 0.5], "bm25": [1.0] * 4, "hybrid": [1.0] * 4})
    summary = ra.summarize_mode(records, "dense")
    assert summary["mrr"] == pytest.approx(0.5)
    assert summary["recall_at_k_n"] == 4

    comp = ra.compare_modes(records)
    assert set(comp) == {"hybrid_vs_dense", "hybrid_vs_bm25"}
    vs_dense = comp["hybrid_vs_dense"]["rr"]
    assert vs_dense["diff"] == pytest.approx(0.5)  # hybrid minus dense
    assert vs_dense["better"] == 3 and vs_dense["same"] == 1
    assert comp["hybrid_vs_bm25"]["rr"]["verdict"] == "flat"


def test_summarize_skips_questions_without_section_gold():
    records = _records({"dense": [1.0, 0.0], "bm25": [1.0, 0.0], "hybrid": [1.0, 0.0]})
    records[1]["modes"]["hybrid"]["section_rr"] = None
    s = ra.summarize_mode(records, "hybrid")
    assert s["section_rr"] == 1.0 and s["section_rr_n"] == 1


def test_check_hybrid_rounds_to_three_places():
    ok = {"hybrid": {"recall_at_k": 0.88, "mrr": 0.8156666666666667}}
    assert ra.check_hybrid(ok, "secondbrain_l1") == []
    bad = {"hybrid": {"recall_at_k": 0.96, "mrr": 0.9206666666666667}}
    problems = ra.check_hybrid(bad, "secondbrain_docs")
    assert len(problems) == 1 and "mrr" in problems[0]
    assert ra.check_hybrid(ok, "other") != []


def test_compare_with_reference_reports_mismatched_top5():
    records = [
        {"id": "q001", "retrieved": {"hybrid": [_chunk("a", "", "x"), _chunk("a", "", "y")]}},
        {"id": "q002", "retrieved": {"hybrid": [_chunk("a", "", "z")]}},
    ]
    reference = {"per_question": [
        {"id": "q001", "answerable": True, "retrieved": [{"child_id": "x"}, {"child_id": "y"}]},
        {"id": "q002", "answerable": True, "retrieved": [{"child_id": "w"}]},
        {"id": "q051", "answerable": False, "retrieved": [{"child_id": "x"}]},
    ]}
    out = ra.compare_with_reference(records, reference)
    assert out == {"compared": 2, "same_top5": 1, "mismatched_ids": ["q002"]}


def test_retrieve_rejects_unknown_mode():
    with pytest.raises(ValueError):
        ra.retrieve("sparse", "q", [0.0])
