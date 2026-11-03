import json

import pytest

from evaluation import stats


def _rec(qid, *, type_="original", answerable=True, faith=None, rr=1.0, rr_raw=1.0,
         rr_reranked=1.0, label="answer", fully_refused=False):
    return {
        "id": qid,
        "type": type_,
        "answerable": answerable,
        "origin": "original_v1" if type_ == "original" else "draft_reviewed",
        "faithfulness": faith,
        "rr": rr,
        "rr_raw": rr_raw,
        "rr_reranked": rr_reranked,
        "refusal_label": label,
        "citation": {"fully_refused": fully_refused},
    }


def _run(records, path="run.json", p95=100.0):
    return {"_path": path, "per_question": records,
            "ir_metrics": {"latency_p50_ms": p95 / 2, "latency_p95_ms": p95}}


def test_per_question_means_average_over_runs_and_skip_missing():
    runs = [
        _run([_rec("q1", faith=0.5), _rec("q2", faith=None)]),
        _run([_rec("q1", faith=1.0), _rec("q2", faith=0.4)]),
    ]
    assert stats.per_question_means(runs, "faithfulness") == {"q1": 0.75, "q2": 0.4}


def test_reranker_lift_is_derived_per_question():
    runs = [_run([_rec("q1", rr_raw=0.5, rr_reranked=1.0)])]
    assert stats.per_question_means(runs, "reranker_lift") == {"q1": 0.5}


def test_paired_compare_counts_and_threshold():
    a = {"q1": 0.5, "q2": 0.5, "q3": 0.5}
    b = {"q1": 0.6, "q2": 0.52, "q3": 0.3}
    result = stats.paired_compare(a, b, {"q1", "q2", "q3"})
    assert (result["better"], result["same"], result["worse"]) == (1, 1, 1)
    assert result["n"] == 3
    assert result["diff"] == pytest.approx((0.1 + 0.02 - 0.2) / 3)


def test_bootstrap_is_deterministic_and_verdict_follows_ci():
    diffs = [0.1, 0.2, 0.15, 0.05, 0.12]
    assert stats.bootstrap_ci(diffs) == stats.bootstrap_ci(diffs)
    assert stats.verdict(stats.bootstrap_ci(diffs)) == "improved"
    assert stats.verdict(stats.bootstrap_ci([-d for d in diffs])) == "declined"
    assert stats.verdict(stats.bootstrap_ci([0.1, -0.1, 0.05, -0.05])) == "flat"


def test_refusal_counts():
    run = _run([
        _rec("q1", label="answer"),
        _rec("q2", label="refusal"),
        _rec("q3", type_="refusal", answerable=False, label="refusal", fully_refused=True),
        _rec("q4", type_="refusal", answerable=False, label="partial"),
    ])
    assert stats.refusal_counts(run) == {
        "refusal_correct": 1,
        "refusal_total": 2,
        "false_refusals": 1,
        "answerable_total": 2,
        "unjudged": 0,
        "excluded": [],
        "fully_refused": 1,
    }


def test_refusal_counts_skip_unscored_refusal_questions():
    q053 = _rec("q053", type_="refusal", answerable=False, label="answer")
    q053["refusal_scored"] = False
    run = _run([_rec("q051", type_="refusal", answerable=False, label="refusal"), q053])
    counts = stats.refusal_counts(run)
    assert (counts["refusal_correct"], counts["refusal_total"]) == (1, 1)
    assert counts["excluded"] == ["q053"]


def test_correctness_score_is_a_paired_metric():
    def rec(qid, score):
        r = _rec(qid)
        r["correctness_score"] = score
        return r

    runs_a = [_run([rec("q1", 1.0), rec("q2", 0.5)])]
    runs_b = [_run([rec("q1", 0.5), rec("q2", 0.5)])]
    report = stats.compare_configs("R0", runs_a, "R1", runs_b)
    paired = report["metrics"]["correctness_score"]["answerable"]["paired"]
    assert paired["n"] == 2
    assert paired["diff"] == pytest.approx(-0.25)


def test_usage_summary_means_per_run_without_ci():
    def rec(qid, tokens, cost):
        r = _rec(qid)
        r.update({"input_tokens": tokens, "output_tokens": 1, "cost_usd_actual": cost, "cost_usd_no_cache": cost})
        return r

    runs = [_run([rec("q1", 100, 0.1), rec("q2", 300, 0.3)]), _run([rec("q1", 200, 0.2), rec("q2", 200, 0.2)])]
    summary = stats.usage_summary(runs)
    assert summary["input_tokens"] == {"mean": 200.0, "min": 200.0, "max": 200.0, "runs": 2}
    assert summary["cost_usd_actual"]["mean"] == pytest.approx(0.2)
    assert "ci95" not in summary["input_tokens"]


def test_rag_vs_longctx_skips_retrieval_metrics_in_markdown():
    rag = _rec("q1")
    rag["correctness_score"] = 1.0
    lc = _rec("q1", rr=None, rr_raw=None, rr_reranked=None)
    lc["correctness_score"] = 1.0
    report = stats.compare_configs("R0", [_run([rag])], "LC0", [_run([lc])])
    md = stats.to_markdown(report)
    assert "| correctness_score |" in md
    assert "| rr | 1.000" not in md  # main table: LC0 has no rr, so the row is skipped


def test_compare_configs_subsets_and_markdown(tmp_path):
    def records(shift):
        return [
            _rec("q1", faith=0.5 + shift),
            _rec("q2", faith=0.6 + shift),
            _rec("q3", type_="term", faith=0.7 + shift),
            _rec("q4", type_="refusal", answerable=False, faith=None, label="refusal"),
        ]

    paths_a, paths_b = [], []
    for i in range(3):
        for shift, paths, tag in ((0.0, paths_a, "a"), (0.2, paths_b, "b")):
            path = tmp_path / f"{tag}{i}.json"
            path.write_text(json.dumps({**_run(records(shift)), "meta": _meta()}), encoding="utf-8")
            paths.append(str(path))

    out = tmp_path / "cmp.json"
    stats.main(["--a", "B0", *paths_a, "--b", "B2", *paths_b, "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))

    assert report["subset_sizes"]["original_20"] == 2
    assert report["subset_sizes"]["answerable"] == 3
    assert report["subset_sizes"]["type_term"] == 1
    faith = report["metrics"]["faithfulness"]["answerable"]["paired"]
    assert faith["diff"] == pytest.approx(0.2)
    assert faith["verdict"] == "improved"
    assert "ci95" not in report["metrics"]["faithfulness"]["type_term"]["paired"]
    assert report["refusal"]["b"]["refusal_accuracy"]["mean"] == 1.0
    md = stats.to_markdown(stats.compare_configs("B0", stats.load_runs(paths_a),
                                                 "B2", stats.load_runs(paths_b)))
    assert "| faithfulness |" in md


def _meta(status="completed", sha="abc", dirty=False, ds="d1", judge="anthropic/claude-haiku-5.5"):
    return {"status": status, "commit": {"sha": sha, "dirty": dirty}, "dataset": {"sha256": ds},
            "models": {"judge": {"model": judge}}}


def test_check_run_meta_passes_when_consistent():
    runs = [{"_path": f"r{i}.json", "meta": _meta()} for i in range(3)]
    assert stats.check_run_meta(runs) == []


def test_check_run_meta_flags_each_problem():
    runs = [
        {"_path": "ok.json", "meta": _meta()},
        {"_path": "bad.json", "meta": _meta(status="invalid", sha="def", dirty=True, ds="d2")},
        {"_path": "nometa.json"},
    ]
    problems = stats.check_run_meta(runs)
    joined = "\n".join(problems)
    assert "bad.json: status='invalid'" in joined
    assert "bad.json: dirty=True" in joined
    assert "nometa.json: dirty=None" in joined
    assert "commit differs" in joined
    assert "dataset sha256 differs" in joined


def test_main_refuses_inconsistent_meta(tmp_path):
    paths = []
    for i, sha in enumerate(("abc", "def")):
        path = tmp_path / f"r{i}.json"
        path.write_text(json.dumps({**_run([_rec("q1", faith=0.5)]), "meta": _meta(sha=sha)}), encoding="utf-8")
        paths.append(str(path))
    with pytest.raises(SystemExit, match="commit differs"):
        stats.main(["--a", "A", paths[0], "--b", "B", paths[1]])


def test_check_run_meta_flags_judge_model_mismatch():
    runs = [
        {"_path": "haiku.json", "meta": _meta()},
        {"_path": "sonnet.json", "meta": _meta(judge="anthropic/claude-sonnet-5.5")},
    ]
    assert stats.check_run_meta(runs) == [
        "judge model differs across runs: ['anthropic/claude-haiku-5.5', 'anthropic/claude-sonnet-5.5']"
    ]


def test_main_refuses_mixed_judge_models(tmp_path):
    paths = []
    for i, judge in enumerate(("anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5")):
        path = tmp_path / f"j{i}.json"
        path.write_text(json.dumps({**_run([_rec("q1", faith=0.5)]), "meta": _meta(judge=judge)}), encoding="utf-8")
        paths.append(str(path))
    with pytest.raises(SystemExit, match="judge model differs"):
        stats.main(["--a", "A", paths[0], "--b", "B", paths[1]])


def test_found_subset_drops_questions_flagged_in_any_run():
    def rec(qid, faith, not_found=False, label="answer"):
        r = _rec(qid, faith=faith, label=label)
        r["not_found"] = not_found
        return r

    runs_a = [_run([rec("q1", 0.5), rec("q2", 0.5, label="refusal"), rec("q3", 0.5)])]
    runs_b = [_run([rec("q1", 0.9), rec("q2", 0.9), rec("q3", 0.9, not_found=True)])]
    report = stats.compare_configs("F-filter", runs_a, "F-new", runs_b)
    assert report["subset_sizes"]["answerable_found"] == 1
    assert report["metrics"]["faithfulness"]["answerable_found"]["paired"]["n"] == 1
    assert report["not_found"]["b"][0]["answerable_not_found"] == 1
    md = stats.to_markdown(report)
    assert "### answerable_found (n=1)" in md
    assert "| answerable_not_found (per run) | 0/3 | 1/3 |" in md
