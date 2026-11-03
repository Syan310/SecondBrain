import json

import pytest

from evaluation import calibrate_judge as cj


def _run(path, judge, scores, labels):
    """scores: {qid: faithfulness}; other RAGAS metrics copy it."""
    recs = []
    for qid, label in labels.items():
        s = scores.get(qid)
        recs.append({
            "id": qid, "type": "original", "answerable": s is not None,
            "faithfulness": s, "answer_relevancy": s, "context_precision": s, "context_recall": s,
            "refusal_label": label,
        })
    return {"_path": path, "meta": {"models": {"judge": {"model": judge}}}, "per_question": recs}


def test_spearman_matches_known_values():
    assert cj.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert cj.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # ties use average ranks: ranks x = [1.5, 1.5, 3], y = [1, 2, 3] -> rho = sqrt(3)/2
    assert cj.spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(0.8660254)


def test_spearman_undefined_cases():
    assert cj.spearman([1, 2], [1, 2]) is None
    assert cj.spearman([1, 1, 1], [1, 2, 3]) is None


def test_metric_uses_haiku_mean_and_flags_low_correlation():
    labels = {"q1": "answer", "q2": "answer", "q3": "answer"}
    sonnet = _run("s", "sonnet", {"q1": 0.9, "q2": 0.5, "q3": 0.1}, labels)
    haiku = [
        _run("h1", "haiku", {"q1": 0.8, "q2": 0.4, "q3": 0.0}, labels),
        _run("h2", "haiku", {"q1": 1.0, "q2": 0.6, "q3": 0.2}, labels),
    ]
    r = cj.compare_metric(sonnet, haiku, "faithfulness")
    assert r["n"] == 3
    assert r["spearman"] == pytest.approx(1.0)
    assert r["mean_diff"] == pytest.approx(0.0)
    assert r["low_correlation"] is False

    reversed_sonnet = _run("s", "sonnet", {"q1": 0.1, "q2": 0.5, "q3": 0.9}, labels)
    assert cj.compare_metric(reversed_sonnet, haiku, "faithfulness")["low_correlation"] is True


def test_refusal_agreement_majority_and_per_run():
    sonnet = _run("s", "sonnet", {}, {"q1": "refusal", "q2": "answer", "q3": "partial"})
    haiku = [
        _run("h1", "haiku", {}, {"q1": "refusal", "q2": "answer", "q3": "answer"}),
        _run("h2", "haiku", {}, {"q1": "refusal", "q2": "partial", "q3": "refusal"}),
        _run("h3", "haiku", {}, {"q1": "answer", "q2": "answer", "q3": "partial"}),
    ]
    r = cj.compare_labels(sonnet, haiku)
    assert r["n"] == 3
    # q1 majority refusal = agree; q2 majority answer = agree; q3 no majority = disagree
    assert r["majority_agree"] == 2
    assert r["per_run_agreement"] == pytest.approx((2 + 2 + 1) / 9)
    assert [d["id"] for d in r["disagreements"]] == ["q3"]


def test_correctness_agreement_skips_unjudged_and_scores_spearman():
    def run(path, judge, labels):
        scores = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
        recs = [
            {"id": qid, "answerable": True, "correctness_label": label, "correctness_score": scores.get(label)}
            for qid, label in labels.items()
        ]
        return {"_path": path, "meta": {"models": {"judge": {"model": judge}}}, "per_question": recs}

    sonnet = run("s", "sonnet", {"q1": "correct", "q2": "partial", "q3": "incorrect", "q4": "unjudged"})
    haiku = [run(f"h{i}", "haiku", {"q1": "correct", "q2": "incorrect", "q3": "incorrect", "q4": "correct"})
             for i in range(3)]
    result = cj.calibrate(sonnet, haiku)
    agree = result["correctness"]
    assert agree["n"] == 3  # q4 unjudged by Sonnet
    assert agree["majority_agree"] == 2
    assert [d["id"] for d in agree["disagreements"]] == ["q2"]
    assert result["metrics"]["correctness_score"]["n"] == 3
    assert "回答正确率标签一致率" in cj.to_markdown(result)


def test_main_writes_json_and_markdown(tmp_path, capsys):
    labels = {"q1": "answer", "q2": "answer", "q3": "refusal"}
    scores = {"q1": 0.9, "q2": 0.4}
    paths = []
    for name, judge in (("s", "anthropic/claude-sonnet-5.5"), ("h1", "anthropic/claude-haiku-5.5"),
                        ("h2", "anthropic/claude-haiku-5.5")):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(_run(name, judge, scores, labels)), encoding="utf-8")
        paths.append(str(p))
    out = tmp_path / "cal.json"
    cj.main(["--sonnet", paths[0], "--haiku", paths[1], paths[2], "--out", str(out)])

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["sonnet_judge"] == "anthropic/claude-sonnet-5.5"
    assert result["haiku_judges"] == ["anthropic/claude-haiku-5.5"]
    assert result["metrics"]["faithfulness"]["n"] == 2
    assert result["refusal"]["majority_agree"] == 3
    assert "Spearman" in capsys.readouterr().out


def _opus_files(tmp_path, labels, qids=("q1", "q2", "q3")):
    packet = tmp_path / "opus_packet_x.json"
    packet.write_text(json.dumps([{"qid": q} for q in qids]), encoding="utf-8")
    path = tmp_path / "opus_labels_x.json"
    path.write_text(json.dumps(labels), encoding="utf-8")
    return str(path), str(packet)


def test_load_opus_labels_reads_correctness_only(tmp_path):
    labels = {"q1": {"label": "correct", "reason": "r"}, "q2": {"label": "partial", "reason": ""},
              "q3": {"label": "incorrect", "reason": ""}}
    run = cj.load_opus_labels(*_opus_files(tmp_path, labels))
    assert [(r["id"], r["correctness_score"]) for r in run["per_question"]] == [
        ("q1", 1.0), ("q2", 0.5), ("q3", 0.0)]
    assert run["meta"]["models"]["judge"]["model"] == cj.OPUS_JUDGE


@pytest.mark.parametrize("labels, message", [
    ({"q1": {"label": "correct"}, "q2": {"label": "correct"}}, "missing qids: ['q3']"),
    ({"q1": {"label": "correct"}, "q2": {"label": "correct"}, "q3": {"label": "wrong"}}, "invalid labels: ['q3']"),
    ({"q1": {"label": "correct"}, "q2": {"label": "correct"}, "q3": "correct"}, "invalid labels: ['q3']"),
    ({"q1": {"label": "correct"}, "q2": {"label": "correct"}, "q3": {"label": "correct"},
      "q9": {"label": "correct"}}, "unknown qids: ['q9']"),
])
def test_load_opus_labels_rejects_bad_files(tmp_path, labels, message):
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        cj.load_opus_labels(*_opus_files(tmp_path, labels))


def test_main_with_opus_labels(tmp_path, capsys):
    scores = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
    haiku_labels = {"q1": "correct", "q2": "partial", "q3": "incorrect"}
    paths = []
    for i in range(3):
        recs = [{"id": q, "answerable": True, "correctness_label": lab, "correctness_score": scores[lab],
                 "refusal_label": "answer", "faithfulness": 0.5}
                for q, lab in haiku_labels.items()]
        p = tmp_path / f"h{i}.json"
        p.write_text(json.dumps({"meta": {"models": {"judge": {"model": "haiku"}}}, "per_question": recs}),
                     encoding="utf-8")
        paths.append(str(p))
    labels_path, packet_path = _opus_files(
        tmp_path, {"q1": {"label": "correct"}, "q2": {"label": "partial"}, "q3": {"label": "correct"}})
    out = tmp_path / "cal.json"
    cj.main(["--opus", labels_path, "--packet", packet_path, "--haiku", *paths, "--out", str(out)])

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["reference"] == "Opus"
    assert result["correctness"]["majority_agree"] == 2
    assert result["metrics"]["correctness_score"]["n"] == 3
    assert result["metrics"]["faithfulness"]["n"] == 0
    assert result["refusal"]["n"] == 0
    md = capsys.readouterr().out
    assert "Opus 均值" in md and "q3: Opus=correct" in md


def test_main_opus_requires_packet(tmp_path):
    with pytest.raises(SystemExit, match="--packet"):
        cj.main(["--opus", "x.json", "--haiku", "h.json"])
