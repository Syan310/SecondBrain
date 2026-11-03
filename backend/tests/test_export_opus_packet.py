import hashlib
import json

import pytest

from evaluation import export_opus_packet as ex
from evaluation.eval_rag import CORRECTNESS_JUDGE_PROMPT


def _report(status="completed", n=10):
    recs = [
        {"id": f"q{i:03d}", "question": f"Q{i}", "ground_truth": f"G{i}", "clean_answer": f"A{i}",
         "correctness_label": "correct", "correctness_score": 1.0, "refusal_label": "answer",
         "faithfulness": 0.9, "retrieved": [{"source_doc": "x.md"}]}
        for i in range(1, n + 1)
    ]
    meta = {"run_id": "rag_20261102_000000", "status": status,
            "config": {"milvus_collection_name": "secondbrain_l1"}}
    return {"meta": meta, "per_question": recs}


def _write(tmp_path, report):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_packet_has_only_four_fields_and_is_shuffled_deterministically(tmp_path):
    json_path, md_path = ex.export_packet(_write(tmp_path, _report()), tmp_path / "out")
    assert json_path.name == "opus_packet_rag_20261102_000000.json"
    assert md_path.name == "opus_packet_rag_20261102_000000.md"
    items = json.loads(json_path.read_text(encoding="utf-8"))
    assert all(set(item) == set(ex.PACKET_FIELDS) for item in items)
    qids = [item["qid"] for item in items]
    assert sorted(qids) == [f"q{i:03d}" for i in range(1, 11)]
    assert qids != sorted(qids)
    first = json_path.read_bytes()
    ex.export_packet(_write(tmp_path, _report()), tmp_path / "out")
    assert json_path.read_bytes() == first
    text = json_path.read_text(encoding="utf-8")
    for leaked in ("secondbrain_l1", "rag_2026", "correctness_label", "faithfulness"):
        assert leaked not in text


def test_instructions_contain_prompt_and_sha(tmp_path):
    _, md_path = ex.export_packet(_write(tmp_path, _report()), tmp_path)
    md = md_path.read_text(encoding="utf-8")
    assert CORRECTNESS_JUDGE_PROMPT.rstrip("\n") in md
    assert hashlib.sha256(CORRECTNESS_JUDGE_PROMPT.encode("utf-8")).hexdigest() in md
    assert "opus_labels_rag_20261102_000000.json" in md


def test_refuses_invalid_run(tmp_path):
    with pytest.raises(SystemExit, match="invalid"):
        ex.export_packet(_write(tmp_path, _report(status="invalid")), tmp_path)
