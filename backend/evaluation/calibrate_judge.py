"""Judge calibration: compare one reference judge with three Haiku runs of the same config.

The reference is either a Sonnet result file (`--sonnet`) or an Opus labels file
(`--opus` + `--packet`, task doc §7.3 step 4). An Opus labels file only has
correctness labels, so only the correctness rows are reported for it.

For each RAGAS metric and the answer correctness score, per question:
  Haiku score = mean over the Haiku runs; Sonnet score = the single Sonnet run.
  Report Spearman correlation and mean difference (Sonnet - Haiku) over the
  questions that have both scores. A correlation below 0.5 is flagged: that
  metric's conclusions can only be "flat" or "for reference".

Refusal labels and correctness labels, per question:
  - majority agreement: Sonnet label vs the Haiku majority label (no majority
    among the Haiku runs counts as disagreement)
  - per-run agreement: Sonnet label vs each Haiku run label, averaged

Usage:
    cd backend
    python -m evaluation.calibrate_judge --sonnet s.json --haiku h1.json h2.json h3.json [--out cal.json]
    python -m evaluation.calibrate_judge --opus opus_labels_X.json --packet opus_packet_X.json \\
        --haiku h1.json h2.json h3.json [--out cal.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.eval_rag import CORRECTNESS_SCORES
from evaluation.stats import load_runs, per_question_means

RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
METRICS = (*RAGAS_METRICS, "correctness_score")
# per-question label field -> labels that count as "not judged"
LABEL_FIELDS = {"refusal_label": (), "correctness_label": ("unjudged",)}
MIN_SPEARMAN = 0.5
OPUS_JUDGE = "claude-opus-5-5 (Claude Code, user-run)"


def load_opus_labels(labels_path: str | Path, packet_path: str | Path) -> dict[str, Any]:
    """Read an Opus labels file as a pseudo result run with correctness labels only.

    Every packet question must have a label, and labels must be one of the 3
    correctness labels; otherwise raise (the user fixes the file, we do not).
    """
    with open(packet_path, "r", encoding="utf-8") as f:
        expected = [item["qid"] for item in json.load(f)]
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    problems = []
    missing = sorted(set(expected) - set(labels))
    extra = sorted(set(labels) - set(expected))
    if missing:
        problems.append(f"missing qids: {missing}")
    if extra:
        problems.append(f"unknown qids: {extra}")
    bad = sorted(
        qid for qid, v in labels.items()
        if not isinstance(v, dict) or v.get("label") not in CORRECTNESS_SCORES
    )
    if bad:
        problems.append(f"invalid labels: {bad}")
    if problems:
        raise ValueError(f"{labels_path}: " + "; ".join(problems))
    records = [
        {"id": qid, "correctness_label": labels[qid]["label"],
         "correctness_score": CORRECTNESS_SCORES[labels[qid]["label"]],
         "correctness_reason": labels[qid].get("reason", "")}
        for qid in sorted(expected)
    ]
    return {"_path": str(labels_path), "meta": {"models": {"judge": {"model": OPUS_JUDGE}}},
            "per_question": records}


def _ranks(values: list[float]) -> np.ndarray:
    """1-based ranks; ties get the average of their ranks."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1)
    for v in np.unique(arr):
        mask = arr == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman rank correlation; None when undefined (n < 3 or a constant side)."""
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def compare_metric(sonnet_run: dict[str, Any], haiku_runs: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    sonnet = per_question_means([sonnet_run], metric)
    haiku = per_question_means(haiku_runs, metric)
    ids = sorted(set(sonnet) & set(haiku))
    xs = [sonnet[i] for i in ids]
    ys = [haiku[i] for i in ids]
    rho = spearman(xs, ys)
    return {
        "n": len(ids),
        "spearman": rho,
        "mean_diff": float(np.mean(np.subtract(xs, ys))) if ids else None,
        "sonnet_mean": float(np.mean(xs)) if ids else None,
        "haiku_mean": float(np.mean(ys)) if ids else None,
        "low_correlation": rho is None or rho < MIN_SPEARMAN,
    }


def _labels(run: dict[str, Any], field: str = "refusal_label") -> dict[str, str]:
    skip = LABEL_FIELDS.get(field, ())
    return {
        r["id"]: r[field] for r in run["per_question"] if r.get(field) and r[field] not in skip
    }


def compare_labels(
    sonnet_run: dict[str, Any], haiku_runs: list[dict[str, Any]], field: str = "refusal_label"
) -> dict[str, Any]:
    sonnet = _labels(sonnet_run, field)
    haiku = [_labels(r, field) for r in haiku_runs]
    ids = sorted(i for i in sonnet if all(i in h for h in haiku))
    majority_agree = 0
    per_run_agree = 0
    disagreements = []
    for qid in ids:
        top, count = Counter(h[qid] for h in haiku).most_common(1)[0]
        majority = top if count * 2 > len(haiku) else None
        if majority == sonnet[qid]:
            majority_agree += 1
        else:
            disagreements.append({"id": qid, "sonnet": sonnet[qid], "haiku": [h[qid] for h in haiku]})
        per_run_agree += sum(h[qid] == sonnet[qid] for h in haiku)
    n = len(ids)
    return {
        "n": n,
        "majority_agree": majority_agree,
        "majority_agreement": majority_agree / n if n else None,
        "per_run_agreement": per_run_agree / (n * len(haiku)) if n else None,
        "disagreements": disagreements,
    }


def calibrate(
    sonnet_run: dict[str, Any], haiku_runs: list[dict[str, Any]], reference: str = "Sonnet"
) -> dict[str, Any]:
    def _judge(run: dict[str, Any]) -> str | None:
        return (((run.get("meta") or {}).get("models") or {}).get("judge") or {}).get("model")

    return {
        "reference": reference,
        "sonnet_file": sonnet_run.get("_path"),
        "haiku_files": [r.get("_path") for r in haiku_runs],
        "sonnet_judge": _judge(sonnet_run),
        "haiku_judges": sorted({str(_judge(r)) for r in haiku_runs}),
        "metrics": {m: compare_metric(sonnet_run, haiku_runs, m) for m in METRICS},
        "refusal": compare_labels(sonnet_run, haiku_runs, "refusal_label"),
        "correctness": compare_labels(sonnet_run, haiku_runs, "correctness_label"),
    }


def _fmt(v: float | None, digits: int = 3) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def to_markdown(result: dict[str, Any]) -> str:
    ref_name = result.get("reference", "Sonnet")
    lines = [
        f"{ref_name} judge: `{result['sonnet_judge']}`; Haiku judge: `{', '.join(result['haiku_judges'])}`",
        "",
        f"| 指标 | 题数 | Spearman | 平均差值（{ref_name} − Haiku） | {ref_name} 均值 | Haiku 均值 | 相关系数 < 0.5 |",
        "|---|---|---|---|---|---|---|",
    ]
    for metric, r in result["metrics"].items():
        if not r["n"]:
            continue
        lines.append(
            f"| {metric} | {r['n']} | {_fmt(r['spearman'])} | {_fmt(r['mean_diff'])} | "
            f"{_fmt(r['sonnet_mean'])} | {_fmt(r['haiku_mean'])} | {'是' if r['low_correlation'] else '否'} |"
        )
    for key, title in (("refusal", "拒答标签"), ("correctness", "回答正确率标签")):
        ref = result.get(key)
        if not ref or not ref["n"]:
            continue
        lines += [
            "",
            f"{title}一致率：多数标签 {ref['majority_agree']}/{ref['n']}（{_fmt(ref['majority_agreement'])}），"
            f"逐次平均 {_fmt(ref['per_run_agreement'])}",
        ]
        for d in ref["disagreements"]:
            lines.append(f"- {d['id']}: {ref_name}={d['sonnet']}, Haiku={d['haiku']}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the Haiku judge against a reference judge.")
    ref = parser.add_mutually_exclusive_group(required=True)
    ref.add_argument("--sonnet", help="Sonnet result file.")
    ref.add_argument("--opus", help="Opus labels file (opus_labels_<run_id>.json).")
    parser.add_argument("--packet", help="Packet file the Opus labels answer (required with --opus).")
    parser.add_argument("--haiku", nargs="+", required=True, help="Haiku result files (same config).")
    parser.add_argument("--out", help="Write the calibration as JSON to this path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.opus:
        if not args.packet:
            raise SystemExit("--opus needs --packet")
        result = calibrate(load_opus_labels(args.opus, args.packet), load_runs(args.haiku), reference="Opus")
    else:
        result = calibrate(load_runs([args.sonnet])[0], load_runs(args.haiku))
    print(to_markdown(result))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
