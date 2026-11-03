"""Paired comparison of two RAG eval configs (each run 3 times).

Input: the result JSON files written by ``eval_rag.py`` (``per_question`` list).

For each metric:
  1. Per question, average the score over the runs of one config.
  2. Compare the two configs question by question: better / same / worse
     (|diff| < 0.05 counts as same).
  3. Paired bootstrap over questions (1000 resamples, fixed seed) gives a 95% CI
     for the mean difference. The verdict is "improved" or "declined" only when
     the CI excludes 0; otherwise "flat".
  4. Run-level range: mean, min, max of the per-run means.

Subsets: the original 20 questions, all answerable questions, and per question
type (term / paraphrase / heading: counts only, no CI). Refusal metrics are
reported as counts (e.g. 8/10), no CI. Refusal questions marked
``refusal_scored: false`` (answered by the L1 notes) are left out of refusal accuracy.

Latency, input tokens and DeepSeek cost: mean and range of the per-run values, no CI.

The two configs may differ in collection and --mode (that is what is compared);
commit, dirty flag, dataset and judge model must match.

Usage:
    cd backend
    python -m evaluation.stats --a A1 r1.json r2.json r3.json --b A2 r4.json r5.json r6.json
    python -m evaluation.stats ... --out results/stats/a1_vs_a2.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# Per-question metrics (``reranker_lift`` is derived: rr_reranked - rr_raw).
METRICS = (
    "recall_at_k",
    "rr",
    "rr_raw",
    "rr_reranked",
    "reranker_lift",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "correctness_score",
)
# Per-question usage fields: per-run means, no CI
USAGE_FIELDS = ("input_tokens", "output_tokens", "cost_usd_actual", "cost_usd_no_cache")
SAME_THRESHOLD = 0.05
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20261102
TYPE_SUBSETS = ("term", "paraphrase", "heading")
REFUSAL_LABEL = "refusal"


# ── Loading ───────────────────────────────────────────────────


def load_runs(paths: list[str | Path]) -> list[dict[str, Any]]:
    runs = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        if not report.get("per_question"):
            raise ValueError(f"{path}: no per_question records")
        report["_path"] = str(path)
        runs.append(report)
    return runs


def check_run_meta(runs: list[dict[str, Any]]) -> list[str]:
    """List meta problems: status not completed, dirty tree, or commit / dataset / judge mismatch.

    All runs in one comparison must come from the same clean commit, question set
    and judge model (so the Sonnet calibration run never mixes into a comparison).
    """
    problems = []
    commits, datasets, judges = set(), set(), set()
    for run in runs:
        meta = run.get("meta") or {}
        path = run.get("_path", "?")
        if meta.get("status") != "completed":
            problems.append(f"{path}: status={meta.get('status')!r}")
        commit = meta.get("commit") or {}
        if commit.get("dirty") is not False:
            problems.append(f"{path}: dirty={commit.get('dirty')!r}")
        commits.add(commit.get("sha"))
        datasets.add((meta.get("dataset") or {}).get("sha256"))
        judges.add(((meta.get("models") or {}).get("judge") or {}).get("model"))
    if len(commits) > 1:
        problems.append(f"commit differs across runs: {sorted(map(str, commits))}")
    if len(datasets) > 1:
        problems.append(f"dataset sha256 differs across runs: {sorted(map(str, datasets))}")
    if len(judges) > 1:
        problems.append(f"judge model differs across runs: {sorted(map(str, judges))}")
    return problems


def _metric_value(record: dict[str, Any], metric: str) -> float | None:
    if metric == "reranker_lift":
        a, b = record.get("rr_reranked"), record.get("rr_raw")
        return None if a is None or b is None else float(a) - float(b)
    value = record.get(metric)
    if isinstance(value, (int, float)) and not math.isnan(value):
        return float(value)
    return None


def question_info(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """id -> {type, answerable, origin, question} from the first run that has it."""
    info: dict[str, dict[str, Any]] = {}
    for run in runs:
        for rec in run["per_question"]:
            info.setdefault(
                rec["id"],
                {
                    "type": rec.get("type"),
                    "answerable": rec.get("answerable", True),
                    "origin": rec.get("origin"),
                    "question": rec.get("question"),
                },
            )
    return info


def per_question_means(runs: list[dict[str, Any]], metric: str) -> dict[str, float]:
    """Average one metric per question over runs, skipping missing values."""
    values: dict[str, list[float]] = {}
    for run in runs:
        for rec in run["per_question"]:
            v = _metric_value(rec, metric)
            if v is not None:
                values.setdefault(rec["id"], []).append(v)
    return {qid: sum(vs) / len(vs) for qid, vs in values.items()}


def run_means(runs: list[dict[str, Any]], metric: str, ids: set[str]) -> list[float]:
    """Mean of one metric over ``ids`` for each run (runs with no values are skipped)."""
    out = []
    for run in runs:
        vals = [
            v
            for rec in run["per_question"]
            if rec["id"] in ids and (v := _metric_value(rec, metric)) is not None
        ]
        if vals:
            out.append(sum(vals) / len(vals))
    return out


def summarize_range(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "min": None, "max": None, "runs": 0}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "runs": len(values),
    }


# ── Paired comparison ─────────────────────────────────────────


def bootstrap_ci(
    diffs: list[float],
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% percentile CI of the mean of paired diffs."""
    arr = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_means = arr[idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def verdict(ci: tuple[float, float] | None) -> str:
    if ci is None:
        return "n/a"
    low, high = ci
    if low > 0:
        return "improved"
    if high < 0:
        return "declined"
    return "flat"


def paired_compare(
    a: dict[str, float],
    b: dict[str, float],
    ids: set[str],
    *,
    with_ci: bool = True,
    threshold: float = SAME_THRESHOLD,
) -> dict[str, Any]:
    """Compare B against A on the questions in ``ids`` that both configs scored."""
    common = sorted(qid for qid in ids if qid in a and qid in b)
    if not common:
        return {"n": 0}
    diffs = [b[qid] - a[qid] for qid in common]
    result: dict[str, Any] = {
        "n": len(common),
        "mean_a": sum(a[q] for q in common) / len(common),
        "mean_b": sum(b[q] for q in common) / len(common),
        "diff": sum(diffs) / len(diffs),
        "better": sum(1 for d in diffs if d >= threshold),
        "same": sum(1 for d in diffs if abs(d) < threshold),
        "worse": sum(1 for d in diffs if d <= -threshold),
    }
    if with_ci:
        ci = bootstrap_ci(diffs)
        result["ci95"] = list(ci)
        result["verdict"] = verdict(ci)
    return result


# ── Refusal metrics ───────────────────────────────────────────


def refusal_counts(run: dict[str, Any]) -> dict[str, Any]:
    """Refusal accuracy (refusal questions judged as refusals) and false refusals."""
    refusal_q = [
        r for r in run["per_question"]
        if not r.get("answerable", True) and r.get("refusal_scored", True)
    ]
    answerable_q = [r for r in run["per_question"] if r.get("answerable", True)]
    return {
        "refusal_correct": sum(1 for r in refusal_q if r.get("refusal_label") == REFUSAL_LABEL),
        "refusal_total": len(refusal_q),
        "false_refusals": sum(1 for r in answerable_q if r.get("refusal_label") == REFUSAL_LABEL),
        "answerable_total": len(answerable_q),
        "unjudged": sum(1 for r in run["per_question"] if r.get("refusal_label") is None),
        "excluded": [
            r["id"] for r in run["per_question"]
            if not r.get("answerable", True) and not r.get("refusal_scored", True)
        ],
        "fully_refused": sum(
            1 for r in run["per_question"] if (r.get("citation") or {}).get("fully_refused")
        ),
    }


def refusal_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    per_run = [refusal_counts(run) for run in runs]

    def _rate(num: str, den: str) -> list[float]:
        return [c[num] / c[den] for c in per_run if c[den]]

    return {
        "per_run": per_run,
        "refusal_accuracy": summarize_range(_rate("refusal_correct", "refusal_total")),
        "false_refusal_rate": summarize_range(_rate("false_refusals", "answerable_total")),
    }


# ── Latency ───────────────────────────────────────────────────


def latency_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for key in ("latency_p50_ms", "latency_p95_ms"):
        vals = [
            run["ir_metrics"][key]
            for run in runs
            if isinstance(run.get("ir_metrics"), dict) and key in run["ir_metrics"]
        ]
        out[key] = summarize_range(vals)
    return out


def usage_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-run mean of each usage field over all questions, then mean and range over runs."""
    out = {}
    for key in USAGE_FIELDS:
        per_run = []
        for run in runs:
            vals = [r[key] for r in run["per_question"] if isinstance(r.get(key), (int, float))]
            if vals:
                per_run.append(sum(vals) / len(vals))
        out[key] = summarize_range(per_run)
    return out


# ── Full comparison ───────────────────────────────────────────


def _is_original(item: dict[str, Any]) -> bool:
    return item.get("type") == "original" or item.get("origin") == "original_v1"


def subset_ids(info: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    answerable = {q for q, i in info.items() if i["answerable"]}
    subsets = {
        "original_20": {q for q in answerable if _is_original(info[q])},
        "answerable": answerable,
    }
    for t in TYPE_SUBSETS:
        subsets[f"type_{t}"] = {q for q in answerable if info[q]["type"] == t}
    return subsets


def signals_missing(rec: dict[str, Any]) -> bool:
    """The answer says the notes have no answer: fallback "not found" or a refusal."""
    return bool(rec.get("not_found")) or rec.get("refusal_label") == REFUSAL_LABEL


def found_ids(runs: list[dict[str, Any]], answerable: set[str]) -> set[str]:
    """Answerable questions that no run (of either config) answered with "not found" or a refusal."""
    missing = {rec["id"] for run in runs for rec in run["per_question"] if signals_missing(rec)}
    return answerable - missing


def not_found_counts(run: dict[str, Any]) -> dict[str, int]:
    """Answers flagged not_found: on scored refusal questions and on answerable questions."""
    recs = run["per_question"]
    refusal = [r for r in recs if not r.get("answerable", True) and r.get("refusal_scored", True)]
    answerable = [r for r in recs if r.get("answerable", True)]
    return {
        "refusal_not_found": sum(1 for r in refusal if r.get("not_found")),
        "refusal_total": len(refusal),
        "answerable_not_found": sum(1 for r in answerable if r.get("not_found")),
        "answerable_total": len(answerable),
    }


def compare_configs(
    label_a: str,
    runs_a: list[dict[str, Any]],
    label_b: str,
    runs_b: list[dict[str, Any]],
) -> dict[str, Any]:
    info = question_info(runs_a + runs_b)
    subsets = subset_ids(info)
    subsets["answerable_found"] = found_ids(runs_a + runs_b, subsets["answerable"])
    report: dict[str, Any] = {
        "a": {"label": label_a, "files": [r["_path"] for r in runs_a]},
        "b": {"label": label_b, "files": [r["_path"] for r in runs_b]},
        "settings": {
            "same_threshold": SAME_THRESHOLD,
            "n_bootstrap": N_BOOTSTRAP,
            "seed": BOOTSTRAP_SEED,
        },
        "subset_sizes": {name: len(ids) for name, ids in subsets.items()},
        "metrics": {},
    }
    for metric in METRICS:
        a_means = per_question_means(runs_a, metric)
        b_means = per_question_means(runs_b, metric)
        if not a_means and not b_means:
            continue
        per_subset = {}
        for name, ids in subsets.items():
            per_subset[name] = {
                "range_a": summarize_range(run_means(runs_a, metric, ids)),
                "range_b": summarize_range(run_means(runs_b, metric, ids)),
                "paired": paired_compare(
                    a_means, b_means, ids, with_ci=not name.startswith("type_")
                ),
            }
        report["metrics"][metric] = per_subset
    report["refusal"] = {"a": refusal_summary(runs_a), "b": refusal_summary(runs_b)}
    report["not_found"] = {
        "a": [not_found_counts(r) for r in runs_a],
        "b": [not_found_counts(r) for r in runs_b],
    }
    report["latency"] = {"a": latency_summary(runs_a), "b": latency_summary(runs_b)}
    report["usage"] = {"a": usage_summary(runs_a), "b": usage_summary(runs_b)}
    return report


# ── Output ────────────────────────────────────────────────────


def _fmt(v: float | None, digits: int = 3) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def _fmt_range(r: dict[str, Any]) -> str:
    if not r.get("runs"):
        return "—"
    return f"{_fmt(r['mean'])} ({_fmt(r['min'])}–{_fmt(r['max'])})"


def to_markdown(report: dict[str, Any]) -> str:
    la, lb = report["a"]["label"], report["b"]["label"]
    lines = [f"## {lb} vs {la}", ""]
    for subset in ("answerable", "original_20", "answerable_found"):
        lines += [
            f"### {subset} (n={report['subset_sizes'].get(subset, 0)})",
            "",
            f"| Metric | {la} mean (min–max) | {lb} mean (min–max) | Diff | 95% CI | Better / Same / Worse | Verdict |",
            "|---|---|---|---|---|---|---|",
        ]
        for metric, per_subset in report["metrics"].items():
            s = per_subset.get(subset)
            if not s:
                continue
            p = s["paired"]
            if not p.get("n"):
                continue
            ci = p.get("ci95")
            lines.append(
                f"| {metric} | {_fmt_range(s['range_a'])} | {_fmt_range(s['range_b'])} | "
                f"{p['diff']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                f"{p['better']} / {p['same']} / {p['worse']} | {p['verdict']} |"
            )
        lines.append("")
    lines += [
        "### By question type (Better / Same / Worse, no CI)",
        "",
        "| Metric | " + " | ".join(TYPE_SUBSETS) + " |",
        "|---|" + "---|" * len(TYPE_SUBSETS),
    ]
    for metric, per_subset in report["metrics"].items():
        cells = []
        for t in TYPE_SUBSETS:
            p = per_subset[f"type_{t}"]["paired"]
            cells.append(f"{p['better']} / {p['same']} / {p['worse']}" if p.get("n") else "—")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["### Refusal, latency, tokens and cost (no CI)", "", f"| | {la} | {lb} |", "|---|---|---|"]
    for key, num, den in (
        ("refusal_accuracy", "refusal_correct", "refusal_total"),
        ("false_refusal_rate", "false_refusals", "answerable_total"),
    ):
        cells = []
        for side in ("a", "b"):
            summ = report["refusal"][side]
            counts = ", ".join(f"{c[num]}/{c[den]}" for c in summ["per_run"])
            cells.append(f"{_fmt_range(summ[key])} [{counts}]")
        lines.append(f"| {key} | {cells[0]} | {cells[1]} |")
    cells = [
        ", ".join(str(c["fully_refused"]) for c in report["refusal"][side]["per_run"])
        for side in ("a", "b")
    ]
    lines.append(f"| fully_refused (per run) | {cells[0]} | {cells[1]} |")
    for num, den in (("refusal_not_found", "refusal_total"), ("answerable_not_found", "answerable_total")):
        cells = [
            ", ".join(f"{c[num]}/{c[den]}" for c in report.get("not_found", {}).get(side, []))
            for side in ("a", "b")
        ]
        lines.append(f"| {num} (per run) | {cells[0]} | {cells[1]} |")
    for key in ("latency_p50_ms", "latency_p95_ms"):
        cells = []
        for side in ("a", "b"):
            r = report["latency"][side][key]
            cells.append(
                "—" if not r["runs"] else f"{r['mean']:.0f} ({r['min']:.0f}–{r['max']:.0f})"
            )
        lines.append(f"| {key} | {cells[0]} | {cells[1]} |")
    for key in USAGE_FIELDS:
        digits = 0 if key.endswith("tokens") else 6
        cells = []
        for side in ("a", "b"):
            r = report.get("usage", {}).get(side, {}).get(key) or {}
            cells.append(
                "—" if not r.get("runs")
                else f"{r['mean']:.{digits}f} ({r['min']:.{digits}f}–{r['max']:.{digits}f})"
            )
        lines.append(f"| {key} (per question) | {cells[0]} | {cells[1]} |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired comparison of two eval configs.")
    parser.add_argument("--a", nargs="+", required=True, metavar=("LABEL", "FILE"),
                        help="Baseline label followed by its result files.")
    parser.add_argument("--b", nargs="+", required=True, metavar=("LABEL", "FILE"),
                        help="Candidate label followed by its result files.")
    parser.add_argument("--out", help="Write the full comparison as JSON to this path.")
    args = parser.parse_args(argv)
    for side in ("a", "b"):
        if len(getattr(args, side)) < 2:
            parser.error(f"--{side} needs a label and at least one result file")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    label_a, *files_a = args.a
    label_b, *files_b = args.b
    runs_a, runs_b = load_runs(files_a), load_runs(files_b)
    problems = check_run_meta(runs_a + runs_b)
    if problems:
        raise SystemExit("Result files are not comparable:\n  " + "\n  ".join(problems))
    report = compare_configs(label_a, runs_a, label_b, runs_b)
    print(to_markdown(report))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
