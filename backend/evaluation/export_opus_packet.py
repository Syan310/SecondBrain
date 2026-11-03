"""Export one result file as a calibration packet for Opus labelling (task doc §7.3 step 4).

Writes two files to the output directory:
  opus_packet_<run_id>.json  questions only (qid, question, ground_truth, clean_answer),
                             shuffled with seed 20261102; no judge scores, config,
                             collection or run number
  opus_packet_<run_id>.md    the correctness judge prompt, its SHA-256, and the
                             output format of opus_labels_<run_id>.json

The user gives the packet to Opus 5.5 in a separate Claude Code session; the labels
file it writes is read by `calibrate_judge.py --opus`.

Usage:
    cd backend
    python -m evaluation.export_opus_packet --run results/rag/rag_eval_X.json [--out-dir results/calibration]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from evaluation.eval_rag import CORRECTNESS_JUDGE_PROMPT, CORRECTNESS_SCORES

SHUFFLE_SEED = 20261102
PACKET_FIELDS = ("qid", "question", "ground_truth", "clean_answer")
DEFAULT_OUT_DIR = Path(__file__).parent / "results" / "calibration"


def build_packet(report: dict[str, Any]) -> list[dict[str, str]]:
    items = [
        {"qid": r["id"], "question": r["question"], "ground_truth": r["ground_truth"],
         "clean_answer": r["clean_answer"]}
        for r in report["per_question"]
    ]
    items.sort(key=lambda item: item["qid"])
    random.Random(SHUFFLE_SEED).shuffle(items)
    return items


def build_instructions(run_id: str, packet_name: str) -> str:
    prompt_sha = hashlib.sha256(CORRECTNESS_JUDGE_PROMPT.encode("utf-8")).hexdigest()
    labels = " | ".join(CORRECTNESS_SCORES)
    return "\n".join([
        f"# 回答正确率校准题目包：{run_id}",
        "",
        f"题目文件：`{packet_name}`。每道题有 `qid`、`question`、`ground_truth`、`clean_answer` 4 个字段。",
        "",
        "对每道题，把 `question`、`ground_truth`、`clean_answer` 分别填进下面提示词的"
        " `{question}`、`{ground_truth}`、`{answer}`，按提示词的要求判定。"
        "提示词里的 `{{` 和 `}}` 是 Python 格式化的转义，读作 `{` 和 `}`。",
        "",
        f"提示词 `CORRECTNESS_JUDGE_PROMPT` 的 SHA-256（UTF-8）：`{prompt_sha}`",
        "",
        "```text",
        CORRECTNESS_JUDGE_PROMPT.rstrip("\n"),
        "```",
        "",
        "## 输出格式",
        "",
        f"写一个 JSON 文件 `opus_labels_{run_id}.json`，放在题目文件的同一个目录。"
        "每道题都要有一项，键是 `qid`：",
        "",
        "```json",
        '{"q001": {"label": "' + labels + '", "reason": "一句话理由"}}',
        "```",
        "",
        "`label` 只能是 " + "、".join(f"`{label}`" for label in CORRECTNESS_SCORES) + " 之一。",
        "",
    ])


def export_packet(run_path: Path, out_dir: Path) -> tuple[Path, Path]:
    with open(run_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    meta = report.get("meta") or {}
    if meta.get("status") != "completed":
        raise SystemExit(f"{run_path}: status={meta.get('status')!r}, export only completed runs")
    run_id = meta["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"opus_packet_{run_id}.json"
    md_path = out_dir / f"opus_packet_{run_id}.md"
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(build_packet(report), f, ensure_ascii=False, indent=2)
        f.write("\n")
    md_path.write_bytes(build_instructions(run_id, json_path.name).encode("utf-8"))
    return json_path, md_path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a calibration packet for Opus labelling.")
    parser.add_argument("--run", required=True, help="Result file to export (R1 run 1).")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)
    for path in export_packet(Path(args.run), Path(args.out_dir)):
        print(f"{path}  sha256={sha256_file(path)}")


if __name__ == "__main__":
    main()
