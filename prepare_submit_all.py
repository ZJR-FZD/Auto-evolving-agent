"""
Prepare submit/ artifacts for Benchmark, SimpleVQA, and 2Wiki.

Missing inputs are reported but do not abort the script.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from answer_utils import clean_pred_for_submit

REQUIRED_RESULT_KEYS = ["index", "instruction", "image", "answer", "pred"]
REQUIRED_TRAJ_KEYS = {"timestamp", "step_id", "role", "content", "tool_call_id"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_pred(raw: Any, question: str = "") -> str:
    return clean_pred_for_submit(raw, question)


def copy_if_exists(src: Path, dst: Path, report: dict[str, Any]) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        report["copied"].append(str(dst))
    else:
        report["missing"].append(str(src))


def normalize_result_file(src: Path, raw_dst: Path, final_dst: Path, report: dict[str, Any]) -> None:
    if not src.exists():
        report["missing"].append(str(src))
        return
    raw_rows = []
    final_rows = []
    for i, row in enumerate(load_jsonl(src)):
        idx = row.get("idx", row.get("index", i))
        instruction = row.get("question") or row.get("instruction") or ""
        image = row.get("image", "")
        answer = row.get("gold") or row.get("ground_truth") or row.get("answer") or ""
        pred = row.get("prediction") or row.get("pred") or row.get("final_prediction") or ""
        base = {"index": idx, "instruction": instruction, "image": image, "answer": answer}
        raw_rows.append({**base, "pred": pred})
        final_rows.append({**base, "pred": clean_pred(pred, instruction)})
    write_jsonl(raw_dst, raw_rows)
    write_jsonl(final_dst, final_rows)
    report["generated"].extend([str(raw_dst), str(final_dst)])


def merge_trajectories(src_dir: Path, dst: Path, report: dict[str, Any]) -> None:
    if not src_dir.exists():
        report["missing"].append(str(src_dir))
        return
    files = sorted(src_dir.glob("*.jsonl"))
    bad = []
    with open(dst, "w", encoding="utf-8") as w:
        for path in files:
            for row in load_jsonl(path):
                if not REQUIRED_TRAJ_KEYS <= set(row.keys()):
                    bad.append(str(path))
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["generated"].append(str(dst))
    if bad:
        report["trajectory_format_errors"][str(dst)] = sorted(set(bad))[:20]


def check_result_format(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "lines": 0, "errors": ["missing"]}
    errors = []
    rows = load_jsonl(path)
    for line_no, row in enumerate(rows, 1):
        if list(row.keys()) != REQUIRED_RESULT_KEYS:
            errors.append({"line": line_no, "keys": list(row.keys())})
    return {"exists": True, "lines": len(rows), "errors": errors[:20]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="submit")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"generated": [], "copied": [], "missing": [], "format": {}, "trajectory_format_errors": {}}

    # Benchmark competition files and normalized jsonl.
    for name in ["group_7.csv", "group_7.json", "group_7.zip"]:
        copy_if_exists(Path("results") / name, out / f"benchmark_{name}", report)
    copy_if_exists(Path("results/group_7_raw_results.jsonl"), out / "benchmark_raw_results.jsonl", report)
    copy_if_exists(Path("results/group_7_final_results.jsonl"), out / "benchmark_final_results.jsonl", report)
    merge_trajectories(Path("trajectories/benchmark"), out / "benchmark_trajectories.jsonl", report)

    # SimpleVQA.
    normalize_result_file(Path("results/simpleqa_results.jsonl"), out / "simpleqa_raw_results.jsonl", out / "simpleqa_final_results.jsonl", report)
    merge_trajectories(Path("trajectories/simpleqa"), out / "simpleqa_trajectories.jsonl", report)

    # 2Wiki baseline / reflection variants.
    normalize_result_file(Path("results/baseline_2wiki_full.jsonl"), out / "2wiki_baseline_raw_results.jsonl", out / "2wiki_baseline_final_results.jsonl", report)
    normalize_result_file(Path("results/reflection_2wiki_full.jsonl"), out / "2wiki_reflection_raw_results.jsonl", out / "2wiki_reflection_final_results.jsonl", report)
    merge_trajectories(Path("trajectories_baseline_2wiki_full"), out / "2wiki_baseline_trajectories.jsonl", report)
    merge_trajectories(Path("trajectories_reflection_2wiki_full"), out / "2wiki_reflection_trajectories.jsonl", report)

    for path in out.glob("*_results.jsonl"):
        report["format"][str(path)] = check_result_format(path)
    report_path = out / "format_check_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
