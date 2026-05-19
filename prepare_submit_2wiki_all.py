"""
Build official-format 2Wiki submission files for baseline and reflection runs.

Output JSONL rows use exactly:
{"index":, "instruction":, "image":, "answer":, "pred":}
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from answer_utils import clean_pred_for_submit

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent))

REQUIRED_KEYS = ["index", "instruction", "image", "answer", "pred"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no} JSON error: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def get_id(ex: dict[str, Any], idx: int) -> str:
    for key in ["id", "_id", "qid", "question_id", "task_id", "index"]:
        if key in ex and ex[key] is not None:
            return str(ex[key])
    return str(idx)


def get_instruction(ex: dict[str, Any]) -> str:
    for key in ["instruction", "question", "query", "input", "text"]:
        if ex.get(key):
            return str(ex[key])
    return ""


def get_answer(ex: dict[str, Any]) -> str:
    for key in ["answer", "gold", "label", "target", "output"]:
        if key in ex and ex[key] is not None:
            return str(ex[key])
    return ""


def get_image(ex: dict[str, Any]) -> str:
    for key in ["image", "image_path", "image_url"]:
        if key in ex and ex[key] is not None:
            return str(ex[key])
    return ""


def clean_pred(raw: Any, question: str = "") -> str:
    return clean_pred_for_submit(raw, question)


def read_final_assistant(path: Path) -> str:
    rows = load_jsonl(path)
    assistants = [
        row for row in rows
        if row.get("role") == "assistant" and str(row.get("content", "") or "").strip()
    ]
    if not assistants:
        return ""
    return str(assistants[-1].get("content", "") or "")


def index_result_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        task_id = row.get("task_id")
        if task_id is not None:
            out[str(task_id)] = row
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_split(
    name: str,
    dataset: list[dict[str, Any]],
    results_path: Path,
    traj_dir: Path,
    out_dir: Path,
    limit: int,
) -> dict[str, Any]:
    result_by_task = index_result_rows(results_path)
    traj_files = sorted(traj_dir.glob("*.jsonl")) if traj_dir.exists() else []
    traj_pred = {path.stem: read_final_assistant(path) for path in traj_files}
    result_metrics = summarize_result_metrics(result_by_task)

    raw_rows = []
    final_rows = []
    missing_pred = []

    for idx, ex in enumerate(dataset[:limit]):
        task_id = get_id(ex, idx)
        raw_pred = traj_pred.get(task_id, "")
        if not raw_pred and task_id in result_by_task:
            raw_pred = str(result_by_task[task_id].get("prediction", "") or "")

        if not raw_pred:
            missing_pred.append({"index": idx, "id": task_id})

        base = {
            "index": idx,
            "instruction": get_instruction(ex),
            "image": get_image(ex),
            "answer": get_answer(ex),
        }
        raw_rows.append({**base, "pred": raw_pred})
        final_rows.append({**base, "pred": clean_pred(raw_pred, base["instruction"])})

    raw_path = out_dir / f"2wiki_{name}_raw_results.jsonl"
    final_path = out_dir / f"2wiki_{name}_final_results.jsonl"
    merged_traj_path = out_dir / f"2wiki_{name}_trajectories.jsonl"
    write_jsonl(raw_path, raw_rows)
    write_jsonl(final_path, final_rows)
    merge_trajectories(traj_files, merged_traj_path)

    raw_check = check_result_format(raw_path)
    final_check = check_result_format(final_path)
    return {
        "name": name,
        "results_path": str(results_path),
        "trajectory_dir": str(traj_dir),
        "raw_result_file": str(raw_path),
        "final_result_file": str(final_path),
        "merged_trajectory_file": str(merged_traj_path),
        "result_rows": len(result_by_task),
        "accuracy": result_metrics["accuracy"],
        "avg_steps": result_metrics["avg_steps"],
        "trajectory_file_count": len(traj_files),
        "raw_result_lines": raw_check["lines"],
        "final_result_lines": final_check["lines"],
        "merged_trajectory_lines": count_lines(merged_traj_path),
        "raw_format_errors": raw_check["errors"][:20],
        "final_format_errors": final_check["errors"][:20],
        "missing_prediction_count": len(missing_pred),
        "missing_prediction_examples": missing_pred[:10],
    }


def summarize_result_metrics(result_by_task: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(result_by_task.values())
    if not rows:
        return {"accuracy": None, "avg_steps": None}
    correct = [bool(row.get("correct")) for row in rows if "correct" in row]
    steps = []
    for row in rows:
        try:
            steps.append(float(row.get("steps")))
        except (TypeError, ValueError):
            pass
    accuracy = round(sum(correct) / len(correct), 4) if correct else None
    avg_steps = round(sum(steps) / len(steps), 3) if steps else None
    return {"accuracy": accuracy, "avg_steps": avg_steps}


def merge_trajectories(files: list[Path], out: Path) -> None:
    with open(out, "w", encoding="utf-8") as w:
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    obj = json.loads(s)
                    w.write(json.dumps(obj, ensure_ascii=False) + "\n")


def check_result_format(path: Path) -> dict[str, Any]:
    errors = []
    lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            lines += 1
            obj = json.loads(line)
            if list(obj.keys()) != REQUIRED_KEYS:
                errors.append({"line": line_no, "keys": list(obj.keys())})
    return {"lines": lines, "errors": errors}


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PROJECT_ROOT.parent / "datasets" / "2wiki.jsonl"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--baseline-results", default="results/baseline_2wiki_full.jsonl")
    parser.add_argument("--baseline-traj-dir", default="trajectories_baseline_2wiki_full")
    parser.add_argument("--reflection-results", default="results/reflection_2wiki_full.jsonl")
    parser.add_argument("--reflection-traj-dir", default="trajectories_reflection_2wiki_full")
    parser.add_argument("--out-dir", default="submit_2wiki")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_jsonl(data_path)

    baseline = build_split(
        "baseline",
        dataset,
        Path(args.baseline_results),
        Path(args.baseline_traj_dir),
        out_dir,
        args.limit,
    )
    reflection = build_split(
        "reflection",
        dataset,
        Path(args.reflection_results),
        Path(args.reflection_traj_dir),
        out_dir,
        args.limit,
    )

    report = {
        "dataset_path": str(data_path),
        "limit": args.limit,
        "required_keys": REQUIRED_KEYS,
        "baseline": baseline,
        "reflection": reflection,
        "pass": (
            baseline["raw_result_lines"] == args.limit
            and baseline["final_result_lines"] == args.limit
            and reflection["raw_result_lines"] == args.limit
            and reflection["final_result_lines"] == args.limit
            and not baseline["raw_format_errors"]
            and not baseline["final_format_errors"]
            and not reflection["raw_format_errors"]
            and not reflection["final_format_errors"]
        ),
    }
    report_path = out_dir / "2wiki_compare_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
