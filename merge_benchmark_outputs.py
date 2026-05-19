"""
Merge multiple benchmark result directories into one ordered official output.

Inputs are directories produced by run_benchmark.py or run_benchmark_parallel.py.
The merge is keyed by benchmark index. Later --result-dirs override earlier
ones, which makes it convenient to combine a partial run with faster reruns
for missing cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from run_benchmark import BENCHMARK_DEFAULT, clean_pred, load_benchmark


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
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


def compact_ranges(values: list[int]) -> list[str]:
    if not values:
        return []
    values = sorted(values)
    out = []
    start = end = values[0]
    for value in values[1:]:
        if value == end + 1:
            end = value
        else:
            out.append(str(start) if start == end else f"{start}-{end}")
            start = end = value
    out.append(str(start) if start == end else f"{start}-{end}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="7")
    parser.add_argument("--dataset", default=str(BENCHMARK_DEFAULT))
    parser.add_argument("--result-dirs", nargs="+", required=True)
    parser.add_argument("--traj-dirs", nargs="*", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-traj-dir", required=True)
    args = parser.parse_args()

    dataset = load_benchmark(args.dataset)
    group = args.group
    rows_by_index: dict[int, dict[str, Any]] = {}

    for result_dir in args.result_dirs:
        path = Path(result_dir) / f"group_{group}_progress.jsonl"
        for row in read_jsonl(path):
            if row.get("index") is None:
                continue
            rows_by_index[int(row["index"])] = row

    out_dir = Path(args.out_dir)
    out_traj_dir = Path(args.out_traj_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_traj_dir.mkdir(parents=True, exist_ok=True)

    # Copy per-task trajectories. Later trajectory dirs override earlier ones.
    for traj_dir in args.traj_dirs:
        src = Path(traj_dir)
        if not src.exists():
            continue
        for path in sorted(src.glob("*.jsonl")):
            shutil.copy2(path, out_traj_dir / path.name)

    completed = sorted(rows_by_index)
    missing = [idx for idx in range(len(dataset)) if idx not in rows_by_index]

    progress_rows = [rows_by_index[idx] for idx in completed]
    write_jsonl(out_dir / f"group_{group}_progress.jsonl", progress_rows)

    entries = []
    for idx in completed:
        row = rows_by_index[idx]
        item = dataset[idx]
        traj = row.get("trajectory", [])
        entries.append(
            {
                "index": idx,
                "problem": item.get("problem", ""),
                "has_image": bool((item.get("image", "") or "").strip()),
                "answer": row.get("answer", ""),
                "raw_answer": row.get("raw_answer", ""),
                "steps": max([int(x.get("step_id", 0)) for x in traj] or [-1]),
                "trajectory": traj,
            }
        )

    json_path = out_dir / f"group_{group}.json"
    json_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    raw_rows = []
    final_rows = []
    csv_path = out_dir / f"group_{group}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["problem", "image", "answer"])
        writer.writeheader()
        for idx, item in enumerate(dataset):
            merged = rows_by_index.get(idx, {})
            raw = str(merged.get("raw_answer", ""))
            final = clean_pred(raw, item.get("problem", "")) if raw else str(merged.get("answer", ""))
            writer.writerow({"problem": item.get("problem", ""), "image": item.get("image", ""), "answer": final})
            base = {
                "index": idx,
                "instruction": item.get("problem", ""),
                "image": item.get("image", ""),
                "answer": item.get("answer", ""),
            }
            raw_rows.append({**base, "pred": raw})
            final_rows.append({**base, "pred": clean_pred(raw, item.get("problem", ""))})

    write_jsonl(out_dir / f"group_{group}_raw_results.jsonl", raw_rows)
    write_jsonl(out_dir / f"group_{group}_final_results.jsonl", final_rows)

    zip_path = out_dir / f"group_{group}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, f"group_{group}.json")
        zf.write(csv_path, f"group_{group}.csv")

    report = {
        "completed_count": len(completed),
        "missing_count": len(missing),
        "completed_ranges": compact_ranges(completed),
        "missing_ranges": compact_ranges(missing),
        "completed_indices": completed,
        "missing_indices": missing,
        "files": {
            "json": str(json_path),
            "csv": str(csv_path),
            "zip": str(zip_path),
            "progress": str(out_dir / f"group_{group}_progress.jsonl"),
            "raw_results": str(out_dir / f"group_{group}_raw_results.jsonl"),
            "final_results": str(out_dir / f"group_{group}_final_results.jsonl"),
            "trajectories": str(out_traj_dir),
        },
    }
    report_path = out_dir / "merge_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
