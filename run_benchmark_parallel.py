"""
Parallel benchmark runner.

Each worker writes to an isolated shard directory, then this script merges
progress, official files, normalized jsonl results, and trajectories. This
avoids corrupting group_7_progress.jsonl with concurrent appends.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

from run_benchmark import BENCHMARK_DEFAULT, check_gpu, check_services, clean_pred, load_benchmark


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


def make_ranges(start: int, end: int, workers: int) -> list[tuple[int, int]]:
    n = max(0, end - start)
    workers = max(1, min(workers, n or 1))
    ranges = []
    for i in range(workers):
        a = start + (n * i) // workers
        b = start + (n * (i + 1)) // workers
        if a < b:
            ranges.append((a, b))
    return ranges


def merge_outputs(
    dataset: list[dict[str, Any]],
    group: str,
    output_dir: Path,
    traj_dir: Path,
    shard_root: Path,
    ranges: list[tuple[int, int]],
) -> None:
    progress_rows_by_index: dict[int, dict[str, Any]] = {}
    merged_trajectories: list[dict[str, Any]] = []

    traj_dir.mkdir(parents=True, exist_ok=True)
    for shard_id, _ in enumerate(ranges):
        shard_out = shard_root / f"shard_{shard_id}" / "results"
        shard_traj = shard_root / f"shard_{shard_id}" / "trajectories"
        for row in read_jsonl(shard_out / f"group_{group}_progress.jsonl"):
            idx = int(row["index"])
            progress_rows_by_index[idx] = row
        if shard_traj.exists():
            for path in sorted(shard_traj.glob("*.jsonl")):
                dst = traj_dir / path.name
                shutil.copy2(path, dst)

    for idx in sorted(progress_rows_by_index):
        row = progress_rows_by_index[idx]
        problem = dataset[idx].get("problem", "")
        raw_answer = row.get("raw_answer", row.get("answer", ""))
        final_answer = clean_pred(raw_answer, problem)
        entry = {
            "index": idx,
            "problem": problem,
            "has_image": bool((dataset[idx].get("image", "") or "").strip()),
            "answer": final_answer,
            "raw_answer": raw_answer,
            "steps": max([int(x.get("step_id", 0)) for x in row.get("trajectory", [])] or [-1]),
            "trajectory": row.get("trajectory", []),
        }
        merged_trajectories.append(entry)

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / f"group_{group}_progress.jsonl"
    write_jsonl(progress_path, [progress_rows_by_index[i] for i in sorted(progress_rows_by_index)])

    json_path = output_dir / f"group_{group}.json"
    json_path.write_text(json.dumps(merged_trajectories, ensure_ascii=False, indent=2), encoding="utf-8")

    answers_by_idx = {
        idx: clean_pred(row.get("raw_answer", row.get("answer", "")), dataset[idx].get("problem", ""))
        for idx, row in progress_rows_by_index.items()
    }
    csv_path = output_dir / f"group_{group}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["problem", "image", "answer"])
        writer.writeheader()
        for idx, row in enumerate(dataset):
            writer.writerow({
                "problem": row.get("problem", ""),
                "image": row.get("image", ""),
                "answer": answers_by_idx.get(idx, ""),
            })

    raw_rows = []
    final_rows = []
    raw_by_idx = {idx: str(row.get("raw_answer", "")) for idx, row in progress_rows_by_index.items()}
    for idx, row in enumerate(dataset):
        raw = raw_by_idx.get(idx, "")
        base = {
            "index": idx,
            "instruction": row.get("problem", ""),
            "image": row.get("image", ""),
            "answer": row.get("answer", ""),
        }
        raw_rows.append({**base, "pred": raw})
        final_rows.append({**base, "pred": clean_pred(raw, row.get("problem", ""))})
    write_jsonl(output_dir / f"group_{group}_raw_results.jsonl", raw_rows)
    write_jsonl(output_dir / f"group_{group}_final_results.jsonl", final_rows)

    zip_path = output_dir / f"group_{group}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, f"group_{group}.json")
        zf.write(csv_path, f"group_{group}.csv")

    print(json.dumps({
        "merged_completed": len(progress_rows_by_index),
        "progress": str(progress_path),
        "json": str(json_path),
        "csv": str(csv_path),
        "zip": str(zip_path),
        "raw_results": str(output_dir / f"group_{group}_raw_results.jsonl"),
        "final_results": str(output_dir / f"group_{group}_final_results.jsonl"),
        "trajectories": str(traj_dir),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel benchmark runner with shard-safe outputs.")
    parser.add_argument("--group", "-g", required=True)
    parser.add_argument("--dataset", default=str(BENCHMARK_DEFAULT))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default="results_parallel")
    parser.add_argument("--traj-dir", default="trajectories/benchmark_parallel")
    parser.add_argument("--skip-env-check", action="store_true")
    args = parser.parse_args()

    if not args.skip_env_check:
        check_gpu()
        check_services(os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))

    dataset = load_benchmark(args.dataset)
    end = min(args.end if args.end is not None else len(dataset), len(dataset))
    ranges = make_ranges(args.start, end, args.workers)
    if not ranges:
        raise RuntimeError("empty benchmark range")

    output_dir = Path(args.output_dir)
    traj_dir = Path(args.traj_dir)
    shard_root = output_dir / "_shards" / f"group_{args.group}_{args.start}_{end}_{int(time.time())}"
    shard_root.mkdir(parents=True, exist_ok=True)

    print("Parallel ranges:", ranges)
    procs: list[tuple[int, tuple[int, int], subprocess.Popen]] = []
    for shard_id, (a, b) in enumerate(ranges):
        shard_base = shard_root / f"shard_{shard_id}"
        shard_out = shard_base / "results"
        shard_traj = shard_base / "trajectories"
        shard_out.mkdir(parents=True, exist_ok=True)
        shard_traj.mkdir(parents=True, exist_ok=True)
        log_path = shard_base / "run.log"
        cmd = [
            sys.executable,
            "run_benchmark.py",
            "--group",
            args.group,
            "--dataset",
            str(args.dataset),
            "--start",
            str(a),
            "--end",
            str(b),
            "--output-dir",
            str(shard_out),
            "--traj-dir",
            str(shard_traj),
            "--skip-env-check",
        ]
        if args.max_steps is not None:
            cmd.extend(["--max-steps", str(args.max_steps)])
        log_f = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parent, env=os.environ.copy())
        procs.append((shard_id, (a, b), proc))
        print(f"started shard_{shard_id} range=[{a},{b}) pid={proc.pid} log={log_path}")

    failed = []
    for shard_id, r, proc in procs:
        rc = proc.wait()
        if rc != 0:
            failed.append({"shard": shard_id, "range": r, "returncode": rc})
    if failed:
        print(json.dumps({"failed": failed, "shard_root": str(shard_root)}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(1)

    merge_outputs(dataset, args.group, output_dir, traj_dir, shard_root, ranges)


if __name__ == "__main__":
    main()
