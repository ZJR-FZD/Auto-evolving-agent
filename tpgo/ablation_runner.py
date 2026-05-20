"""
Run small benchmark ablations and analyze the produced trajectories.

The script is designed for quick "does it run" validation before a full
submission run. It creates a small benchmark CSV from the first N rows,
runs selected harness modes with a hard timeout, then summarizes the
resulting trajectories through `tpgo_tools`.

Example:
    python -m tpgo.ablation_runner --dataset ..\\datasets\\benchmark.csv --end 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tpgo.tpgo_tools import analyze_trajectory, summarize_metrics, write_analysis_outputs


DEFAULT_MODES = ("basic", "plan_react", "plan_react_negcrit")


@dataclass
class ModeRunResult:
    mode: str
    status: str
    command: list[str]
    elapsed_seconds: float
    returncode: int | None
    stdout_tail: str
    stderr_tail: str
    output_dir: str
    trajectory_dir: str
    summary: dict[str, Any] | None
    error: str | None = None


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        fieldnames = reader.fieldnames or ["problem", "image", "answer"]
    return fieldnames, rows


def write_small_benchmark(source: Path, target: Path, start: int, end: int) -> int:
    fieldnames, rows = read_csv_rows(source)
    subset = rows[start:end]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in subset:
            writer.writerow(row)
    return len(subset)


def tail_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def find_latest_traj_dir(traj_root: Path, mode: str) -> Path | None:
    mode_root = traj_root / "benchmark" / mode
    if not mode_root.exists():
        return None
    dirs = [p for p in mode_root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def analyze_traj_dir(traj_dir: Path, out_dir: Path) -> dict[str, Any]:
    metrics = []
    memories = []
    for path in sorted(traj_dir.glob("*.jsonl")):
        metric, reflection = analyze_trajectory(path)
        metrics.append(metric)
        memories.extend(reflection)
    write_analysis_outputs(metrics, memories, out_dir)
    return summarize_metrics(metrics)


def run_mode(
    mode: str,
    dataset: Path,
    run_root: Path,
    timeout_seconds: int,
    concurrency: int,
    max_steps: int | None,
    group: str,
) -> ModeRunResult:
    output_root = run_root / "results"
    traj_root = run_root / "trajectories"
    output_root.mkdir(parents=True, exist_ok=True)
    traj_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "run_benchmark.py",
        "--group",
        group,
        "--dataset",
        str(dataset),
        "--output-dir",
        str(output_root),
        "--traj-dir",
        str(traj_root / "benchmark"),
        "--mode",
        mode,
        "--concurrency",
        str(concurrency),
    ]
    env = os.environ.copy()
    if max_steps is not None:
        env["MAX_STEPS"] = str(max_steps)

    start = time.time()
    stdout = ""
    stderr = ""
    returncode: int | None = None
    status = "unknown"
    error = None
    try:
        proc = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        returncode = proc.returncode
        status = "ok" if proc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
        status = "timeout"
        error = f"timed out after {timeout_seconds}s"
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    elapsed = round(time.time() - start, 3)
    latest_traj_dir = find_latest_traj_dir(traj_root, mode)
    summary = None
    if latest_traj_dir and any(latest_traj_dir.glob("*.jsonl")):
        summary = analyze_traj_dir(latest_traj_dir, run_root / "analysis" / mode)

    return ModeRunResult(
        mode=mode,
        status=status,
        command=cmd,
        elapsed_seconds=elapsed,
        returncode=returncode,
        stdout_tail=tail_text(stdout),
        stderr_tail=tail_text(stderr),
        output_dir=str(output_root),
        trajectory_dir=str(latest_traj_dir or ""),
        summary=summary,
        error=error,
    )


def write_report(results: list[ModeRunResult], run_root: Path) -> Path:
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "results": [asdict(r) for r in results],
    }
    path = run_root / "ablation_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def print_report(results: list[ModeRunResult], report_path: Path) -> None:
    print("\nTPGO benchmark ablation")
    print("=" * 70)
    for result in results:
        print(
            f"{result.mode}: status={result.status}, elapsed={result.elapsed_seconds}s, "
            f"returncode={result.returncode}, traj={result.trajectory_dir or 'none'}"
        )
        if result.error:
            print(f"  error: {result.error}")
        if result.summary:
            print(
                "  summary: "
                f"count={result.summary.get('count')}, "
                f"avg_steps={result.summary.get('avg_steps')}, "
                f"avg_search={result.summary.get('avg_search_calls')}, "
                f"avg_dup={result.summary.get('avg_duplicate_adjacent_queries')}, "
                f"avg_low_signal={result.summary.get('avg_low_signal_tool_results')}"
            )
        if result.status != "ok":
            if result.stderr_tail.strip():
                print("  stderr_tail:")
                print(indent(result.stderr_tail.strip(), "    "))
            elif result.stdout_tail.strip():
                print("  stdout_tail:")
                print(indent(result.stdout_tail.strip(), "    "))
    print(f"\nreport: {report_path}")


def indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark ablations with timeout and TPGO analysis")
    parser.add_argument("--dataset", default=str(Path("..") / "datasets" / "benchmark.csv"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES), choices=list(DEFAULT_MODES))
    parser.add_argument("--timeout-seconds", type=int, default=480)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional MAX_STEPS override for quick smoke runs")
    parser.add_argument("--group", default="7")
    parser.add_argument("--out-dir", default="tpgo/ablation_runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.out_dir) / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    if not dataset.exists():
        result = ModeRunResult(
            mode="dataset_check",
            status="failed",
            command=[],
            elapsed_seconds=0.0,
            returncode=None,
            stdout_tail="",
            stderr_tail="",
            output_dir=str(run_root),
            trajectory_dir="",
            summary=None,
            error=f"benchmark dataset not found: {dataset}",
        )
        report_path = write_report(
            [result],
            run_root,
        )
        print_report([result], report_path)
        raise SystemExit(2)

    small_dataset = run_root / f"benchmark_{args.start}_{args.end}.csv"
    count = write_small_benchmark(dataset, small_dataset, args.start, args.end)
    if count == 0:
        raise SystemExit(f"no rows selected from {dataset} with range [{args.start}:{args.end}]")

    print(f"small benchmark: {small_dataset} ({count} rows)")
    results = []
    for mode in args.modes:
        print(f"\n[RUN] mode={mode} timeout={args.timeout_seconds}s")
        result = run_mode(
            mode=mode,
            dataset=small_dataset,
            run_root=run_root,
            timeout_seconds=args.timeout_seconds,
            concurrency=args.concurrency,
            max_steps=args.max_steps,
            group=args.group,
        )
        results.append(result)
        print(f"[DONE] {mode}: {result.status} in {result.elapsed_seconds}s")
    report_path = write_report(results, run_root)
    print_report(results, report_path)


if __name__ == "__main__":
    main()
