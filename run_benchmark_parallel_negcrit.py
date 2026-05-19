"""
Benchmark parallel launcher for neg-critic mode.

This wrapper fixes mode=plan_react_negcrit and reuses run_benchmark.py pipeline.
"""

from __future__ import annotations

import argparse

from run_benchmark import DATASET_PATH, load_benchmark, run_benchmark


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark parallel runner (plan_react_negcrit)")
    p.add_argument("--group", "-g", default="7", help="Group ID")
    p.add_argument("--dataset", default=DATASET_PATH)
    p.add_argument("--output-dir", "-o", default="results")
    p.add_argument("--traj-dir", default="trajectories/benchmark")
    p.add_argument("--concurrency", "-c", type=int, default=3)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    args = p.parse_args()

    dataset = load_benchmark(args.dataset)
    zip_path = run_benchmark(
        dataset=dataset,
        group_id=args.group,
        output_dir=args.output_dir,
        traj_dir=args.traj_dir,
        start=args.start,
        end=int(args.end) if args.end is not None else len(dataset),
        mode="plan_react_negcrit",
        concurrency=args.concurrency,
    )
    print(f"\nDone! Submission: {zip_path}")


if __name__ == "__main__":
    main()
