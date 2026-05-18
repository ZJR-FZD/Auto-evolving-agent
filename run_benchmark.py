"""
Benchmark Runner for Competition
=================================
Reads benchmark.csv, runs each question through the agent,
saves trajectories to group_{}.json and answers to group_{}.csv,
then zips both files.

Usage:
    python run_benchmark.py --group 1
    python run_benchmark.py --group 1 --start 0 --end 10   # partial run
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import zipfile
from pathlib import Path

csv.field_size_limit(sys.maxsize)

from task_runner import run_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("eval.benchmark")

DATASET_PATH = "/inspire/qb-ilm2/project/26summer-camp-01/26210094/datasets/benchmark.csv"


def load_benchmark(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)
    return items


def run_benchmark(
    dataset: list[dict],
    group_id: str,
    output_dir: str = "results",
    traj_dir: str = "trajectories/benchmark",
    start: int = 0,
    end: int | None = None,
):
    end = end or len(dataset)
    subset = dataset[start:end]
    logger.info("Running benchmark: %d items [%d:%d]", len(subset), start, end)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"group_{group_id}.json")
    csv_path = os.path.join(output_dir, f"group_{group_id}.csv")
    zip_path = os.path.join(output_dir, f"group_{group_id}.zip")

    # Load existing progress if resuming
    all_trajectories = []
    answers = [""] * len(dataset)
    progress_path = os.path.join(output_dir, f"group_{group_id}_progress.jsonl")

    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                idx = rec["index"]
                answers[idx] = rec["answer"]
                all_trajectories.append(rec["trajectory"])
        logger.info("Resumed from progress: %d items done", len(all_trajectories))

    done_indices = set()
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                done_indices.add(rec["index"])

    for i, item in enumerate(subset):
        idx = start + i
        if idx in done_indices:
            logger.info("[%d/%d] Already done, skipping", idx, end)
            continue

        problem = item["problem"]
        image_b64 = item.get("image", "").strip() or None

        logger.info("[%d/%d] %s", idx, end, problem[:80])

        task = {
            "id": f"bench_{idx:03d}",
            "instruction": problem,
            "image_b64": image_b64,
            "image_url": None,
        }

        t0 = time.time()
        try:
            result = run_task(task, trajectory_dir=traj_dir)
            answer = result.get("answer", "")
        except Exception as e:
            logger.error("run_task failed for idx=%d: %s", idx, e)
            answer = ""
            result = {"error": str(e)}
        elapsed = time.time() - t0

        # Read trajectory file for this task
        traj_file = os.path.join(traj_dir, f"bench_{idx:03d}.jsonl")
        traj_data = []
        if os.path.exists(traj_file):
            with open(traj_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        traj_data.append(json.loads(line))

        trajectory_entry = {
            "index": idx,
            "problem": problem,
            "has_image": image_b64 is not None,
            "answer": answer,
            "steps": result.get("steps", -1),
            "elapsed_s": round(elapsed, 1),
            "trajectory": traj_data,
        }

        answers[idx] = answer
        all_trajectories.append(trajectory_entry)

        # Save progress incrementally
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "index": idx,
                "answer": answer,
                "trajectory": traj_data,
            }, ensure_ascii=False) + "\n")

        logger.info("  => answer=%s  %.1fs", answer[:80], elapsed)

    # --- Write final outputs ---
    # 1. group_{}.json - all trajectories
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_trajectories, f, ensure_ascii=False, indent=2)
    logger.info("Trajectories saved to %s", json_path)

    # 2. group_{}.csv - answers
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["problem", "image", "answer"])
        writer.writeheader()
        for j, row in enumerate(dataset):
            writer.writerow({
                "problem": row["problem"],
                "image": row.get("image", ""),
                "answer": answers[j],
            })
    logger.info("Answers saved to %s", csv_path)

    # 3. group_{}.zip
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, f"group_{group_id}.json")
        zf.write(csv_path, f"group_{group_id}.csv")
    logger.info("Zip saved to %s", zip_path)

    return zip_path


def main():
    p = argparse.ArgumentParser(description="Benchmark runner for competition")
    p.add_argument("--group", "-g", required=True, help="Group ID number")
    p.add_argument("--dataset", default=DATASET_PATH)
    p.add_argument("--output-dir", "-o", default="results")
    p.add_argument("--traj-dir", default="trajectories/benchmark")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    args = p.parse_args()

    dataset = load_benchmark(args.dataset)
    logger.info("Loaded %d items from %s", len(dataset), args.dataset)

    zip_path = run_benchmark(
        dataset=dataset,
        group_id=args.group,
        output_dir=args.output_dir,
        traj_dir=args.traj_dir,
        start=args.start,
        end=int(args.end) if args.end is not None else len(dataset),
    )
    print(f"\nDone! Submission file: {zip_path}")


if __name__ == "__main__":
    main()
