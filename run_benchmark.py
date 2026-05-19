"""
Benchmark Runner — 打榜数据集
==============================
输出格式：
  结果: {"index":, "instruction":, "image":, "answer":}
  轨迹: 所有题目的 trajectory 拼接成一个 JSONL

Usage:
    python run_benchmark.py --group 7
    python run_benchmark.py --group 7 --start 0 --end 3
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import zipfile
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

csv.field_size_limit(sys.maxsize)

def _import_runner(mode: str):
    if mode == "plan_react":
        from task_runner_plan_react import run_task
    elif mode == "plan_react_negcrit":
        from task_runner_plan_react_negcrit import run_task
    else:
        from task_runner import run_task
    return run_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("eval.benchmark")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "../datasets/benchmark.csv")


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
    mode: str = "basic",
    concurrency: int = 3,
):
    run_task = _import_runner(mode)
    end = end or len(dataset)
    subset = dataset[start:end]
    logger.info("Running benchmark [mode=%s]: %d items [%d:%d]", mode, len(subset), start, end)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(output_dir, "benchmark", mode, timestamp)
    traj_dir = os.path.join(traj_dir, mode, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    progress_path = os.path.join(output_dir, f"group_{group_id}_progress.jsonl")

    # Load existing progress for resume
    done_indices = set()
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done_indices.add(rec["index"])
        logger.info("Resumed: %d items already done", len(done_indices))

    # Lock for thread-safe progress file writing
    progress_lock = threading.Lock()

    def _run_one(idx, item):
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
        elapsed = time.time() - t0

        with progress_lock:
            progress_rec = {"index": idx, "answer": answer}
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(progress_rec, ensure_ascii=False) + "\n")

        logger.info("  [%d] => answer=%s  %.1fs", idx, answer[:80], elapsed)
        return idx, answer

    # Build work items
    work_items = []
    for i, item in enumerate(subset):
        idx = start + i
        if idx in done_indices:
            logger.info("[%d/%d] Already done, skipping", idx, end)
            continue
        work_items.append((idx, item))

    # Execute concurrently
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_run_one, idx, item) for idx, item in work_items]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.error("Unexpected error: %s", e)

    # --- Assemble final outputs ---
    # Reload all progress
    all_answers = {}
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    all_answers[rec["index"]] = rec["answer"]

    # 1. Answer CSV: same format as benchmark.csv (problem, image, answer)
    csv_path = os.path.join(output_dir, f"group_{group_id}.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["problem", "image", "answer"])
        writer.writeheader()
        for j in range(len(dataset)):
            writer.writerow({
                "problem": dataset[j]["problem"],
                "image": dataset[j].get("image", "").strip() or "",
                "answer": all_answers.get(j, ""),
            })
    logger.info("CSV saved to %s", csv_path)

    # 2. Trajectory JSON: concatenate all bench_XXX.jsonl (one entry per line)
    json_path = os.path.join(output_dir, f"group_{group_id}.json")
    with open(json_path, "w", encoding="utf-8") as out:
        for j in range(len(dataset)):
            tf = os.path.join(traj_dir, f"bench_{j:03d}.jsonl")
            if os.path.exists(tf):
                with open(tf, "r", encoding="utf-8") as inp:
                    for line in inp:
                        if line.strip():
                            out.write(line)
    logger.info("Trajectory JSON saved to %s", json_path)

    # 3. Zip: group_{group_id}.zip containing both files
    zip_path = os.path.join(output_dir, f"group_{group_id}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, f"group_{group_id}.csv")
        zf.write(json_path, f"group_{group_id}.json")
    logger.info("Zip saved to %s", zip_path)

    return zip_path


def main():
    p = argparse.ArgumentParser(description="Benchmark runner (打榜数据集)")
    p.add_argument("--group", "-g", default="7", help="Group ID")
    p.add_argument("--dataset", default=DATASET_PATH)
    p.add_argument("--output-dir", "-o", default="results")
    p.add_argument("--traj-dir", default="trajectories/benchmark")
    p.add_argument("--mode", "-m", choices=["basic", "plan_react", "plan_react_negcrit"], default="basic",
                   help="Runner mode: basic | plan_react | plan_react_negcrit")
    p.add_argument("--concurrency", "-c", type=int, default=2,
                   help="Number of questions to run concurrently (default: 2)")
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
        mode=args.mode,
        concurrency=args.concurrency,
    )
    print(f"\nDone! Submission: {zip_path}")


if __name__ == "__main__":
    main()
