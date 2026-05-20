"""Benchmark runner that isolates text and image task strategies.

Items before --image-start use plan_react_negcrit. Items from --image-start
onward use the older plan_react runner so TPGO/negative-critic changes do not
affect image tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from run_benchmark import DATASET_PATH, load_benchmark


csv.field_size_limit(sys.maxsize)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("eval.benchmark.mixed")


def _clean_answer(answer: str) -> str:
    """Normalize only output wrappers, leaving the answer content intact."""
    text = str(answer or "").strip()
    match = re.search(r"<answer(?:\s+[^>]*)?>(.*?)</answer>", text, flags=re.I | re.S)
    if match:
        text = match.group(1).strip()
    text = re.sub(r"^\[LOW_CONFIDENCE\]\s*", "", text, flags=re.I).strip()
    return re.sub(r"\s+", " ", text).strip()


def _runner_for_index(idx: int, image_start: int):
    """Select the runner for a benchmark index."""
    if idx >= image_start:
        from task_runner_plan_react import run_task

        return run_task, "plan_react"
    from task_runner_plan_react_negcrit import run_task

    return run_task, "plan_react_negcrit"


def run_mixed_benchmark(
    dataset: list[dict],
    group_id: str,
    output_dir: str,
    traj_dir: str,
    start: int,
    end: int,
    concurrency: int,
    image_start: int,
) -> str:
    """Run benchmark with text/image runner isolation and write submission zip."""
    subset = dataset[start:end]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "mixed_negcrit_text_planreact_image"
    output_dir = os.path.join(output_dir, "benchmark", mode, timestamp)
    traj_dir = os.path.join(traj_dir, mode, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    progress_path = os.path.join(output_dir, f"group_{group_id}_progress.jsonl")
    progress_lock = threading.Lock()

    logger.info(
        "Running mixed benchmark: %d items [%d:%d], image_start=%d",
        len(subset),
        start,
        end,
        image_start,
    )

    def _run_one(idx: int, item: dict):
        problem = item["problem"]
        image_b64 = item.get("image", "").strip() or None
        run_task, runner_name = _runner_for_index(idx, image_start)
        runner_traj_dir = os.path.join(traj_dir, runner_name)
        os.makedirs(runner_traj_dir, exist_ok=True)

        task = {
            "id": f"bench_{idx:03d}",
            "instruction": problem,
            "image_b64": image_b64,
            "image_url": None,
        }
        logger.info("[%d/%d] runner=%s %s", idx, end, runner_name, problem[:80])
        t0 = time.time()
        try:
            result = run_task(task, trajectory_dir=runner_traj_dir)
            answer = _clean_answer(result.get("answer", ""))
        except Exception as exc:
            logger.error("run_task failed for idx=%d: %s", idx, exc)
            answer = ""
        elapsed = time.time() - t0

        with progress_lock:
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"index": idx, "answer": answer}, ensure_ascii=False) + "\n")
        logger.info("  [%d] => answer=%s  %.1fs", idx, answer[:80], elapsed)
        return idx, answer

    work_items = [(start + i, item) for i, item in enumerate(subset)]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_run_one, idx, item) for idx, item in work_items]
        for future in as_completed(futures):
            future.result()

    all_answers: dict[int, str] = {}
    with open(progress_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                all_answers[int(rec["index"])] = rec["answer"]

    csv_path = os.path.join(output_dir, f"group_{group_id}.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["problem", "image", "answer"])
        writer.writeheader()
        for j, item in enumerate(dataset):
            writer.writerow(
                {
                    "problem": item["problem"],
                    "image": item.get("image", "").strip() or "",
                    "answer": all_answers.get(j, ""),
                }
            )
    logger.info("CSV saved to %s", csv_path)

    json_path = os.path.join(output_dir, f"group_{group_id}.json")
    with open(json_path, "w", encoding="utf-8") as out:
        for runner_name in ("plan_react_negcrit", "plan_react"):
            runner_traj_dir = os.path.join(traj_dir, runner_name)
            for j in range(len(dataset)):
                tf = os.path.join(runner_traj_dir, f"bench_{j:03d}.jsonl")
                if os.path.exists(tf):
                    with open(tf, "r", encoding="utf-8") as inp:
                        for line in inp:
                            if line.strip():
                                out.write(line)
    logger.info("Trajectory JSON saved to %s", json_path)

    zip_path = os.path.join(output_dir, f"group_{group_id}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, f"group_{group_id}.csv")
        zf.write(json_path, f"group_{group_id}.json")
    logger.info("Zip saved to %s", zip_path)
    return zip_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Mixed benchmark runner")
    parser.add_argument("--group", "-g", default="7")
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--output-dir", "-o", default="results")
    parser.add_argument("--traj-dir", default="trajectories/benchmark")
    parser.add_argument("--concurrency", "-c", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--image-start", type=int, default=50)
    args = parser.parse_args()

    dataset = load_benchmark(args.dataset)
    zip_path = run_mixed_benchmark(
        dataset=dataset,
        group_id=args.group,
        output_dir=args.output_dir,
        traj_dir=args.traj_dir,
        start=args.start,
        end=int(args.end),
        concurrency=args.concurrency,
        image_start=args.image_start,
    )
    print(f"\nDone! Submission: {zip_path}")


if __name__ == "__main__":
    main()
