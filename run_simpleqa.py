"""
SimpleVQA Evaluation Runner — 评测集
=====================================
输出格式：
  结果: {"index":, "instruction":, "image":, "answer":, "pred":}
  轨迹: 所有题目的 trajectory 拼接成一个 JSONL

Usage:
    python run_simpleqa.py --group 7
    python run_simpleqa.py --group 7 --start 0 --end 5
"""

import argparse
import base64
import json
import logging
import os
import time
from datetime import datetime

def _import_runner(mode: str):
    if mode == "plan_react":
        from task_runner_plan_react import run_task
    else:
        from task_runner import run_task
    return run_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("eval.simpleqa")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "../datasets/simpleVQA/SimpleVQA.jsonl")
IMAGE_DIR = os.path.join(SCRIPT_DIR, "../datasets/simpleVQA")


def load_dataset(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def run_eval(
    dataset: list[dict],
    group_id: str,
    image_dir: str = IMAGE_DIR,
    output_dir: str = "results",
    traj_dir: str = "trajectories/simpleqa",
    start: int = 0,
    end: int | None = None,
    mode: str = "basic",
):
    run_task = _import_runner(mode)
    end = end or len(dataset)
    subset = dataset[start:end]
    logger.info("Running SimpleVQA eval [mode=%s]: %d items [%d:%d]", mode, len(subset), start, end)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(output_dir, "simpleqa", mode, timestamp)
    traj_dir = os.path.join(traj_dir, mode, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    progress_path = os.path.join(output_dir, f"group_{group_id}_progress.jsonl")

    result_path = os.path.join(output_dir, f"group_{group_id}_simpleqa.jsonl")
    traj_path = os.path.join(output_dir, f"group_{group_id}_simpleqa_traj.jsonl")

    # Load existing progress for resume
    done_indices = set()
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done_indices.add(rec["index"])
        logger.info("Resumed: %d items already done", len(done_indices))

    for i, item in enumerate(subset):
        idx = start + i
        if idx in done_indices:
            logger.info("[%d/%d] Already done, skipping", idx, end)
            continue

        question = item["question"]
        answer = item["answer"]
        image_rel = item.get("image", "")
        image_url = item.get("image_url", "")

        logger.info("[%d/%d] q=%s", idx, end, question[:60])

        # Load local image as base64
        image_b64 = None
        image_path = os.path.join(image_dir, image_rel) if image_rel else ""
        if image_path and os.path.isfile(image_path):
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()

        task = {
            "id": f"simpleqa_{idx:03d}",
            "instruction": question,
            "image_b64": image_b64,
            "image_url": image_url if image_url else None,
        }

        t0 = time.time()
        try:
            result = run_task(task, trajectory_dir=traj_dir)
            pred = result.get("answer", "")
        except Exception as e:
            logger.error("run_task failed for idx=%d: %s", idx, e)
            pred = ""
        elapsed = time.time() - t0

        # Save progress
        progress_rec = {"index": idx, "pred": pred, "answer": answer}
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(progress_rec, ensure_ascii=False) + "\n")

        logger.info("  => pred=%s  gt=%s  %.1fs", pred[:50], answer, elapsed)

    # --- Assemble final outputs ---
    all_preds = {}
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    all_preds[rec["index"]] = rec["pred"]

    # 1. Result JSONL: {"index":, "instruction":, "image":, "answer":, "pred":}
    with open(result_path, "w", encoding="utf-8") as f:
        for j, item in enumerate(dataset):
            rec = {
                "index": j,
                "instruction": item["question"],
                "image": item.get("image", ""),
                "answer": item["answer"],
                "pred": all_preds.get(j, ""),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Results saved to %s", result_path)

    # 2. Trajectory JSONL: concatenate all simpleqa_XXX.jsonl
    with open(traj_path, "w", encoding="utf-8") as out:
        for j in range(len(dataset)):
            tf = os.path.join(traj_dir, f"simpleqa_{j:03d}.jsonl")
            if os.path.exists(tf):
                with open(tf, "r", encoding="utf-8") as inp:
                    for line in inp:
                        if line.strip():
                            out.write(line)
    logger.info("Trajectories saved to %s", traj_path)


def main():
    p = argparse.ArgumentParser(description="SimpleVQA eval runner (评测集)")
    p.add_argument("--group", "-g", default="7")
    p.add_argument("--dataset", default=DATASET_PATH)
    p.add_argument("--image-dir", default=IMAGE_DIR)
    p.add_argument("--output-dir", "-o", default="results")
    p.add_argument("--traj-dir", default="trajectories/simpleqa")
    p.add_argument("--mode", "-m", choices=["basic", "plan_react"], default="basic",
                   help="Runner mode: basic (task_runner) or plan_react (task_runner_plan_react)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    args = p.parse_args()

    dataset = load_dataset(args.dataset)
    logger.info("Loaded %d items from %s", len(dataset), args.dataset)

    run_eval(
        dataset=dataset,
        group_id=args.group,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        traj_dir=args.traj_dir,
        start=args.start,
        end=int(args.end) if args.end is not None else len(dataset),
        mode=args.mode,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
