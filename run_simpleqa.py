"""
SimpleVQA Batch Evaluation Runner
=================================
Reads SimpleVQA.jsonl, runs each question through the agent, and scores results.

Usage:
    python run_simpleqa.py \
        --dataset /inspire/qb-ilm2/project/26summer-camp-01/public/datasets/simpleVQA/SimpleVQA.jsonl \
        --image-dir /inspire/qb-ilm2/project/26summer-camp-01/public/datasets/simpleVQA \
        --output results/simpleqa_results.jsonl \
        --start 0 --end 99
"""

import argparse
import base64
import json
import logging
import os
import re
import time
from pathlib import Path

from task_runner import run_task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("eval.simpleqa")

DATASET_DEFAULT = "/inspire/qb-ilm2/project/26summer-camp-01/public/datasets/simpleVQA/SimpleVQA.jsonl"
IMAGE_DIR_DEFAULT = "/inspire/qb-ilm2/project/26summer-camp-01/public/datasets/simpleVQA"


def load_dataset(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def normalize(text: str) -> str:
    """Normalize answer text for comparison."""
    text = text.strip().lower()
    text = re.sub(r"[。，、；：！？\.\,\;\:\!\?\s]+$", "", text)
    return text


def check_answer(prediction: str, ground_truth: str) -> bool:
    """Check if prediction contains the ground truth answer."""
    pred_norm = normalize(prediction)
    gt_norm = normalize(ground_truth)
    if not gt_norm:
        return False
    return gt_norm in pred_norm


def run_batch(
    dataset: list[dict],
    image_dir: str,
    output_path: str,
    start: int = 0,
    end: int | None = None,
    traj_dir: str = "trajectories/simpleqa",
):
    end = end or len(dataset)
    subset = dataset[start:end]
    logger.info("Running SimpleVQA eval: %d items [%d:%d]", len(subset), start, end)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    results = []
    correct = 0
    total = 0

    for i, item in enumerate(subset):
        idx = start + i
        data_id = item["data_id"]
        question = item["question"]
        answer = item["answer"]
        image_rel = item.get("image", "")
        image_url = item.get("image_url", "")

        logger.info("[%d/%d] data_id=%s q=%s", idx, end, data_id, question[:60])

        image_b64 = None
        image_path = os.path.join(image_dir, image_rel) if image_rel else ""
        if image_path and os.path.isfile(image_path):
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
        else:
            logger.warning("Image not found: %s", image_path)

        task = {
            "id": f"simpleqa_{data_id}",
            "instruction": question,
            "image_b64": image_b64,
            "image_url": image_url,
        }

        t0 = time.time()
        try:
            result = run_task(task, trajectory_dir=traj_dir)
            prediction = result.get("answer", "")
        except Exception as e:
            logger.error("run_task failed for data_id=%s: %s", data_id, e)
            prediction = ""
            result = {"error": str(e)}
        elapsed = time.time() - t0

        is_correct = check_answer(prediction, answer)
        if is_correct:
            correct += 1
        total += 1

        record = {
            "data_id": data_id,
            "question": question,
            "ground_truth": answer,
            "prediction": prediction,
            "correct": is_correct,
            "elapsed_s": round(elapsed, 1),
            "steps": result.get("steps", -1),
        }
        results.append(record)

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        acc = correct / total * 100
        logger.info(
            "  => correct=%s  pred=%s  gt=%s  acc=%.1f%% (%d/%d)  %.1fs",
            is_correct, prediction[:50], answer, acc, correct, total, elapsed,
        )

    logger.info("=== DONE === Accuracy: %.2f%% (%d/%d)", correct/total*100, correct, total)
    return results


def main():
    p = argparse.ArgumentParser(description="Batch eval for SimpleVQA")
    p.add_argument("--dataset", default=DATASET_DEFAULT)
    p.add_argument("--image-dir", default=IMAGE_DIR_DEFAULT)
    p.add_argument("--output", "-o", default="results/simpleqa_results.jsonl")
    p.add_argument("--traj-dir", default="trajectories/simpleqa")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    args = p.parse_args()

    dataset = load_dataset(args.dataset)
    logger.info("Loaded %d items from %s", len(dataset), args.dataset)

    run_batch(
        dataset=dataset,
        image_dir=args.image_dir,
        output_path=args.output,
        start=args.start,
        end=int(args.end) if args.end else len(dataset),
        traj_dir=args.traj_dir,
    )


if __name__ == "__main__":
    main()
