"""
SimpleVQA Batch Evaluation Runner.

Gold answers are used only for offline accuracy statistics and are never sent
to task_runner or the reflection critic.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from answer_utils import clean_pred_for_submit

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("eval.simpleqa")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent))
DATASET_DEFAULT = PROJECT_ROOT.parent / "datasets" / "simpleVQA" / "SimpleVQA.jsonl"
IMAGE_DIR_DEFAULT = PROJECT_ROOT.parent / "datasets" / "simpleVQA"


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    text = str(text or "").strip().lower()
    import re

    return re.sub(r"[。，、；：！？\.\,\;\:\!\?\s]+$", "", text)


def check_answer(prediction: str, ground_truth: str) -> bool:
    gt = normalize(ground_truth)
    return bool(gt and gt in normalize(prediction))


def clean_pred(raw: Any, question: str = "") -> str:
    return clean_pred_for_submit(raw, question)


def check_gpu() -> None:
    try:
        proc = subprocess.run(["nvidia-smi"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"nvidia-smi is not executable here ({exc}). Please run SimpleVQA on the GPU instance.") from None
    smi = proc.stdout or ""
    print(smi)
    if proc.returncode != 0 or "CUDA Version" not in smi:
        raise RuntimeError("GPU/CUDA is not visible. Refusing to run SimpleVQA.")
    if "NVIDIA H200" not in smi:
        logger.warning("CUDA is visible, but this does not look like the expected H200 GPU instance.")
    if not any(name in smi.lower() for name in ("sglang", "vllm", "python")):
        logger.warning(
            "No obvious local model-serving process is shown by nvidia-smi. "
            "run_simpleqa.py is only an API client; the GPU should be used by "
            "the service behind LLM_BASE_URL."
        )
    else:
        logger.info("GPU is visible and nvidia-smi shows a local process using it.")


def load_done(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("data_id") is not None:
                done.add(str(obj["data_id"]))
    return done


def encode_image(path: Path) -> str | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def run_batch(
    dataset: list[dict[str, Any]],
    image_dir: str | Path,
    output_path: str | Path,
    start: int = 0,
    end: int | None = None,
    traj_dir: str = "trajectories/simpleqa",
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    end = min(end or len(dataset), len(dataset))
    subset = dataset[start:end]
    output = Path(output_path)
    image_root = Path(image_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    Path(traj_dir).mkdir(parents=True, exist_ok=True)

    done = set() if overwrite else load_done(output)
    if overwrite and output.exists():
        output.unlink()

    raw_path = output.parent / "simpleqa_raw_results.jsonl"
    final_path = output.parent / "simpleqa_final_results.jsonl"
    if overwrite:
        for path in [raw_path, final_path]:
            if path.exists():
                path.unlink()

    logger.info("Running SimpleVQA eval: %d items [%d:%d], resume=%d", len(subset), start, end, len(done))
    from task_runner import run_task  # noqa: PLC0415
    results = []
    correct = 0
    total = 0

    for i, item in enumerate(subset):
        idx = start + i
        data_id = str(item.get("data_id") or item.get("id") or idx)
        if data_id in done:
            logger.info("[%d/%d] data_id=%s already done, skipping", idx, end, data_id)
            continue

        question = str(item.get("question", ""))
        gold = str(item.get("answer", ""))
        image_rel = str(item.get("image", "") or "")
        image_url = str(item.get("image_url", "") or "")
        image_path = image_root / image_rel if image_rel else Path("")
        image_b64 = encode_image(image_path) if image_rel else None
        if image_rel and image_b64 is None:
            logger.warning("Image not found: %s", image_path)

        task = {
            "id": f"simpleqa_{data_id}",
            "instruction": question,
            "image_b64": image_b64,
            "image_url": image_url,
            "task_type": "simplevqa_multimodal",
            "image_info": {
                "image_path": str(image_path) if image_rel else "",
                "image_url": image_url,
                "image_description": str(item.get("image_description", "") or ""),
                "source": str(item.get("source", "") or ""),
                "has_local_image": image_b64 is not None,
            },
        }

        logger.info("[%d/%d] data_id=%s q=%s", idx, end, data_id, question[:60])
        t0 = time.time()
        try:
            result = run_task(task, trajectory_dir=traj_dir)
            prediction = result.get("answer", "")
            err = "" if image_b64 or not image_rel else f"local image missing: {image_path}"
        except Exception as exc:  # noqa: BLE001
            logger.error("run_task failed for data_id=%s: %s", data_id, exc)
            result = {"error": str(exc), "steps": -1, "trajectory_path": ""}
            prediction = ""
            err = repr(exc)

        final_pred = clean_pred(prediction, question)
        is_correct = check_answer(final_pred, gold)
        correct += int(is_correct)
        total += 1
        record = {
            "data_id": data_id,
            "question": question,
            "ground_truth": gold,
            "prediction": prediction,
            "final_prediction": final_pred,
            "correct": is_correct,
            "elapsed_s": round(time.time() - t0, 1),
            "steps": result.get("steps", -1),
            "trajectory_path": result.get("trajectory_path", ""),
            "error": err,
        }
        results.append(record)
        append_jsonl(output, record)

        base = {"index": idx, "instruction": question, "image": image_rel, "answer": gold}
        append_jsonl(raw_path, {**base, "pred": prediction})
        append_jsonl(final_path, {**base, "pred": final_pred})

        logger.info("  => correct=%s pred=%s gt=%s acc=%.1f%%", is_correct, final_pred[:50], gold, correct / max(total, 1) * 100)

    logger.info("=== DONE === Accuracy: %.2f%% (%d/%d)", correct / max(total, 1) * 100, correct, total)
    return results


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch eval for SimpleVQA")
    parser.add_argument("--dataset", default=str(DATASET_DEFAULT))
    parser.add_argument("--image-dir", default=str(IMAGE_DIR_DEFAULT))
    parser.add_argument("--output", "-o", default="results/simpleqa_results.jsonl")
    parser.add_argument("--traj-dir", default="trajectories/simpleqa")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-gpu-check", action="store_true", help="Skip GPU check for local dry validation only.")
    args = parser.parse_args()

    if not args.skip_gpu_check:
        check_gpu()
    dataset_path = Path(args.dataset)
    image_dir = Path(args.image_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"SimpleVQA dataset not found: {dataset_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"SimpleVQA image_dir not found: {image_dir}")
    dataset = load_dataset(dataset_path)
    end = args.end
    if args.limit is not None:
        end = args.start + args.limit
    run_batch(dataset, image_dir, args.output, args.start, end if end is not None else len(dataset), args.traj_dir, args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)
