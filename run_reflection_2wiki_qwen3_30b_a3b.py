"""
Run 2Wiki with Qwen3.5-9B as main agent and Qwen3-30B-A3B as open-source
reflection critic. The critic diagnoses failures only and never answers tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from run_reflection_2wiki_full_gpu import get_answer, get_question, get_task_id, is_correct, load_existing_task_ids


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent))
DEFAULT_DATA = PROJECT_ROOT.parent / "datasets" / "2wiki.jsonl"
DEFAULT_OUT = "results/reflection_qwen3_30b_a3b_2wiki_full.jsonl"
DEFAULT_TRAJ_DIR = "trajectories_reflection_qwen3_30b_a3b_2wiki_full"


def check_gpu() -> None:
    try:
        proc = subprocess.run(["nvidia-smi"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"nvidia-smi is not executable here ({exc}); run this on the GPU instance.") from None
    print(proc.stdout or "")
    if proc.returncode != 0 or "CUDA Version" not in (proc.stdout or ""):
        raise RuntimeError("CUDA device is not visible; refusing to run the main Qwen agent.")


def configure_env() -> None:
    os.environ["ENABLE_REFLECTION"] = "1"
    os.environ["REFLECTION_USE_LLM"] = "1"
    os.environ["REFLECTION_MODEL"] = "Qwen3-30B-A3B"
    os.environ.setdefault("REFLECTION_BASE_URL", "http://127.0.0.1:8001/v1")
    os.environ.setdefault("REFLECTION_API_KEY", "EMPTY")
    os.environ["MODEL_NAME"] = "Qwen3.5-9B"
    os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8000/v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--traj-dir", default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()

    check_gpu()
    configure_env()
    from task_runner import run_task  # noqa: PLC0415

    out_path = Path(args.out)
    traj_dir = Path(args.traj_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    existing = load_existing_task_ids(out_path)

    total = 0
    correct = 0
    processed = 0
    with open(args.data, "r", encoding="utf-8") as f, open(out_path, "a", encoding="utf-8") as w:
        for idx, line in enumerate(f):
            if idx < args.start:
                continue
            if processed >= args.limit:
                break
            ex = json.loads(line)
            processed += 1
            task_id = get_task_id(ex, idx)
            if task_id in existing:
                print(f"[skip] idx={idx} task_id={task_id}")
                continue
            question = get_question(ex)
            gold = get_answer(ex)
            instruction = (
                "Answer the following 2Wiki multi-hop question. "
                "You may use search and browser tools if needed. "
                "Give the final answer as concise text only.\n\n"
                f"Question: {question}"
            )
            t0 = time.time()
            try:
                result = run_task(
                    {"id": task_id, "instruction": instruction, "task_type": "2wiki_text"},
                    max_steps=args.max_steps,
                    llm_base_url=os.environ["LLM_BASE_URL"],
                    model_name=os.environ["MODEL_NAME"],
                    trajectory_dir=str(traj_dir),
                )
                pred = result.get("answer", "")
                raw_pred = result.get("raw_answer", pred)
                err = ""
            except Exception as exc:  # noqa: BLE001
                result = {}
                pred = ""
                raw_pred = ""
                err = repr(exc)
            ok = is_correct(pred, gold)
            total += 1
            correct += int(ok)
            row = {
                "idx": idx,
                "task_id": task_id,
                "question": question,
                "gold": gold,
                "prediction": pred,
                "raw_prediction": raw_pred,
                "correct": ok,
                "steps": result.get("steps"),
                "trajectory_path": result.get("trajectory_path", ""),
                "summary": result.get("summary", {}),
                "error": err,
                "latency": round(time.time() - t0, 3),
                "reflection_mode": "Qwen3-30B-A3B",
            }
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            w.flush()
            print(f"[{total}] idx={idx} correct={ok} acc={correct / max(total, 1):.3f} task_id={task_id}")

    print(f"saved to {out_path}")
    print(f"trajectories in {traj_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)
