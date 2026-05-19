"""
Run the first 100 2Wiki tasks with the reflection module enabled.

This script is intentionally GPU-gated. It runs `nvidia-smi` before importing
the harness and exits if an H200/CUDA device is not visible, so the reflection
evaluation is not accidentally launched on a CPU instance.
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


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent))
DEFAULT_DATA = str(PROJECT_ROOT.parent / "datasets" / "2wiki.jsonl")
DEFAULT_OUT = "results/reflection_2wiki_full.jsonl"
DEFAULT_TRAJ_DIR = "trajectories_reflection_2wiki_full"


def norm_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def get_question(ex: dict[str, Any]) -> str:
    for key in ["question", "query", "instruction", "input", "text"]:
        if ex.get(key):
            return str(ex[key])
    raise KeyError(f"Cannot find question field in keys={list(ex.keys())}")


def get_answer(ex: dict[str, Any]) -> str:
    for key in ["answer", "gold", "label", "target", "output"]:
        if key in ex and ex[key] is not None:
            return str(ex[key])
    return ""


def get_task_id(ex: dict[str, Any], idx: int) -> str:
    for key in ["id", "_id", "qid", "question_id", "task_id", "index"]:
        if key in ex and ex[key] is not None:
            return str(ex[key])
    return f"2wiki_{idx}"


def is_correct(pred: str, gold: str) -> bool:
    pred_n = norm_text(pred)
    gold_n = norm_text(gold)
    if not gold_n:
        return False
    return gold_n == pred_n or gold_n in pred_n


def assert_h200_cuda_visible() -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"nvidia-smi is not executable here ({exc}). This does not look like a GPU instance.") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError("nvidia-smi timed out; refusing to start evaluation.") from None

    output = proc.stdout or ""
    print(output)
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed with code {proc.returncode}; refusing to start evaluation.")
    if "CUDA Version" not in output:
        raise RuntimeError("nvidia-smi output does not show CUDA Version; refusing to start evaluation.")
    if "H200" not in output:
        raise RuntimeError("nvidia-smi output does not show an H200 device; refusing to start evaluation.")
    return output


def load_existing_task_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if obj.get("task_id"):
                seen.add(str(obj["task_id"]))
    return seen


def configure_reflection_env() -> None:
    os.environ["ENABLE_REFLECTION"] = "1"
    os.environ["REFLECTION_USE_LLM"] = "0"
    os.environ["MODEL_NAME"] = "Qwen3.5-9B"
    os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8000/v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--traj-dir", default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()

    assert_h200_cuda_visible()
    configure_reflection_env()

    # Import after environment setup so task_runner reads the intended values.
    from task_runner import run_task  # noqa: PLC0415

    data_path = Path(args.data)
    out_path = Path(args.out)
    traj_dir = Path(args.traj_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    existing = load_existing_task_ids(out_path)
    total_existing = len(existing)
    print(f"Existing reflection result rows with task_id: {total_existing}")

    total_new = 0
    correct_new = 0
    processed = 0

    with open(data_path, "r", encoding="utf-8") as f, open(out_path, "a", encoding="utf-8") as w:
        for idx, line in enumerate(f):
            if idx < args.start:
                continue
            if processed >= args.limit:
                break

            ex = json.loads(line)
            task_id = get_task_id(ex, idx)
            processed += 1
            if task_id in existing:
                print(f"[skip] idx={idx} task_id={task_id} already in {out_path}")
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
                    {"id": task_id, "instruction": instruction},
                    max_steps=args.max_steps,
                    llm_base_url=os.environ["LLM_BASE_URL"],
                    model_name=os.environ["MODEL_NAME"],
                    trajectory_dir=str(traj_dir),
                )
                pred = result.get("answer", "")
                err = ""
                steps = result.get("steps")
                traj = result.get("trajectory_path", "")
                summary = result.get("summary", {})
            except Exception as exc:  # noqa: BLE001
                pred = ""
                err = repr(exc)
                steps = None
                traj = ""
                summary = {}

            ok = is_correct(pred, gold)
            total_new += 1
            correct_new += int(ok)
            rec = {
                "idx": idx,
                "task_id": task_id,
                "question": question,
                "gold": gold,
                "prediction": pred,
                "correct": ok,
                "steps": steps,
                "trajectory_path": traj,
                "summary": summary,
                "error": err,
                "latency": round(time.time() - t0, 3),
                "reflection_enabled": True,
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            w.flush()
            print(f"[{total_new}] idx={idx} correct={ok} acc_new={correct_new / max(total_new, 1):.3f} task_id={task_id}")

    print("=" * 80)
    print("reflection 2Wiki run finished")
    print(f"out={out_path}")
    print(f"traj_dir={traj_dir}")
    print(f"new_rows={total_new}")
    print(f"existing_rows_before={total_existing}")
    print("Next checks:")
    print(f"  wc -l {out_path}")
    print(f"  find {traj_dir} -name \"*.jsonl\" | wc -l")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)
