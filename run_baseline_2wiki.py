import argparse
import json
import os
import re
import time
from pathlib import Path

from task_runner import run_task


def norm_text(s):
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def get_question(ex):
    for k in ["question", "query", "instruction", "input", "text"]:
        if k in ex and ex[k]:
            return str(ex[k])
    raise KeyError(f"Cannot find question field in keys={list(ex.keys())}")


def get_answer(ex):
    for k in ["answer", "gold", "label", "target", "output"]:
        if k in ex and ex[k] is not None:
            return str(ex[k])
    return ""


def is_correct(pred, gold):
    pred_n = norm_text(pred)
    gold_n = norm_text(gold)
    if not gold_n:
        return False
    return gold_n == pred_n or gold_n in pred_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/baseline_2wiki.jsonl")
    ap.add_argument("--traj-dir", default="trajectories_baseline_2wiki")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--llm-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--model", default=os.getenv("MODEL_NAME", "Qwen3.5-9B"))
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.traj_dir).mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0

    with open(args.data, "r", encoding="utf-8") as f, open(args.out, "a", encoding="utf-8") as w:
        for idx, line in enumerate(f):
            if idx < args.start:
                continue
            if args.limit is not None and total >= args.limit:
                break

            ex = json.loads(line)
            question = get_question(ex)
            gold = get_answer(ex)
            task_id = str(ex.get("id") or ex.get("_id") or f"2wiki_{idx}")

            instruction = (
                "Answer the following 2Wiki multi-hop question. "
                "You may use search and browser tools if needed. "
                "Give the final answer as concise text only.\n\n"
                f"Question: {question}"
            )

            t0 = time.time()
            try:
                result = run_task(
                    {
                        "id": task_id,
                        "instruction": instruction,
                    },
                    max_steps=args.max_steps,
                    llm_base_url=args.llm_url,
                    model_name=args.model,
                    trajectory_dir=args.traj_dir,
                )
                pred = result.get("answer", "")
                raw_pred = result.get("raw_answer", pred)
                err = ""
                steps = result.get("steps", None)
                traj = result.get("trajectory_path", "")
                summary = result.get("summary", {})
            except Exception as e:
                pred = ""
                raw_pred = ""
                err = repr(e)
                steps = None
                traj = ""
                summary = {}

            ok = is_correct(pred, gold)
            total += 1
            correct += int(ok)

            rec = {
                "idx": idx,
                "task_id": task_id,
                "question": question,
                "gold": gold,
                "prediction": pred,
                "raw_prediction": raw_pred,
                "correct": ok,
                "steps": steps,
                "trajectory_path": traj,
                "summary": summary,
                "error": err,
                "latency": round(time.time() - t0, 3),
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            w.flush()

            print(f"[{total}] idx={idx} correct={ok} acc={correct/total:.3f} task_id={task_id}")

    print("=" * 60)
    print(f"done total={total} correct={correct} acc={correct / max(total, 1):.4f}")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
