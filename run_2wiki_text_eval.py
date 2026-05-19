"""Run and evaluate 2Wiki-style text tasks with the current ReAct harness.

The script reuses ``task_runner.run_task`` and therefore uses the same
Qwen3.5-9B main agent, reflection settings, tools, and TEXT_BOUNDED_MODE
configuration as the benchmark runner.

It supports two evaluation modes:
1. If gold answers exist in the JSONL rows, report EM / contains accuracy / F1.
2. If gold answers are absent, report run-quality metrics only and mark
   ``gold_available=false``.  This is the case for the provided
   ``datasets/2wiki.jsonl`` in this workspace.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from answer_utils import clean_pred_for_submit, infer_answer_type, normalize_for_metric
from run_benchmark import probe_openai_models


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent))
DEFAULT_DATA = PROJECT_ROOT.parent / "datasets" / "2wiki.jsonl"
DEFAULT_OUT = "results/2wiki_text_eval_results.jsonl"
DEFAULT_REPORT = "results/2wiki_text_eval_metrics.json"
DEFAULT_TRAJ_DIR = "trajectories/2wiki_text_eval"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_question(row: dict[str, Any]) -> str:
    for key in ("question", "query", "instruction", "input", "text"):
        value = row.get(key)
        if value:
            return str(value)
    raise KeyError(f"question field not found; keys={list(row)}")


def get_gold(row: dict[str, Any]) -> str:
    for key in ("answer", "gold", "label", "target", "output"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def get_task_id(row: dict[str, Any], idx: int) -> str:
    return str(row.get("id") or row.get("_id") or f"2wiki_{idx:04d}")


def normalize_answer(text: Any) -> str:
    """SQuAD/HotpotQA-style normalization for EM/F1."""
    return normalize_for_metric(text)


def exact_match(pred: str, gold: str) -> bool:
    return bool(gold.strip()) and normalize_answer(pred) == normalize_answer(gold)


def contains_match(pred: str, gold: str) -> bool:
    pred_n = normalize_answer(pred)
    gold_n = normalize_answer(gold)
    return bool(gold_n) and (pred_n == gold_n or gold_n in pred_n)


def token_f1(pred: str, gold: str) -> float | None:
    gold_n = normalize_answer(gold)
    if not gold_n:
        return None
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = gold_n.split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def read_trajectory(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"_broken_json": line[:200]})
    return rows


def summarize_trajectory(rows: list[dict[str, Any]], max_steps: int) -> dict[str, Any]:
    required = {"timestamp", "step_id", "role", "content", "tool_call_id"}
    tool_calls = Counter()
    tool_errors = Counter()
    reflection_modes = Counter()
    finish_reasons = Counter()
    role_counts = Counter()
    total_tokens = 0
    assistant_turns = 0
    empty_assistant = 0
    malformed_tool_call = 0
    textual_tool_rescue = 0
    final_synthesis = 0
    invalid_lines = 0
    max_step_seen = 0
    search_proxy_degraded = 0

    for row in rows:
        if "_broken_json" in row:
            invalid_lines += 1
            continue
        missing = required - set(row)
        if missing:
            invalid_lines += 1
        role = str(row.get("role") or "")
        content = str(row.get("content") or "")
        extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
        step = _safe_int(row.get("step_id"), 0)
        max_step_seen = max(max_step_seen, step)
        role_counts[role] += 1

        if role == "assistant":
            assistant_turns += 1
            tool_call_data = row.get("tool_calls") or extra.get("tool_calls") or []
            reasoning = str(row.get("reasoning_content") or extra.get("reasoning_content") or "")
            if not content.strip() and not tool_call_data:
                empty_assistant += 1
                if "<tool_call>" in reasoning or "<function=" in reasoning:
                    malformed_tool_call += 1
            if row.get("total_tokens") or extra.get("total_tokens"):
                total_tokens += _safe_int(row.get("total_tokens") or extra.get("total_tokens"), 0)
            if extra.get("textual_tool_rescue") or row.get("textual_tool_rescue"):
                textual_tool_rescue += 1
            finish = row.get("finish_reason") or extra.get("finish_reason")
            if finish:
                finish_reasons[str(finish)] += 1

        if role == "tool":
            fn = row.get("fn_name") or extra.get("fn_name") or "unknown_tool"
            tool_calls[fn] += 1
            low = content.lower()
            if any(x in low for x in ("[error]", "proxy-error", "timeout", "timed out", "ok=false")):
                tool_errors[fn] += 1

        if extra.get("reflection_trigger") or row.get("reflection_trigger"):
            reflection_modes[str(extra.get("reflection_mode") or row.get("reflection_mode") or "unknown")] += 1
        if extra.get("final_synthesis") or row.get("final_synthesis"):
            final_synthesis += 1
        if "search_proxy_degraded" in content:
            search_proxy_degraded += 1

    return {
        "num_turns": len(rows),
        "max_step_seen": max_step_seen,
        "hit_max_steps": max_step_seen >= max_steps,
        "invalid_lines": invalid_lines,
        "role_counts": dict(role_counts),
        "assistant_turns": assistant_turns,
        "total_tokens": total_tokens,
        "tool_calls": dict(tool_calls),
        "tool_errors": dict(tool_errors),
        "reflection_modes": dict(reflection_modes),
        "finish_reasons": dict(finish_reasons),
        "empty_assistant": empty_assistant,
        "malformed_tool_call": malformed_tool_call,
        "textual_tool_rescue": textual_tool_rescue,
        "final_synthesis": final_synthesis,
        "search_proxy_degraded": search_proxy_degraded,
    }


def load_existing(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "idx" in row:
                out[int(row["idx"])] = row
    return out


def check_runtime(llm_base_url: str, critic_base_url: str | None, require_critic: bool) -> None:
    proc = subprocess.run(["nvidia-smi"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
    print(proc.stdout or "")
    if proc.returncode != 0 or "CUDA Version" not in (proc.stdout or ""):
        raise RuntimeError("CUDA/GPU is not visible. Run this evaluation on the GPU instance.")
    print("main_models=", probe_openai_models(llm_base_url))
    if critic_base_url:
        try:
            print("critic_models=", probe_openai_models(critic_base_url))
        except Exception as exc:  # noqa: BLE001
            if require_critic:
                raise
            print(f"[WARN] critic endpoint unavailable: {exc}")


def build_instruction(question: str, include_context: bool, row: dict[str, Any]) -> str:
    context_text = ""
    if include_context and isinstance(row.get("context"), dict):
        titles = row["context"].get("title") or []
        sentences = row["context"].get("sentences") or []
        chunks = []
        for title, sents in zip(titles, sentences):
            if isinstance(sents, list):
                chunks.append(f"{title}: {' '.join(str(x) for x in sents[:3])}")
        context_text = "\n\nContext snippets:\n" + "\n".join(chunks[:12])

    return (
        "Answer the following 2Wiki multi-hop text question. "
        "Use concise multi-hop reasoning internally. Use tools when external verification is needed. "
        "Return only <answer>...</answer> with the final answer. "
        f"Expected answer type: {infer_answer_type(question)}. "
        "The final answer must be one short span of that type, with no explanation.\n\n"
        f"Question: {question}"
        f"{context_text}"
    )


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    data = load_jsonl(args.data)
    end = min(args.end if args.end is not None else len(data), len(data))
    selected = [(idx, data[idx]) for idx in range(args.start, end)]
    if args.limit is not None:
        selected = selected[: args.limit]

    out_path = Path(args.out)
    traj_dir = Path(args.traj_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    existing = load_existing(out_path)

    if not selected:
        return build_report([], args)

    from task_runner import run_task  # noqa: PLC0415

    for idx, row in selected:
        if idx in existing and not args.overwrite:
            print(f"[skip] idx={idx} already exists")
            continue
        question = get_question(row)
        gold = get_gold(row)
        task_id = get_task_id(row, idx)
        instruction = build_instruction(question, args.include_context, row)
        t0 = time.time()
        error = ""
        try:
            result = run_task(
                {
                    "id": task_id,
                    "instruction": instruction,
                    "task_type": "2wiki_text",
                },
                max_steps=args.max_steps,
                llm_base_url=args.llm_base_url,
                model_name=args.model,
                trajectory_dir=str(traj_dir),
            )
            raw_pred = str(result.get("raw_answer", result.get("answer", "")) or "")
            trajectory_path = str(result.get("trajectory_path", ""))
            steps = result.get("steps")
        except Exception as exc:  # noqa: BLE001
            raw_pred = ""
            trajectory_path = str(traj_dir / f"{task_id}.jsonl")
            steps = None
            error = repr(exc)

        pred = clean_pred_for_submit(raw_pred, question)
        traj_rows = read_trajectory(trajectory_path)
        traj_stats = summarize_trajectory(traj_rows, args.max_steps)
        em = exact_match(pred, gold) if gold else None
        contains = contains_match(pred, gold) if gold else None
        f1 = token_f1(pred, gold) if gold else None
        rec = {
            "idx": idx,
            "task_id": task_id,
            "question": question,
            "gold": gold,
            "gold_available": bool(gold),
            "prediction": pred,
            "raw_prediction": raw_pred,
            "exact_match": em,
            "contains_match": contains,
            "f1": f1,
            "correct": contains if gold else None,
            "steps": steps,
            "latency_s": round(time.time() - t0, 3),
            "trajectory_path": trajectory_path,
            "trajectory_stats": traj_stats,
            "error": error,
        }
        mode = "w" if args.overwrite and idx == selected[0][0] else "a"
        with open(out_path, mode, encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        existing[idx] = rec
        print(
            f"[idx={idx}] gold={bool(gold)} em={em} f1={f1} "
            f"steps={steps} tokens={traj_stats['total_tokens']} latency={rec['latency_s']} pred={pred[:80]!r}"
        )

    rows = [existing[idx] for idx, _ in selected if idx in existing]
    return build_report(rows, args)


def build_report(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    n = len(rows)
    gold_rows = [r for r in rows if r.get("gold_available")]
    tool_calls = Counter()
    tool_errors = Counter()
    reflection_modes = Counter()
    finish_reasons = Counter()
    role_counts = Counter()
    invalid_trajectories = 0
    harness_answers = 0
    empty_predictions = 0
    insufficient = 0
    total_tokens = 0
    total_llm_calls = 0
    total_tool_calls = 0

    for row in rows:
        pred = str(row.get("prediction") or "")
        raw = str(row.get("raw_prediction") or "")
        if not pred.strip():
            empty_predictions += 1
        if "[HARNESS]" in pred or "[HARNESS]" in raw:
            harness_answers += 1
        if re.search(r"insufficient evidence|unable to determine|cannot determine|无法确定|证据不足", pred, re.I):
            insufficient += 1
        st = row.get("trajectory_stats") or {}
        if st.get("invalid_lines"):
            invalid_trajectories += 1
        total_tokens += int(st.get("total_tokens") or 0)
        total_llm_calls += int(st.get("assistant_turns") or 0)
        for k, v in (st.get("tool_calls") or {}).items():
            tool_calls[k] += int(v)
            total_tool_calls += int(v)
        for k, v in (st.get("tool_errors") or {}).items():
            tool_errors[k] += int(v)
        for k, v in (st.get("reflection_modes") or {}).items():
            reflection_modes[k] += int(v)
        for k, v in (st.get("finish_reasons") or {}).items():
            finish_reasons[k] += int(v)
        for k, v in (st.get("role_counts") or {}).items():
            role_counts[k] += int(v)

    f1_values = [r["f1"] for r in gold_rows if r.get("f1") is not None]
    report = {
        "data": str(args.data),
        "out": str(args.out),
        "traj_dir": str(args.traj_dir),
        "range": {"start": args.start, "end": args.end, "limit": args.limit},
        "num_rows": n,
        "gold_available": bool(gold_rows),
        "gold_count": len(gold_rows),
        "exact_match": _mean_bool([r.get("exact_match") for r in gold_rows]),
        "contains_accuracy": _mean_bool([r.get("contains_match") for r in gold_rows]),
        "avg_f1": round(mean(f1_values), 4) if f1_values else None,
        "avg_latency_s": _mean_num([r.get("latency_s") for r in rows]),
        "avg_steps": _mean_num([r.get("steps") for r in rows]),
        "total_tokens": total_tokens,
        "avg_tokens_per_task": round(total_tokens / n, 2) if n else None,
        "total_llm_calls": total_llm_calls,
        "avg_llm_calls_per_task": round(total_llm_calls / n, 2) if n else None,
        "total_tool_calls": total_tool_calls,
        "avg_tool_calls_per_task": round(total_tool_calls / n, 2) if n else None,
        "tool_calls": dict(tool_calls),
        "tool_errors": dict(tool_errors),
        "tool_error_rate_by_call": {
            k: round(tool_errors[k] / v, 4) for k, v in tool_calls.items() if v
        },
        "reflection_modes": dict(reflection_modes),
        "finish_reasons": dict(finish_reasons),
        "role_counts": dict(role_counts),
        "invalid_trajectory_count": invalid_trajectories,
        "empty_prediction_count": empty_predictions,
        "harness_answer_count": harness_answers,
        "insufficient_or_uncertain_count": insufficient,
        "invalid_answer_rate": round((empty_predictions + harness_answers + insufficient) / n, 4) if n else None,
        "note": (
            "Gold answers are absent in the provided file, so EM/F1/accuracy are null."
            if not gold_rows
            else "Gold answers were available and used only for offline metrics."
        ),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _mean_bool(values: list[Any]) -> float | None:
    clean = [bool(v) for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def _mean_num(values: list[Any]) -> float | None:
    nums = []
    for value in values:
        try:
            nums.append(float(value))
        except (TypeError, ValueError):
            pass
    return round(mean(nums), 4) if nums else None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--traj-dir", default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "Qwen3.5-9B"))
    parser.add_argument("--include-context", action="store_true", help="Include provided 2Wiki context snippets in the prompt.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-env-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_env_check:
        critic_url = os.getenv("REFLECTION_BASE_URL") if os.getenv("REFLECTION_USE_LLM") == "1" else None
        check_runtime(args.llm_base_url, critic_url, os.getenv("REQUIRE_REFLECTION_CRITIC", "0") == "1")
    run_eval(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)
