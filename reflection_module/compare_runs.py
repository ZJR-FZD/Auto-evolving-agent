"""
Compare baseline, rule-reflection, and Qwen3-30B-A3B critic reflection runs.

This script only analyzes existing run outputs. It never calls the model and
uses gold answers only for offline accuracy statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def summarize_results(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    n = len(rows)
    correct_values = [bool(r.get("correct")) for r in rows if "correct" in r]
    steps = [_to_number(r.get("steps")) for r in rows if _to_number(r.get("steps")) is not None]

    totals = {
        "tool_error": 0,
        "max_steps": 0,
        "malformed_tool_call": 0,
        "empty_assistant": 0,
        "reflection_trigger": 0,
        "critic_success": 0,
        "critic_fallback": 0,
        "memory_hit": 0,
        "recovery_after_reflection": 0,
    }

    for row in rows:
        pred = str(row.get("prediction", "") or row.get("pred", "") or "")
        err = str(row.get("error", "") or "")
        task_flags = {
            "tool_error": "[ERROR]" in pred or "[HARNESS ERROR]" in pred or bool(err),
            "max_steps": "Max steps reached" in pred,
            "malformed_tool_call": False,
            "empty_assistant": False,
            "reflection_trigger": bool(row.get("reflection_used")),
            "critic_success": False,
            "critic_fallback": False,
            "memory_hit": False,
            "recovery_after_reflection": False,
        }

        traj_path = Path(str(row.get("trajectory_path", "") or ""))
        if traj_path.exists():
            traj_stats = summarize_trajectory(traj_path)
            for key in task_flags:
                task_flags[key] = bool(task_flags[key] or traj_stats.get(key))

        for key, value in task_flags.items():
            totals[key] += int(value)

    return {
        "path": str(path),
        "num_rows": n,
        "accuracy": round(sum(correct_values) / len(correct_values), 4) if correct_values else None,
        "avg_steps": round(mean(steps), 3) if steps else None,
        "tool_error_rate": _rate(totals["tool_error"], n),
        "max_steps_rate": _rate(totals["max_steps"], n),
        "malformed_tool_call_rate": _rate(totals["malformed_tool_call"], n),
        "empty_assistant_rate": _rate(totals["empty_assistant"], n),
        "reflection_trigger_count": totals["reflection_trigger"],
        "critic_success_count": totals["critic_success"],
        "critic_fallback_count": totals["critic_fallback"],
        "memory_hit_count": totals["memory_hit"],
        "recovery_after_reflection_rate": _rate(totals["recovery_after_reflection"], totals["reflection_trigger"]),
    }


def summarize_trajectory(path: Path) -> dict[str, bool]:
    rows = load_jsonl(path)
    has_tool_error = False
    has_max_steps = False
    malformed_tool_call = False
    empty_assistant = False
    reflection_trigger = False
    critic_success = False
    critic_fallback = False
    memory_hit = False
    recovery_after_reflection = False
    max_step = 0
    last_assistant_text = ""
    reflection_step = None

    for row in rows:
        role = row.get("role")
        text = str(row.get("content", "") or "")
        step = int(row.get("step_id") or 0)
        max_step = max(max_step, step)
        if role == "tool" and ("[ERROR]" in text or "[HARNESS ERROR]" in text or '"ok": false' in text):
            has_tool_error = True
        if role == "assistant":
            tool_calls = row.get("tool_calls") or []
            reasoning = str(row.get("reasoning_content", "") or "")
            if text.strip():
                last_assistant_text = text
            if not text.strip() and not tool_calls:
                if "<tool_call>" in reasoning or "<function=" in reasoning:
                    malformed_tool_call = True
                else:
                    empty_assistant = True
        if row.get("reflection_trigger") or "[Reflection]" in text:
            reflection_trigger = True
            reflection_step = step if reflection_step is None else min(reflection_step, step)
            mode = str(row.get("reflection_mode", ""))
            critic_success = critic_success or (mode == "Qwen3-30B-A3B")
            critic_fallback = critic_fallback or (mode == "rule_fallback")
            memory_hit = memory_hit or int(row.get("memory_hits") or 0) > 0

    if max_step >= 6 and not last_assistant_text:
        has_max_steps = True

    if reflection_step is not None:
        for row in rows:
            if int(row.get("step_id") or 0) <= reflection_step:
                continue
            if row.get("role") == "assistant" and (
                str(row.get("content", "") or "").strip() or row.get("tool_calls")
            ):
                recovery_after_reflection = True
                break

    return {
        "tool_error": has_tool_error,
        "max_steps": has_max_steps,
        "malformed_tool_call": malformed_tool_call,
        "empty_assistant": empty_assistant,
        "reflection_trigger": reflection_trigger,
        "critic_success": critic_success,
        "critic_fallback": critic_fallback,
        "memory_hit": memory_hit,
        "recovery_after_reflection": recovery_after_reflection,
    }


def build_report(
    baseline_path: str | Path,
    reflection_path: str | Path | None = None,
    rule_path: str | Path | None = None,
    critic_path: str | Path | None = None,
) -> dict[str, Any]:
    baseline = Path(baseline_path)
    if not baseline.exists():
        return {"ok": False, "error": f"baseline results not found: {baseline}"}

    groups: dict[str, Path] = {"baseline": baseline}
    if rule_path:
        groups["rule_reflection"] = Path(rule_path)
    if critic_path:
        groups["qwen3_30b_a3b_critic_reflection"] = Path(critic_path)
    if reflection_path and not rule_path and not critic_path:
        groups["reflection"] = Path(reflection_path)

    missing = {name: str(path) for name, path in groups.items() if not path.exists()}
    if missing:
        return {
            "ok": False,
            "error": "one or more result files are missing",
            "missing": missing,
            "how_to_run": (
                "Run the corresponding reflection evaluation first, then rerun compare_runs.py. "
                "For critic mode use run_reflection_2wiki_qwen3_30b_a3b.py or run_reflection_simplevqa_qwen3_30b_a3b.py."
            ),
        }

    summaries = {name: summarize_results(path) for name, path in groups.items()}
    report: dict[str, Any] = {"ok": True, "groups": summaries}
    if "rule_reflection" in summaries:
        report["delta_rule_vs_baseline"] = _delta_summary(summaries["rule_reflection"], summaries["baseline"])
    if "qwen3_30b_a3b_critic_reflection" in summaries:
        report["delta_critic_vs_baseline"] = _delta_summary(summaries["qwen3_30b_a3b_critic_reflection"], summaries["baseline"])
    if "reflection" in summaries:
        report["delta_reflection_vs_baseline"] = _delta_summary(summaries["reflection"], summaries["baseline"])
    return report


def _delta_summary(new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "accuracy",
        "avg_steps",
        "tool_error_rate",
        "max_steps_rate",
        "malformed_tool_call_rate",
        "empty_assistant_rate",
        "recovery_after_reflection_rate",
    ]
    return {f"delta_{key}": _delta(new.get(key), old.get(key)) for key in keys}


def _rate(count: int, denom: int) -> float | None:
    return round(count / denom, 4) if denom else None


def _delta(new: Any, old: Any) -> Any:
    if new is None or old is None:
        return None
    return round(new - old, 4)


def _to_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", default="results/baseline_2wiki_full.jsonl")
    parser.add_argument("--reflection-results", default=None, help="Backward-compatible two-run comparison path.")
    parser.add_argument("--rule-results", default=None)
    parser.add_argument("--critic-results", default=None)
    parser.add_argument("--out", default="reflection_module/reflection_compare_report.json")
    args = parser.parse_args()

    report = build_report(
        baseline_path=args.baseline_results,
        reflection_path=args.reflection_results,
        rule_path=args.rule_results,
        critic_path=args.critic_results,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
