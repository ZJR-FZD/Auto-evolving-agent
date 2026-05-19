"""Summarize and compare harness evaluation runs.

Examples:
    python harness-sii/analyze_runs.py harness-sii/runs/simplevqa_evolved
    python harness-sii/analyze_runs.py harness-sii/runs/simplevqa_raw harness-sii/runs/simplevqa_evolved
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_PUNCT_RE = re.compile(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()\[\]{}<>《》【】]+")


def _normalize_answer(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    return _PUNCT_RE.sub("", text.lower()).strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pick_rows_path(run_dir: Path) -> Path:
    for name in ("progress.jsonl", "predictions.jsonl", "final_results.jsonl"):
        path = run_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No progress/predictions JSONL found in {run_dir}")


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sum_stat(rows: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for row in rows:
        stats = row.get("stats") if isinstance(row.get("stats"), dict) else {}
        value = stats.get(key, 0)
        try:
            total += float(value or 0)
        except (TypeError, ValueError):
            pass
    return total


def summarize(run_dir: Path) -> dict[str, Any]:
    path = _pick_rows_path(run_dir)
    rows = _read_jsonl(path)
    total = len(rows)
    errors = sum(1 for row in rows if row.get("error"))

    exact = 0
    contains = 0
    for row in rows:
        if "exact_match" in row:
            exact += bool(row.get("exact_match"))
            contains += bool(row.get("contains_gold", row.get("exact_match")))
            continue
        pred = _normalize_answer(row.get("pred", row.get("prediction", "")))
        gold = _normalize_answer(row.get("answer", ""))
        exact += bool(gold and pred == gold)
        contains += bool(gold and gold in pred)

    steps = [float(row.get("steps", 0) or 0) for row in rows]
    elapsed = [float(row.get("elapsed_seconds", 0) or 0) for row in rows]
    tool_calls = _sum_stat(rows, "tool_calls")
    tool_errors = _sum_stat(rows, "tool_errors")
    repeated_tool_calls = _sum_stat(rows, "repeated_tool_calls")
    reflection_hints = _sum_stat(rows, "reflection_hints")
    total_tokens = _sum_stat(rows, "total_tokens")
    prompt_tokens = _sum_stat(rows, "prompt_tokens")
    completion_tokens = _sum_stat(rows, "completion_tokens")
    memory_recalled = _sum_stat(rows, "memory_recalled")
    memory_written = _sum_stat(rows, "memory_written")

    return {
        "run_dir": str(run_dir),
        "rows_path": str(path),
        "total": total,
        "errors": errors,
        "exact": exact,
        "contains": contains,
        "exact_rate": exact / total if total else 0.0,
        "contains_rate": contains / total if total else 0.0,
        "avg_steps": _avg(steps),
        "avg_elapsed_seconds": _avg(elapsed),
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens / total if total else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tool_calls": tool_calls,
        "avg_tool_calls": tool_calls / total if total else 0.0,
        "tool_errors": tool_errors,
        "repeated_tool_calls": repeated_tool_calls,
        "reflection_hints": reflection_hints,
        "memory_recalled": memory_recalled,
        "memory_written": memory_written,
    }


def _format_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "run",
        "total",
        "exact",
        "contains",
        "avg_steps",
        "avg_tokens",
        "avg_tool",
        "tool_err",
        "repeat",
        "reflect",
        "mem_in",
        "mem_out",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [
            Path(row["run_dir"]).name,
            str(row["total"]),
            f"{row['exact_rate']:.2%}",
            f"{row['contains_rate']:.2%}",
            f"{row['avg_steps']:.2f}",
            f"{row['avg_tokens']:.1f}",
            f"{row['avg_tool_calls']:.2f}",
            f"{row['tool_errors']:.0f}",
            f"{row['repeated_tool_calls']:.0f}",
            f"{row['reflection_hints']:.0f}",
            f"{row['memory_recalled']:.0f}",
            f"{row['memory_written']:.0f}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_delta(base: dict[str, Any], evolved: dict[str, Any]) -> str:
    keys = [
        ("exact_rate", "exact_rate", True),
        ("contains_rate", "contains_rate", True),
        ("avg_steps", "avg_steps", False),
        ("avg_tokens", "avg_tokens", False),
        ("avg_tool_calls", "avg_tool_calls", False),
        ("tool_errors", "tool_errors", False),
        ("repeated_tool_calls", "repeated_tool_calls", False),
    ]
    lines = ["| metric | raw | evolved | delta |", "| --- | ---: | ---: | ---: |"]
    for label, key, percent in keys:
        raw = float(base.get(key, 0) or 0)
        new = float(evolved.get(key, 0) or 0)
        delta = new - raw
        if percent:
            values = [f"{raw:.2%}", f"{new:.2%}", f"{delta:+.2%}"]
        else:
            values = [f"{raw:.2f}", f"{new:.2f}", f"{delta:+.2f}"]
        lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize harness run metrics")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="One or two run directories")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    summaries = [summarize(path) for path in args.run_dirs]
    if args.json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return
    print(_format_table(summaries))
    if len(summaries) == 2:
        print()
        print(_format_delta(summaries[0], summaries[1]))


if __name__ == "__main__":
    main()

