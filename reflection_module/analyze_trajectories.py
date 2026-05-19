"""
Offline trajectory failure analysis.

The analyzer uses only agent-side signals: tool errors, empty actions,
malformed tool calls, repeated queries, truncation metadata, and budget-like
termination. It never looks at gold answers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


KNOWN_TOOLS = {
    "search_text",
    "search_image",
    "browser_navigate",
    "browser_get_text",
    "browser_click",
    "browser_type",
    "browser_parallel",
}


SUGGESTIONS = {
    "tool_error": "解析工具错误，改写参数或换用更稳的工具，避免原样重试。",
    "empty_assistant": "要求下一步必须二选一：真实工具调用或最终答案，避免空转。",
    "max_steps": "压缩当前证据，限制下一轮搜索深度，证据足够时立即回答。",
    "finish_reason_length": "压缩中间推理和上下文，只保留关键实体、证据和下一步动作。",
    "malformed_tool_call": "将 reasoning 中的伪工具调用改为 OpenAI function-calling 格式。",
    "unknown_tool": "改用 harness 注册过的工具名，并只传 schema 中声明的字段。",
    "repeated_search_query": "改变查询表达，加入别名、年份、地点或实体类型，而不是重复同一 query。",
}


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


def analyze_file(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    task_id = path.stem
    assistant_rows = [r for r in rows if r.get("role") == "assistant"]
    tool_rows = [r for r in rows if r.get("role") == "tool"]
    events = []
    query_counter = Counter()
    tool_call_count = 0

    for row in assistant_rows:
        content = str(row.get("content", "") or "")
        reasoning = str(row.get("reasoning_content", "") or "")
        tool_calls = row.get("tool_calls") or []
        if isinstance(tool_calls, list):
            tool_call_count += len(tool_calls)
            for tc in tool_calls:
                fn = ((tc or {}).get("function") or {}).get("name", "")
                args_raw = ((tc or {}).get("function") or {}).get("arguments", "{}")
                args = _loads_dict(args_raw)
                if fn not in KNOWN_TOOLS:
                    events.append(_event(task_id, "unknown_tool", f"unknown tool_call name={fn}", row))

        if not content.strip() and not tool_calls:
            if "<tool_call>" in reasoning or "<function=" in reasoning:
                events.append(_event(task_id, "malformed_tool_call", reasoning[:500], row))
            else:
                events.append(_event(task_id, "empty_assistant", "assistant has no content and no tool_calls", row))

        if row.get("finish_reason") == "length":
            events.append(_event(task_id, "finish_reason_length", "finish_reason=length", row))

    for row in tool_rows:
        text = str(row.get("content", "") or "")
        fn = str(row.get("fn_name", "") or "")
        args = row.get("fn_args") if isinstance(row.get("fn_args"), dict) else {}
        lowered = text.lower()
        if fn and fn not in KNOWN_TOOLS and "unknown tool" not in lowered:
            events.append(_event(task_id, "unknown_tool", f"unknown tool result fn_name={fn}", row))
        query = _query_from_args(fn, args)
        if query:
            query_counter[(fn, query)] += 1
        if "[error]" in lowered or "ok=false" in lowered or '"ok": false' in lowered or "proxy-error" in lowered:
            ftype = "unknown_tool" if "unknown tool" in lowered else "tool_error"
            events.append(_event(task_id, ftype, text[:500], row))

    repeated = [
        {"tool": fn, "query": query, "count": count}
        for (fn, query), count in query_counter.items()
        if count > 1
    ]
    for item in repeated:
        events.append(
            {
                "task_id": task_id,
                "failure_type": "repeated_search_query",
                "evidence": f"{item['tool']} query repeated {item['count']} times: {item['query']}",
                "suggested_reflection": SUGGESTIONS["repeated_search_query"],
            }
        )

    max_step = max([r.get("step_id", 0) or 0 for r in rows], default=0)
    last_assistant = assistant_rows[-1] if assistant_rows else {}
    has_final_text = bool(str(last_assistant.get("content", "") or "").strip())
    if max_step >= 6 and not has_final_text:
        events.append(_event(task_id, "max_steps", f"last step={max_step}, no final assistant text", last_assistant))

    return {
        "task_id": task_id,
        "steps": max_step,
        "tool_calls": tool_call_count if tool_call_count else len(tool_rows),
        "events": events,
        "repeated_queries": repeated,
    }


def analyze_dir(traj_dir: str | Path) -> dict[str, Any]:
    path = Path(traj_dir)
    files = sorted(path.glob("*.jsonl"))
    per_task = [analyze_file(p) for p in files]

    failure_type_counts = Counter()
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    repeated_counter = Counter()
    failed_like = 0

    for row in per_task:
        if row["events"]:
            failed_like += 1
        seen_types = set()
        for event in row["events"]:
            ftype = event["failure_type"]
            failure_type_counts[ftype] += 1
            seen_types.add(ftype)
            if len(samples[ftype]) < 3:
                samples[ftype].append(event)
        for item in row["repeated_queries"]:
            repeated_counter[item["query"]] += item["count"]

    return {
        "trajectory_dir": str(path),
        "num_tasks": len(per_task),
        "num_failed_like": failed_like,
        "failure_type_counts": dict(failure_type_counts),
        "avg_steps": round(mean([r["steps"] for r in per_task]), 3) if per_task else 0,
        "avg_tool_calls": round(mean([r["tool_calls"] for r in per_task]), 3) if per_task else 0,
        "top_repeated_queries": [
            {"query": query, "count": count}
            for query, count in repeated_counter.most_common(20)
        ],
        "samples": dict(samples),
    }


def _event(task_id: str, failure_type: str, evidence: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "failure_type": failure_type,
        "step_id": row.get("step_id"),
        "evidence": str(evidence)[:700],
        "suggested_reflection": SUGGESTIONS.get(failure_type, "先定位失败原因，再修改下一步动作。"),
    }


def _loads_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _query_from_args(fn: str, args: dict[str, Any]) -> str:
    if fn == "search_text":
        return str(args.get("query", "") or "").strip()
    if fn == "search_image":
        return str(args.get("image_url", "") or "").strip()
    if fn == "browser_navigate":
        return str(args.get("url", "") or "").strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", default="trajectories_baseline_2wiki_full")
    parser.add_argument("--out", default="reflection_module/trajectory_failure_report.json")
    args = parser.parse_args()

    report = analyze_dir(args.traj_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
