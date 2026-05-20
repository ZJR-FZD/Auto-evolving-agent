"""
TPGO trajectory analysis utilities.

This module is intentionally dependency-free so it can run on the current
workspace immediately. It provides:

- trajectory JSONL parsing
- efficiency metrics
- heuristic reflection/memory extraction
- Mermaid trajectory graph rendering
- an initial Textual Parameter Graph (TPG) config

The 32B judge can later be plugged into the same reflection schema produced
by `build_reflections`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LOW_SIGNAL_PATTERNS = (
    "captcha",
    "forbidden",
    "403",
    "access denied",
    "timed out",
    "readtimeout",
    "[error]",
    "[proxy-error]",
    "[jina-error]",
    "[harness] all results",
    "[harness] blocked",
    "low-signal",
    "just a moment",
)

SOCIAL_OR_NOISY_DOMAINS = (
    "youtube.com",
    "youtu.be",
    "reddit.com",
    "x.com",
    "twitter.com",
    "instagram.com",
    "facebook.com",
    "tiktok.com",
    "pinterest.com",
    "etsy.com",
)


@dataclass
class TrajectoryMetrics:
    task_id: str
    path: str
    turns: int
    steps: int
    assistant_turns: int
    tool_results: int
    tool_calls_requested: int
    search_calls: int
    browser_calls: int
    duplicate_adjacent_queries: int
    low_signal_tool_results: int
    critic_calls: int
    critic_bad: int
    forced_reflections: int
    state_prompts: int
    final_answer: str
    has_final_answer: bool
    low_confidence: bool
    max_total_tokens: int
    approx_context_tokens: int
    elapsed_seconds: float | None
    failure_types: list[str]


@dataclass
class ReflectionMemory:
    memory_id: str
    task_id: str
    source_path: str
    failure_type: str
    root_cause: str
    lesson: str
    applicable_when: list[str]
    avoid: list[str]
    prefer: list[str]
    source_steps: list[int]
    blame_nodes: list[str]
    confidence: float
    created_at: float


def safe_text(value: Any, limit: int | None = None) -> str:
    if value is None:
        out = ""
    elif isinstance(value, str):
        out = value
    else:
        out = json.dumps(value, ensure_ascii=False, sort_keys=True)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    if limit is not None and len(out) > limit:
        return out[: max(0, limit - 3)] + "..."
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} invalid JSON: {exc}") from exc
        if isinstance(obj, dict):
            entries.append(obj)
    return entries


def find_trajectory_files(path: Path, limit: int | None = None) -> list[Path]:
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("*.jsonl"))
    if limit is not None:
        files = files[:limit]
    return files


def entry_step(entry: dict[str, Any]) -> int:
    try:
        return int(entry.get("step_id") or 0)
    except Exception:
        return 0


def extract_tool_calls(entry: dict[str, Any]) -> list[dict[str, Any]]:
    calls = entry.get("tool_calls") or []
    if not isinstance(calls, list):
        return []
    out: list[dict[str, Any]] = []
    for call in calls:
        if isinstance(call, dict):
            out.append(call)
    return out


def extract_tool_name_from_call(call: dict[str, Any]) -> str:
    fn = call.get("function") or {}
    if isinstance(fn, dict):
        return str(fn.get("name") or "")
    return ""


def extract_fn_name(entry: dict[str, Any]) -> str:
    name = entry.get("fn_name")
    if name:
        return str(name)
    calls = extract_tool_calls(entry)
    if calls:
        return extract_tool_name_from_call(calls[0])
    return ""


def extract_query(entry: dict[str, Any]) -> str:
    args = entry.get("fn_args")
    if isinstance(args, dict) and args.get("query"):
        return str(args.get("query") or "")
    calls = extract_tool_calls(entry)
    for call in calls:
        fn = call.get("function") or {}
        if not isinstance(fn, dict):
            continue
        raw_args = fn.get("arguments")
        if not raw_args:
            continue
        try:
            parsed = json.loads(raw_args)
        except Exception:
            continue
        if isinstance(parsed, dict) and parsed.get("query"):
            return str(parsed.get("query") or "")
    return ""


def keyword_set(query: str) -> set[str]:
    normalized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", " ", query.lower())
    return {w for w in normalized.split() if len(w) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def count_adjacent_duplicate_queries(queries: list[str], threshold: float = 0.50) -> int:
    total = 0
    last_kw: set[str] | None = None
    for query in queries:
        kw = keyword_set(query)
        if last_kw is not None and jaccard(last_kw, kw) >= threshold:
            total += 1
        last_kw = kw
    return total


def looks_low_signal(entry: dict[str, Any]) -> bool:
    content = safe_text(entry.get("content")).lower()
    if not content.strip():
        return True
    if any(pattern in content for pattern in LOW_SIGNAL_PATTERNS):
        return True
    if any(domain in content for domain in SOCIAL_OR_NOISY_DOMAINS):
        return True
    try:
        parsed = json.loads(entry.get("content") or "")
    except Exception:
        parsed = None
    if isinstance(parsed, list) and parsed:
        useful = 0
        for item in parsed:
            if not isinstance(item, dict):
                continue
            blob = safe_text(item).lower()
            if not any(pattern in blob for pattern in LOW_SIGNAL_PATTERNS):
                useful += 1
        return useful == 0
    return False


def parse_critic_judgment(entry: dict[str, Any]) -> str | None:
    if entry.get("judgment"):
        return str(entry.get("judgment")).upper()
    content = safe_text(entry.get("content"))
    if "[NEG_CRITIC]" not in content and not entry.get("neg_critic"):
        return None
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            judgment = obj.get("judgment") or obj.get("verdict")
            if judgment:
                return str(judgment).upper()
        except Exception:
            pass
    if "BAD" in content.upper():
        return "BAD"
    if "GOOD" in content.upper():
        return "GOOD"
    return "UNKNOWN"


def extract_final_answer(entries: list[dict[str, Any]]) -> tuple[str, bool]:
    for entry in reversed(entries):
        if entry.get("role") != "assistant":
            continue
        content = safe_text(entry.get("content"))
        if not content.strip():
            continue
        match = re.search(r"<answer(?:\s+[^>]*)?>(.*?)</answer>", content, flags=re.DOTALL | re.I)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip(), True
        return re.sub(r"\s+", " ", content).strip(), False
    return "", False


def approx_tokens_for_entries(entries: list[dict[str, Any]]) -> int:
    chars = 0
    for entry in entries:
        chars += len(safe_text(entry.get("content")))
        chars += len(safe_text(entry.get("reasoning_content")))
    return math.ceil(chars / 4)


def infer_elapsed_seconds(entries: list[dict[str, Any]]) -> float | None:
    timestamps = []
    for entry in entries:
        try:
            timestamps.append(float(entry.get("timestamp")))
        except Exception:
            pass
    if len(timestamps) < 2:
        return None
    return round(max(timestamps) - min(timestamps), 3)


def analyze_trajectory(path: Path) -> tuple[TrajectoryMetrics, list[ReflectionMemory]]:
    entries = load_jsonl(path)
    assistant_entries = [e for e in entries if e.get("role") == "assistant"]
    tool_entries = [e for e in entries if e.get("role") == "tool"]
    user_entries = [e for e in entries if e.get("role") == "user"]
    tool_calls_requested = sum(len(extract_tool_calls(e)) for e in assistant_entries)
    search_queries = [extract_query(e) for e in tool_entries if extract_fn_name(e) == "search_text" and extract_query(e)]
    search_calls = sum(1 for e in tool_entries if extract_fn_name(e) in {"search_text", "search_image"})
    browser_calls = sum(1 for e in tool_entries if extract_fn_name(e).startswith("browser_"))
    critic_judgments = [parse_critic_judgment(e) for e in user_entries]
    critic_judgments = [j for j in critic_judgments if j]
    final_answer, tagged_answer = extract_final_answer(entries)
    low_conf = "confidence=\"low\"" in safe_text(entries[-1].get("content") if entries else "").lower()
    low_conf = low_conf or final_answer.lower().startswith("[low_confidence]")
    low_conf = low_conf or "unable to determine" in final_answer.lower()
    token_values = []
    for entry in assistant_entries:
        try:
            token_values.append(int(entry.get("total_tokens") or 0))
        except Exception:
            pass
    duplicate_count = count_adjacent_duplicate_queries(search_queries)
    low_signal_count = sum(1 for e in tool_entries if looks_low_signal(e))
    forced_reflections = sum(1 for e in user_entries if "[FORCED_REFLECTION]" in safe_text(e.get("content")))
    state_prompts = sum(1 for e in user_entries if "[STATE=" in safe_text(e.get("content")))
    failure_types = infer_failure_types(
        steps=max((entry_step(e) for e in entries), default=0),
        search_calls=search_calls,
        duplicate_count=duplicate_count,
        low_signal_count=low_signal_count,
        critic_bad=sum(1 for j in critic_judgments if j == "BAD"),
        forced_reflections=forced_reflections,
        low_confidence=low_conf,
        has_final_answer=bool(final_answer),
    )
    metrics = TrajectoryMetrics(
        task_id=path.stem,
        path=str(path),
        turns=len(entries),
        steps=max((entry_step(e) for e in entries), default=0),
        assistant_turns=len(assistant_entries),
        tool_results=len(tool_entries),
        tool_calls_requested=tool_calls_requested,
        search_calls=search_calls,
        browser_calls=browser_calls,
        duplicate_adjacent_queries=duplicate_count,
        low_signal_tool_results=low_signal_count,
        critic_calls=len(critic_judgments),
        critic_bad=sum(1 for j in critic_judgments if j == "BAD"),
        forced_reflections=forced_reflections,
        state_prompts=state_prompts,
        final_answer=final_answer[:300],
        has_final_answer=bool(final_answer) or tagged_answer,
        low_confidence=low_conf,
        max_total_tokens=max(token_values) if token_values else 0,
        approx_context_tokens=approx_tokens_for_entries(entries),
        elapsed_seconds=infer_elapsed_seconds(entries),
        failure_types=failure_types,
    )
    return metrics, build_reflections(metrics, entries)


def infer_failure_types(
    steps: int,
    search_calls: int,
    duplicate_count: int,
    low_signal_count: int,
    critic_bad: int,
    forced_reflections: int,
    low_confidence: bool,
    has_final_answer: bool,
) -> list[str]:
    failures = []
    if duplicate_count >= 2:
        failures.append("duplicate_search")
    if low_signal_count >= 2:
        failures.append("low_signal_retrieval")
    if critic_bad >= 3:
        failures.append("persistent_critic_bad")
    if forced_reflections >= 2:
        failures.append("reflection_not_resolving")
    if search_calls >= 15:
        failures.append("over_searching")
    if steps >= 15 and (low_confidence or not has_final_answer):
        failures.append("budget_exhaustion")
    if low_confidence:
        failures.append("low_confidence_answer")
    return failures


def steps_for_entries(entries: list[dict[str, Any]], predicate) -> list[int]:
    out = []
    for entry in entries:
        if predicate(entry):
            out.append(entry_step(entry))
    return sorted(set(out))


def build_reflections(metrics: TrajectoryMetrics, entries: list[dict[str, Any]]) -> list[ReflectionMemory]:
    reflections: list[ReflectionMemory] = []
    now = time.time()

    def add(
        failure_type: str,
        root_cause: str,
        lesson: str,
        applicable_when: list[str],
        avoid: list[str],
        prefer: list[str],
        source_steps: list[int],
        blame_nodes: list[str],
        confidence: float,
    ) -> None:
        reflections.append(
            ReflectionMemory(
                memory_id=str(uuid.uuid4())[:8],
                task_id=metrics.task_id,
                source_path=metrics.path,
                failure_type=failure_type,
                root_cause=root_cause,
                lesson=lesson,
                applicable_when=applicable_when,
                avoid=avoid,
                prefer=prefer,
                source_steps=source_steps,
                blame_nodes=blame_nodes,
                confidence=confidence,
                created_at=now,
            )
        )

    if "duplicate_search" in metrics.failure_types:
        add(
            "duplicate_search",
            "Adjacent search queries share too many keywords, so the agent is rephrasing instead of changing clue chains.",
            "After two similar or low-signal searches, force a query using unused hard constraints from the question.",
            ["search task", "no verified candidate", "recent queries overlap"],
            ["synonym-only query rewrites", "same entity plus generic terms"],
            ["unused exact date/count/name", "different starting entity", "one independent verification query"],
            steps_for_entries(entries, lambda e: extract_query(e) != ""),
            ["tool_policy.search", "reflection_policy"],
            min(0.95, 0.55 + 0.05 * metrics.duplicate_adjacent_queries),
        )
    if "low_signal_retrieval" in metrics.failure_types:
        add(
            "low_signal_retrieval",
            "Tool results contain blocked, social, timeout, or empty pages that do not reduce uncertainty.",
            "Prefer snippet-only search first; use browser only for reliable source URLs and pivot after repeated low-signal results.",
            ["search results include captcha/social/timeout", "tool output is empty or blocked"],
            ["fetching broad noisy pages", "social/video result chains"],
            ["authoritative domains", "short constrained queries", "browser_parallel on reliable URLs"],
            steps_for_entries(entries, looks_low_signal),
            ["tool_policy.search", "tool_policy.browser"],
            min(0.95, 0.55 + 0.04 * metrics.low_signal_tool_results),
        )
    if "persistent_critic_bad" in metrics.failure_types:
        add(
            "persistent_critic_bad",
            "The critic repeatedly marks the trajectory as BAD, but the policy does not convert feedback into a decisive pivot.",
            "A BAD critic event should produce a typed repair target and block the same failure mode for the next few steps.",
            ["critic emits BAD repeatedly", "forced reflection does not change behavior"],
            ["generic reflection without a concrete policy constraint"],
            ["typed failure label", "one forbidden action pattern", "one required next action family"],
            steps_for_entries(entries, lambda e: parse_critic_judgment(e) == "BAD"),
            ["critic_policy", "reflection_policy", "edge.critic_to_reflection"],
            min(0.98, 0.50 + 0.03 * metrics.critic_bad),
        )
    if "over_searching" in metrics.failure_types or "budget_exhaustion" in metrics.failure_types:
        add(
            "budget_or_over_searching",
            "The trajectory spends many steps searching without committing to a candidate or answer.",
            "Track candidate confidence and force answer/verification once evidence stops increasing.",
            ["steps near max budget", "many search calls", "candidate evidence is stale"],
            ["open-ended exploratory searching after step budget is mostly used"],
            ["candidate table", "evidence gain counter", "forced concise answer with confidence"],
            list(range(max(1, metrics.steps - 4), metrics.steps + 1)),
            ["state_machine", "reflection_policy"],
            0.75,
        )
    return reflections


def summarize_metrics(metrics: list[TrajectoryMetrics]) -> dict[str, Any]:
    if not metrics:
        return {"count": 0}

    def avg(field: str) -> float:
        vals = [getattr(m, field) for m in metrics]
        vals = [v for v in vals if isinstance(v, (int, float)) and v is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    failures: dict[str, int] = {}
    for item in metrics:
        for failure in item.failure_types:
            failures[failure] = failures.get(failure, 0) + 1
    elapsed = [m.elapsed_seconds for m in metrics if m.elapsed_seconds is not None]
    return {
        "count": len(metrics),
        "avg_steps": avg("steps"),
        "avg_tool_results": avg("tool_results"),
        "avg_search_calls": avg("search_calls"),
        "avg_duplicate_adjacent_queries": avg("duplicate_adjacent_queries"),
        "avg_low_signal_tool_results": avg("low_signal_tool_results"),
        "avg_critic_calls": avg("critic_calls"),
        "avg_critic_bad": avg("critic_bad"),
        "avg_approx_context_tokens": avg("approx_context_tokens"),
        "max_steps": max(m.steps for m in metrics),
        "max_search_calls": max(m.search_calls for m in metrics),
        "low_confidence_count": sum(1 for m in metrics if m.low_confidence),
        "empty_answer_count": sum(1 for m in metrics if not m.has_final_answer),
        "elapsed_seconds_avg": round(statistics.mean(elapsed), 3) if elapsed else None,
        "failure_counts": dict(sorted(failures.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def write_analysis_outputs(
    metrics: list[TrajectoryMetrics],
    memories: list[ReflectionMemory],
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = out_dir / "trajectory_metrics.json"
    metrics_csv = out_dir / "trajectory_metrics.csv"
    memories_jsonl = out_dir / "reflection_memories.jsonl"
    summary_json = out_dir / "summary.json"
    metrics_json.write_text(
        json.dumps([asdict(m) for m in metrics], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if metrics:
        with metrics_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(metrics[0]).keys()))
            writer.writeheader()
            for metric in metrics:
                row = asdict(metric)
                row["failure_types"] = "|".join(row["failure_types"])
                writer.writerow(row)
    else:
        metrics_csv.write_text("", encoding="utf-8")
    with memories_jsonl.open("w", encoding="utf-8") as f:
        for memory in memories:
            f.write(json.dumps(asdict(memory), ensure_ascii=False) + "\n")
    summary_json.write_text(json.dumps(summarize_metrics(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "metrics_json": metrics_json,
        "metrics_csv": metrics_csv,
        "memories_jsonl": memories_jsonl,
        "summary_json": summary_json,
    }


def mermaid_escape(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = text.replace('"', "'")
    text = text.replace("[", "(").replace("]", ")")
    text = text.replace("{", "(").replace("}", ")")
    return text


def render_mermaid(entries: list[dict[str, Any]], max_nodes: int = 80) -> str:
    lines = [
        "flowchart TD",
        "  classDef system fill:#eef2ff,stroke:#4f46e5,color:#111827;",
        "  classDef user fill:#ecfeff,stroke:#0891b2,color:#111827;",
        "  classDef assistant fill:#fef3c7,stroke:#d97706,color:#111827;",
        "  classDef tool fill:#dcfce7,stroke:#16a34a,color:#111827;",
        "  classDef critic fill:#fee2e2,stroke:#dc2626,color:#111827;",
        "  classDef final fill:#f5f3ff,stroke:#7c3aed,color:#111827;",
    ]
    clipped = entries[:max_nodes]
    for idx, entry in enumerate(clipped):
        role = str(entry.get("role") or "unknown")
        step = entry_step(entry)
        content = safe_text(entry.get("content"), 120)
        fn = extract_fn_name(entry)
        query = extract_query(entry)
        label_bits = [f"{idx}. {role}", f"step={step}"]
        if fn:
            label_bits.append(fn)
        if query:
            label_bits.append("q=" + query[:60])
        elif content:
            label_bits.append(content)
        label = mermaid_escape("\\n".join(label_bits))
        lines.append(f'  n{idx}["{label}"]')
        class_name = role
        if parse_critic_judgment(entry):
            class_name = "critic"
        if role == "assistant" and "<answer" in safe_text(entry.get("content")).lower():
            class_name = "final"
        if class_name not in {"system", "user", "assistant", "tool", "critic", "final"}:
            class_name = "user"
        lines.append(f"  class n{idx} {class_name};")
        if idx > 0:
            lines.append(f"  n{idx - 1} --> n{idx}")
    if len(entries) > max_nodes:
        lines.append(f'  more["... clipped {len(entries) - max_nodes} more events ..."]')
        lines.append(f"  n{len(clipped) - 1} --> more")
    return "\n".join(lines) + "\n"


def initial_tpg() -> dict[str, Any]:
    return {
        "version": 1,
        "name": "harness_plan_react_tpgo_seed",
        "description": "Initial Textual Parameter Graph for the current ReAct harness.",
        "nodes": [
            {
                "id": "system_prompt",
                "type": "prompt",
                "owner_file": "task_runner_plan_react.py",
                "content_summary": "Base task-solving rules, search discipline, answer format.",
                "optimizable": True,
            },
            {
                "id": "state_machine",
                "type": "workflow",
                "owner_file": "task_runner_plan_react.py",
                "content_summary": "S0_PARSE -> S1_SUBJECT -> S2_EVENT -> S3_DETAIL -> S4_DONE.",
                "optimizable": True,
            },
            {
                "id": "tool_policy.search",
                "type": "tool_policy",
                "owner_file": "task_runner_plan_react.py",
                "content_summary": "Search query quality, duplicate blocking, low-signal filtering, fetch policy.",
                "optimizable": True,
            },
            {
                "id": "tool_policy.browser",
                "type": "tool_policy",
                "owner_file": "tools/browser_tool.py",
                "content_summary": "Browser navigation, text extraction, click/type, parallel page handling.",
                "optimizable": True,
            },
            {
                "id": "critic_policy",
                "type": "judge",
                "owner_file": "task_runner_plan_react_negcrit.py",
                "content_summary": "External 32B trajectory evaluator prompt and output schema.",
                "optimizable": True,
            },
            {
                "id": "reflection_policy",
                "type": "reflection",
                "owner_file": "task_runner_plan_react_negcrit.py",
                "content_summary": "Forced reflection prompt and BAD-critic response.",
                "optimizable": True,
            },
            {
                "id": "memory_policy",
                "type": "memory",
                "owner_file": "tpgo/tpgo_tools.py",
                "content_summary": "Reflection memory extraction, retrieval, and future prompt injection.",
                "optimizable": True,
            },
        ],
        "edges": [
            {"id": "state_to_tool", "source": "state_machine", "target": "tool_policy.search", "relation": "constrains"},
            {"id": "tool_to_critic", "source": "tool_policy.search", "target": "critic_policy", "relation": "produces_events_for"},
            {"id": "critic_to_reflection", "source": "critic_policy", "target": "reflection_policy", "relation": "triggers"},
            {"id": "reflection_to_memory", "source": "reflection_policy", "target": "memory_policy", "relation": "writes_lessons"},
            {"id": "memory_to_system", "source": "memory_policy", "target": "system_prompt", "relation": "injects_compact_constraints"},
        ],
        "edit_operations": ["REWRITE_NODE", "ADD_NODE", "PRUNE_EDGE", "ADD_EDGE", "ADJUST_THRESHOLD"],
    }


def print_summary(summary: dict[str, Any], outputs: dict[str, Path] | None = None) -> None:
    print("\nTPGO analysis summary")
    print("=" * 60)
    for key, value in summary.items():
        if key == "failure_counts":
            print("failure_counts:")
            for failure, count in value.items():
                print(f"  - {failure}: {count}")
        else:
            print(f"{key}: {value}")
    if outputs:
        print("\noutputs:")
        for key, path in outputs.items():
            print(f"  - {key}: {path}")


def command_analyze(args: argparse.Namespace) -> None:
    files = find_trajectory_files(Path(args.traj_dir), args.limit)
    metrics: list[TrajectoryMetrics] = []
    memories: list[ReflectionMemory] = []
    for path in files:
        try:
            metric, reflection = analyze_trajectory(path)
        except Exception as exc:
            print(f"[WARN] skip {path}: {exc}")
            continue
        metrics.append(metric)
        memories.extend(reflection)
    outputs = write_analysis_outputs(metrics, memories, Path(args.out_dir))
    summary = summarize_metrics(metrics)
    print_summary(summary, outputs)
    worst = sorted(
        metrics,
        key=lambda m: (
            len(m.failure_types),
            m.duplicate_adjacent_queries,
            m.low_signal_tool_results,
            m.search_calls,
            m.steps,
        ),
        reverse=True,
    )[: args.top_k]
    print("\nworst_cases:")
    for item in worst:
        print(
            f"  - {item.task_id}: steps={item.steps}, search={item.search_calls}, "
            f"dup={item.duplicate_adjacent_queries}, low_signal={item.low_signal_tool_results}, "
            f"critic_bad={item.critic_bad}, failures={','.join(item.failure_types) or 'none'}"
        )


def command_graph(args: argparse.Namespace) -> None:
    path = Path(args.traj)
    entries = load_jsonl(path)
    out = Path(args.out) if args.out else Path(args.out_dir) / f"{path.stem}.mmd"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_mermaid(entries, max_nodes=args.max_nodes), encoding="utf-8")
    metric, reflections = analyze_trajectory(path)
    print(f"graph: {out}")
    print(
        f"metrics: task={metric.task_id}, steps={metric.steps}, search={metric.search_calls}, "
        f"dup={metric.duplicate_adjacent_queries}, low_signal={metric.low_signal_tool_results}, "
        f"failures={','.join(metric.failure_types) or 'none'}"
    )
    print(f"reflections: {len(reflections)}")


def command_init_tpg(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(initial_tpg(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"initial TPG written: {out}")


def command_demo(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    command_init_tpg(argparse.Namespace(out=str(out_dir / "current_tpg.json")))
    command_analyze(
        argparse.Namespace(
            traj_dir=args.traj_dir,
            out_dir=str(out_dir),
            limit=args.limit,
            top_k=args.top_k,
        )
    )
    files = find_trajectory_files(Path(args.traj_dir), args.limit)
    if files:
        graph_out = out_dir / f"{files[0].stem}.mmd"
        command_graph(
            argparse.Namespace(
                traj=str(files[0]),
                out=str(graph_out),
                out_dir=str(out_dir),
                max_nodes=args.max_nodes,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TPGO trajectory analysis and graph tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_analyze = sub.add_parser("analyze", help="Analyze trajectory files and write metrics/memories")
    p_analyze.add_argument("--traj-dir", required=True, help="Trajectory file or directory")
    p_analyze.add_argument("--out-dir", default="tpgo/outputs", help="Output directory")
    p_analyze.add_argument("--limit", type=int, default=None, help="Optional max number of files")
    p_analyze.add_argument("--top-k", type=int, default=10, help="Worst cases to print")
    p_analyze.set_defaults(func=command_analyze)

    p_graph = sub.add_parser("graph", help="Render one trajectory as Mermaid")
    p_graph.add_argument("--traj", required=True, help="Trajectory JSONL path")
    p_graph.add_argument("--out", default=None, help="Output .mmd path")
    p_graph.add_argument("--out-dir", default="tpgo/outputs/graphs", help="Output directory if --out omitted")
    p_graph.add_argument("--max-nodes", type=int, default=80, help="Maximum trajectory nodes to render")
    p_graph.set_defaults(func=command_graph)

    p_tpg = sub.add_parser("init-tpg", help="Write initial Textual Parameter Graph config")
    p_tpg.add_argument("--out", default="tpgo/current_tpg.json")
    p_tpg.set_defaults(func=command_init_tpg)

    p_demo = sub.add_parser("demo", help="Run init/analyze/graph on a small trajectory slice")
    p_demo.add_argument("--traj-dir", required=True, help="Trajectory file or directory")
    p_demo.add_argument("--out-dir", default="tpgo/outputs/demo")
    p_demo.add_argument("--limit", type=int, default=5)
    p_demo.add_argument("--top-k", type=int, default=5)
    p_demo.add_argument("--max-nodes", type=int, default=60)
    p_demo.set_defaults(func=command_demo)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

