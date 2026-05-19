"""
Offline checks for the reflection module.

This script does not call the main LLM. It verifies whether the reflection
hook can detect representative failures, generate actionable feedback, and
persist lessons that can be retrieved by later tasks.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from reflection_module.core import ReflectionConfig, ReflectionManager, summarize_recent, validate_critic_output
from reflection_module.analyze_trajectories import analyze_dir
from reflection_module.compare_runs import build_report


CASES = [
    {
        "name": "textual_tool_call",
        "instruction": "Find the birthplace of a composer's spouse.",
        "assistant": "",
        "reasoning": "<tool_call><function=search_text><parameter=query>composer spouse birthplace</parameter></function></tool_call>",
        "expected_type": "malformed_tool_call",
    },
    {
        "name": "empty_assistant",
        "instruction": "Answer whether two stations are in the same country.",
        "assistant": "",
        "reasoning": "I should search both entities.",
        "expected_type": "no_action",
    },
    {
        "name": "tool_timeout",
        "instruction": "Search a page and extract the answer.",
        "tool_result": "[ERROR] Tool 'browser_get_text' raised: TimeoutError: timed out after 15s",
        "tool_name": "browser_get_text",
        "tool_args": {"max_chars": 5000},
        "expected_type": "tool_timeout",
    },
    {
        "name": "tool_error",
        "instruction": "Use search to answer a multi-hop question.",
        "tool_result": "[ERROR] Unknown tool: google_search",
        "tool_name": "google_search",
        "tool_args": {"q": "Ray Taylor birth date"},
        "expected_type": "tool_error",
    },
    {
        "name": "max_steps",
        "instruction": "Compare two film directors by age.",
        "max_steps": True,
        "expected_type": "budget_exhausted",
    },
]


def run_eval(memory_path: str) -> dict:
    path = Path(memory_path)
    if path.exists():
        path.unlink()

    manager = ReflectionManager(ReflectionConfig(memory_path=memory_path))
    passed = 0
    rows = []

    for idx, case in enumerate(CASES, start=1):
        recent = summarize_recent(
            [
                {"role": "user", "content": case["instruction"]},
                {"role": "assistant", "content": case.get("assistant", "")},
            ]
        )
        if case.get("max_steps"):
            event = manager.max_steps_failure(
                task_id=f"eval_{idx}",
                step_id=6,
                instruction=case["instruction"],
                recent_messages=recent,
            )
        elif "tool_result" in case:
            event = manager.detect_tool_failure(
                task_id=f"eval_{idx}",
                step_id=idx,
                instruction=case["instruction"],
                tool_name=case["tool_name"],
                tool_args=case["tool_args"],
                tool_result=case["tool_result"],
                recent_messages=recent,
            )
        else:
            event = manager.detect_assistant_failure(
                task_id=f"eval_{idx}",
                step_id=idx,
                instruction=case["instruction"],
                content=case["assistant"],
                reasoning_content=case["reasoning"],
                tool_calls=None,
                finish_reason=None,
                recent_messages=recent,
            )

        detected = event is not None
        record = manager.reflect(event) if event else None
        feedback = manager.to_feedback_message(record) if record else ""
        ok = bool(
            detected
            and record
            and record.failure_type == case["expected_type"]
            and record.root_cause
            and record.correction_strategy
            and record.next_prompt
            and "不要简单重复" in feedback
        )
        passed += int(ok)
        rows.append(
            {
                "case": case["name"],
                "expected": case["expected_type"],
                "detected": detected,
                "actual": record.failure_type if record else None,
                "ok": ok,
                "feedback": feedback,
            }
        )

    retrieval = manager.build_system_appendix("Compare two film directors by age in a multi-hop question.")
    smoke = run_smoke_tests()
    return {
        "total": len(CASES),
        "passed": passed,
        "pass_rate": round(passed / len(CASES), 3),
        "memory_path": str(path),
        "memory_items": len(manager.memory.load()),
        "retrieval_non_empty": bool(retrieval),
        "smoke_tests": smoke,
        "smoke_passed": all(item["ok"] for item in smoke),
        "cases": rows,
    }


def run_smoke_tests() -> list[dict]:
    checks = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # 1. memory dedup
        memory_path = tmp / "memory.jsonl"
        manager = ReflectionManager(ReflectionConfig(memory_path=str(memory_path)))
        event = manager.max_steps_failure(
            task_id="dedup_task",
            step_id=6,
            instruction="Compare two film directors by age.",
            recent_messages=[],
        )
        first = manager.reflect(event)
        second = manager.reflect(event)
        rows = manager.memory.load()
        checks.append(
            {
                "name": "memory_dedup",
                "ok": first.memory_written and not second.memory_written and len(rows) == 1,
                "detail": {"memory_items": len(rows)},
            }
        )

        # 2. old memory schema compatibility
        old_memory = tmp / "old_memory.jsonl"
        old_memory.write_text(
            json.dumps(
                {
                    "task_id": "old_task",
                    "failure_type": "tool_error",
                    "root_cause": "old schema root cause",
                    "correction_strategy": "change query after tool error",
                    "reusable_lesson": "avoid blind retry",
                    "next_prompt": "change the next action",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        old_manager = ReflectionManager(ReflectionConfig(memory_path=str(old_memory), min_relevance=0.01))
        hits = old_manager.memory.retrieve("tool error change query", limit=2, min_score=0.01)
        checks.append(
            {
                "name": "old_memory_schema_compatible",
                "ok": bool(hits) and "source_task_id" in hits[0] and "use_count" in hits[0],
                "detail": {"hits": len(hits)},
            }
        )

        # 3. analyze_trajectories on temporary trajectory
        traj_dir = tmp / "traj"
        traj_dir.mkdir()
        traj_file = traj_dir / "sample_task.jsonl"
        traj_rows = [
            {"timestamp": 1.0, "step_id": 0, "role": "system", "content": "s", "tool_call_id": None},
            {"timestamp": 2.0, "step_id": 0, "role": "user", "content": "u", "tool_call_id": None},
            {
                "timestamp": 3.0,
                "step_id": 1,
                "role": "assistant",
                "content": "",
                "tool_call_id": None,
                "reasoning_content": "<tool_call><function=search_text></function></tool_call>",
            },
            {
                "timestamp": 4.0,
                "step_id": 1,
                "role": "tool",
                "content": "[ERROR] Unknown tool: google_search",
                "tool_call_id": "x",
                "fn_name": "google_search",
                "fn_args": {"query": "same query"},
            },
        ]
        traj_file.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in traj_rows) + "\n",
            encoding="utf-8",
        )
        analysis = analyze_dir(traj_dir)
        checks.append(
            {
                "name": "analyze_trajectories_tmp",
                "ok": analysis["num_tasks"] == 1 and analysis["num_failed_like"] == 1,
                "detail": analysis["failure_type_counts"],
            }
        )

        # 4. compare_runs friendly missing reflection result
        baseline = tmp / "baseline.jsonl"
        baseline.write_text(
            json.dumps({"correct": True, "steps": 2, "prediction": "ok"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        missing_report = build_report(baseline, tmp / "missing_reflection.jsonl")
        checks.append(
            {
                "name": "compare_missing_reflection_friendly",
                "ok": missing_report.get("ok") is False and "how_to_run" in missing_report,
                "detail": missing_report,
            }
        )

        # 5. critic validator rejects direct-answer output
        try:
            validate_critic_output(
                json.dumps(
                    {
                        "failure_type": "tool_error",
                        "root_cause": "tool returned an error",
                        "evidence": ["[ERROR] bad args"],
                        "correction_strategy": "fix the tool args",
                        "next_prompt": "call the tool with legal args",
                        "next_action_type": "call_search_text",
                        "should_retry_same_action": False,
                        "memory_lesson": "tool args must match schema",
                        "applicable_task_types": ["2wiki_text"],
                        "confidence": 0.8,
                        "answer": "the answer is X",
                    }
                )
            )
            validator_ok = False
        except ValueError:
            validator_ok = True
        checks.append({"name": "critic_validator_rejects_answer", "ok": validator_ok, "detail": {}})

        # 6. LLM critic path falls back cleanly without REFLECTION_API_KEY
        fallback_manager = ReflectionManager(
            ReflectionConfig(
                use_llm=True,
                api_key="",
                memory_path=str(tmp / "fallback_memory.jsonl"),
            )
        )
        fallback_event = fallback_manager.max_steps_failure(
            task_id="fallback_task",
            step_id=6,
            instruction="Compare two films by release year.",
            recent_messages=[],
            task_type="2wiki_text",
        )
        fallback_record = fallback_manager.reflect(fallback_event)
        checks.append(
            {
                "name": "no_api_key_rule_fallback",
                "ok": bool(fallback_record.critic_model == "rule_fallback" and fallback_record.root_cause),
                "detail": {"critic_model": fallback_record.critic_model},
            }
        )

    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memory-path",
        default="reflection_module/tmp_eval_memory.jsonl",
        help="Temporary JSONL memory path for offline evaluation.",
    )
    parser.add_argument(
        "--report",
        default="reflection_module/eval_report.json",
        help="Where to write the JSON report.",
    )
    args = parser.parse_args()

    report = run_eval(args.memory_path)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
