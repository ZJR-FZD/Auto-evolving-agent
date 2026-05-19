"""Memory-augmented Reflection Skill for the Qwen ReAct harness.

This file is a small integration layer around ``reflection_module.core``.
It is intentionally framework-light so it can be dropped into the existing
Qwen3.5-9B + Qwen3-30B-A3B harness without changing the official runner
interface.

High-level flow
---------------
1. Harness detects a real failure signal from the trajectory.
2. ReflectionSkill stores the failed trajectory summary in a JSONL database.
3. It first retrieves similar failed cases and reuses their validated strategy.
4. If no strong match exists, it calls ReflectionManager, which can use:
   memory -> Qwen3.5-9B light critic -> Qwen3-30B-A3B deep critic -> rules.
5. Optional async mode returns a fast rule strategy immediately while the
   30B critic writes a reusable lesson in the background.

The critic never receives gold answers and never directly answers the task.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .core import (
    FailureEvent,
    ReflectionConfig,
    ReflectionManager,
    ReflectionRecord,
    summarize_recent,
)


TEXT_TASK_TYPES = {"2wiki_text", "benchmark_text"}
MULTIMODAL_TASK_TYPES = {"simplevqa_multimodal", "benchmark_multimodal"}


@dataclass
class ReflectionSkillConfig:
    """Runtime configuration for the higher-level Reflection Skill."""

    failure_db_path: str = os.getenv(
        "REFLECTION_FAILURE_DB_PATH",
        "reflection_memory/failure_trajectories.jsonl",
    )
    memory_path: str = os.getenv(
        "REFLECTION_MEMORY_PATH",
        "reflection_memory/reflection_memory.jsonl",
    )
    reuse_threshold: float = float(os.getenv("REFLECTION_SKILL_REUSE_THRESHOLD", "0.42"))
    max_db_hits: int = int(os.getenv("REFLECTION_SKILL_MAX_DB_HITS", "3"))
    async_deep_reflection: bool = os.getenv("REFLECTION_ASYNC_CRITIC", "0") == "1"
    async_workers: int = int(os.getenv("REFLECTION_ASYNC_WORKERS", "1"))


@dataclass
class FailureTrajectoryRecord:
    """Compact failure trajectory persisted for future strategy reuse."""

    record_id: str
    task_id: str
    task_type: str
    failure_type: str
    task_features: dict[str, Any]
    tool_history: list[dict[str, Any]]
    reasoning_chain: list[str]
    final_failure: str
    correction_strategy: str = ""
    next_prompt: str = ""
    source: str = "trajectory"
    created_at: float = field(default_factory=time.time)
    use_count: int = 0


@dataclass
class SkillDecision:
    """Decision returned to the harness after a failure."""

    record: ReflectionRecord
    source: str
    db_hits: list[dict[str, Any]] = field(default_factory=list)
    async_submitted: bool = False


class FailureTrajectoryDB:
    """Append-only JSONL database of failed trajectories.

    The database stores summaries rather than full trajectories to keep lookup
    cheap and avoid bloating prompts.  Full trajectories remain in the harness
    trajectory directory.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: FailureTrajectoryRecord) -> None:
        row = asdict(record)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(self._normalize(obj))
        return rows

    def query(
        self,
        event: FailureEvent,
        limit: int = 3,
        min_score: float = 0.35,
    ) -> list[dict[str, Any]]:
        query_terms = _tokens(_event_query_text(event))
        if not query_terms:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self.load():
            row_terms = _tokens(_row_query_text(row))
            if not row_terms:
                continue
            score = len(query_terms & row_terms) / max(len(query_terms | row_terms), 1)
            if row.get("task_type") == event.task_type:
                score += 0.25
            if row.get("failure_type") == event.failure_type:
                score += 0.25
            if score >= min_score:
                row = dict(row)
                row["_score"] = round(score, 4)
                scored.append((score, row))

        scored.sort(key=lambda item: (item[0], item[1].get("created_at", 0)), reverse=True)
        return [row for _, row in scored[:limit]]

    def _normalize(self, row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        out.setdefault("record_id", str(uuid.uuid4()))
        out.setdefault("task_type", "general")
        out.setdefault("failure_type", "unknown")
        out.setdefault("task_features", {})
        out.setdefault("tool_history", [])
        out.setdefault("reasoning_chain", [])
        out.setdefault("final_failure", "")
        out.setdefault("correction_strategy", "")
        out.setdefault("next_prompt", "")
        out.setdefault("created_at", 0)
        out.setdefault("use_count", 0)
        return out


class ReflectionSkill:
    """Reusable skill that coordinates memory, failure DB, and critic calls."""

    def __init__(
        self,
        skill_config: Optional[ReflectionSkillConfig] = None,
        reflection_config: Optional[ReflectionConfig] = None,
    ):
        self.skill_config = skill_config or ReflectionSkillConfig()
        self.reflection_config = reflection_config or ReflectionConfig(
            memory_path=self.skill_config.memory_path,
            async_critic=self.skill_config.async_deep_reflection,
        )
        self.manager = ReflectionManager(self.reflection_config)
        self.failure_db = FailureTrajectoryDB(self.skill_config.failure_db_path)
        self._executor: ThreadPoolExecutor | None = None
        if self.skill_config.async_deep_reflection:
            self._executor = ThreadPoolExecutor(
                max_workers=max(1, self.skill_config.async_workers),
                thread_name_prefix="reflection_skill",
            )

    def task_prompt_appendix(self, task_type: str) -> str:
        """Task-type aware policy injected into the system prompt."""
        if task_type in TEXT_TASK_TYPES:
            return (
                "\n## Reflection Skill: Text Multi-hop Policy\n"
                "- Split the question into 2-3 entity/relation subgoals.\n"
                "- Prefer short search_text queries; avoid browser unless needed.\n"
                "- After mid-budget, produce a candidate answer from current evidence.\n"
                "- At final synthesis, return only <answer>...</answer>.\n"
            )
        if task_type in MULTIMODAL_TASK_TYPES:
            return (
                "\n## Reflection Skill: Multimodal Policy\n"
                "- First identify visual entities, text, scene, object, brand, or location.\n"
                "- If image_url is missing, do not call search_image; use text search.\n"
                "- Resolve conflicts between visual description and web evidence before answering.\n"
            )
        return "\n## Reflection Skill\n- Use previous failure lessons when a similar tool or reasoning error appears.\n"

    def on_task_start(self, instruction: str, task_type: str) -> str:
        """Return memory hints to inject before the first ReAct step."""
        return self.manager.build_system_appendix(instruction, task_type=task_type)

    def on_failure(
        self,
        event: FailureEvent,
        trajectory_rows: Iterable[dict[str, Any]],
    ) -> SkillDecision:
        """Handle one detected failure and return a strategy for the harness.

        This is the main hook used by a ReAct runner after detecting tool error,
        loop, max-step, malformed action, or weak visual grounding.
        """
        trajectory_summary = summarize_failure_trajectory(trajectory_rows)

        db_hits = self.failure_db.query(
            event,
            limit=self.skill_config.max_db_hits,
            min_score=0.20,
        )
        reusable = self._select_reusable_db_hit(db_hits)
        if reusable:
            record = self._record_from_db_hit(event, reusable)
            self._store_failure(event, trajectory_summary, record)
            return SkillDecision(record=record, source="failure_db_reuse", db_hits=db_hits)

        if self.skill_config.async_deep_reflection and self._executor is not None:
            fast = self.manager._reflect_with_rules(event)  # deterministic fallback, no LLM wait
            fast.critic_model = "rule_fallback_async_skill"
            self._store_failure(event, trajectory_summary, fast)
            self._submit_background_reflection(event, trajectory_summary)
            return SkillDecision(
                record=fast,
                source="rule_fast_async_deep_reflection",
                db_hits=db_hits,
                async_submitted=True,
            )

        record = self.manager.reflect(event)
        self._store_failure(event, trajectory_summary, record)
        return SkillDecision(record=record, source=record.critic_model, db_hits=db_hits)

    def feedback_message(self, decision: SkillDecision) -> str:
        """Format the decision as a user/tool feedback message for ReAct."""
        prefix = (
            f"[ReflectionSkill]\n"
            f"source={decision.source}, db_hits={len(decision.db_hits)}, "
            f"async_submitted={decision.async_submitted}\n"
        )
        return prefix + self.manager.to_feedback_message(decision.record)

    def _select_reusable_db_hit(self, hits: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        for row in hits:
            score = float(row.get("_score", 0.0) or 0.0)
            if score < self.skill_config.reuse_threshold:
                continue
            if not str(row.get("correction_strategy") or "").strip():
                continue
            if not str(row.get("next_prompt") or "").strip():
                continue
            return row
        return None

    def _record_from_db_hit(self, event: FailureEvent, row: dict[str, Any]) -> ReflectionRecord:
        strategy = str(row.get("correction_strategy") or "")
        next_prompt = str(row.get("next_prompt") or strategy)
        return ReflectionRecord(
            task_id=event.task_id,
            step_id=event.step_id,
            trigger=event.trigger,
            failure_type=event.failure_type,
            root_cause=f"Reused similar failed trajectory {row.get('record_id')} with score={row.get('_score')}.",
            correction_strategy=strategy,
            reusable_lesson=strategy,
            next_prompt=next_prompt,
            evidence=[str(row.get("final_failure") or event.evidence)[:240]],
            task_type=event.task_type,
            critic_model="failure_db_reuse",
            source_task_id=str(row.get("task_id") or ""),
            confidence=min(0.95, max(0.55, float(row.get("_score", 0.0) or 0.0))),
        )

    def _store_failure(
        self,
        event: FailureEvent,
        trajectory_summary: dict[str, Any],
        reflection: ReflectionRecord,
    ) -> None:
        task_features = extract_task_features(event.instruction, event.task_type, event.image_info)
        record = FailureTrajectoryRecord(
            record_id=str(uuid.uuid4()),
            task_id=event.task_id,
            task_type=event.task_type,
            failure_type=event.failure_type,
            task_features=task_features,
            tool_history=trajectory_summary.get("tool_history", []),
            reasoning_chain=trajectory_summary.get("reasoning_chain", []),
            final_failure=event.evidence,
            correction_strategy=reflection.correction_strategy,
            next_prompt=reflection.next_prompt,
            source=reflection.critic_model,
        )
        self.failure_db.append(record)

    def _submit_background_reflection(
        self,
        event: FailureEvent,
        trajectory_summary: dict[str, Any],
    ) -> Future | None:
        if self._executor is None:
            return None

        def job() -> None:
            record = self.manager.reflect(event)
            self._store_failure(event, trajectory_summary, record)

        return self._executor.submit(job)


def summarize_failure_trajectory(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Extract tool history and reasoning snippets from a trajectory JSONL."""
    tool_history: list[dict[str, Any]] = []
    reasoning_chain: list[str] = []
    for row in rows:
        role = str(row.get("role") or "")
        if role == "assistant":
            content = str(row.get("content") or "")
            reasoning = str(row.get("reasoning_content") or row.get("extra", {}).get("reasoning_content") or "")
            snippet = reasoning or content
            if snippet:
                reasoning_chain.append(_clip(snippet, 600))
            for tc in row.get("tool_calls", []) or row.get("extra", {}).get("tool_calls", []) or []:
                tool_history.append({"source": "assistant_tool_call", "tool_call": tc})
        elif role == "tool":
            tool_history.append(
                {
                    "source": "tool_result",
                    "fn_name": row.get("fn_name") or row.get("extra", {}).get("fn_name"),
                    "fn_args": row.get("fn_args") or row.get("extra", {}).get("fn_args"),
                    "content": _clip(str(row.get("content") or ""), 800),
                }
            )
    return {
        "tool_history": tool_history[-12:],
        "reasoning_chain": reasoning_chain[-8:],
    }


def extract_task_features(instruction: str, task_type: str, image_info: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Build lightweight task features for failure retrieval."""
    image_info = image_info or {}
    years = re.findall(r"\b(?:17|18|19|20)\d{2}\b", instruction)
    capitalized = re.findall(r"\b[A-Z][A-Za-z0-9&.'-]{2,}\b", instruction)
    return {
        "task_type": task_type,
        "has_image": bool(image_info.get("has_image") or image_info.get("image_url")),
        "has_image_url": bool(image_info.get("image_url")),
        "years": years[:8],
        "entities": capitalized[:12],
        "instruction_terms": sorted(_tokens(instruction))[:30],
    }


def build_failure_event_from_runner(
    task_id: str,
    step_id: int,
    instruction: str,
    task_type: str,
    failure_type: str,
    evidence: str,
    trajectory_rows: Iterable[dict[str, Any]],
    tool_name: str | None = None,
    tool_args: Optional[dict[str, Any]] = None,
    image_info: Optional[dict[str, Any]] = None,
) -> FailureEvent:
    """Convenience adapter for existing harness code."""
    return FailureEvent(
        task_id=task_id,
        step_id=step_id,
        trigger=failure_type,
        failure_type=failure_type,
        evidence=evidence,
        tool_name=tool_name,
        tool_args=tool_args or {},
        instruction=instruction,
        task_type=task_type,
        recent_messages=summarize_recent(list(trajectory_rows), limit=8),
        image_info=image_info or {},
    )


def _event_query_text(event: FailureEvent) -> str:
    return " ".join(
        [
            event.task_type,
            event.failure_type,
            event.trigger,
            event.tool_name or "",
            json.dumps(event.tool_args or {}, ensure_ascii=False),
            event.instruction,
            event.evidence,
        ]
    )


def _row_query_text(row: dict[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("task_type") or ""),
            str(row.get("failure_type") or ""),
            json.dumps(row.get("task_features") or {}, ensure_ascii=False),
            json.dumps(row.get("tool_history") or [], ensure_ascii=False)[:2000],
            " ".join(str(x) for x in row.get("reasoning_chain") or []),
            str(row.get("final_failure") or ""),
        ]
    )


def _tokens(text: str) -> set[str]:
    return {
        t.lower()
        for t in re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]{2,}", str(text))
        if t.lower() not in {"the", "and", "for", "with", "that", "this", "工具", "任务"}
    }


def _clip(text: str, max_chars: int) -> str:
    text = str(text or "")
    return text if len(text) <= max_chars else text[:max_chars] + f"\n...[truncated at {max_chars} chars]"

