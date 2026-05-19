"""
Failure-triggered reflection for the agent harness.

The module is intentionally dependency-light. It can call the same
OpenAI-compatible LLM endpoint for richer diagnosis, but always has a
rule-based fallback so reflection remains available during tool/network
failures.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger("harness.reflection")


FAILURE_PATTERNS = (
    "[ERROR]",
    "[HARNESS ERROR]",
    "ok=false",
    '"ok": false',
    "'ok': False",
    "proxy-error",
    "jina-error",
    "Traceback",
    "Exception",
    "Timeout",
    "timed out",
    "Max steps reached",
)

EMPTY_RESULT_PATTERNS = (
    '"results": []',
    "'results': []",
    "[]",
    "no results",
    "not found",
)

FAILURE_TYPES = {
    "malformed_tool_call",
    "no_action",
    "tool_error",
    "tool_timeout",
    "repeated_query",
    "budget_exhausted",
    "generation_truncated",
    "weak_visual_grounding",
    "insufficient_evidence",
    "browser_navigation_error",
    "answer_format_error",
    "image_url_missing",
    "weak_visual_grounding",
    "tool_schema_or_args",
    "empty_evidence",
    "unknown",
}

NEXT_ACTION_TYPES = {
    "revise_query",
    "call_search_text",
    "call_search_image",
    "call_browser",
    "answer_with_evidence",
    "stop_and_format",
    "reduce_scope",
}

TASK_TYPES = {"2wiki_text", "simplevqa_multimodal", "benchmark_text", "benchmark_multimodal", "general"}

REFLECTION_CRITIC_PROMPT = """You are Qwen3-30B-A3B acting only as an open-source reflection critic / failure-analysis agent.

Your job is to analyze an observed agent failure from trajectory evidence and propose a correction strategy.

Hard constraints:
- Return JSON only. No markdown, no prose outside JSON.
- Do not answer the original task.
- Do not use or infer the gold answer.
- Do not modify the prediction.
- Ground every root_cause in observed trajectory evidence.
- Use only trajectory text, tool outputs/errors, and image metadata. Do not inspect the image directly.
- If the failure is visual grounding related, say what information is missing and which image/search/browser action should be tried next.
- If the failure is text or benchmark reasoning related, suggest entity decomposition, query rewrite, or evidence chaining.
- If evidence is insufficient, set confidence below 0.5.

Return this strict JSON schema:
{
  "failure_type": "malformed_tool_call | no_action | tool_error | tool_timeout | repeated_query | budget_exhausted | generation_truncated | weak_visual_grounding | insufficient_evidence | browser_navigation_error | answer_format_error | unknown",
  "root_cause": "...",
  "evidence": ["..."],
  "correction_strategy": "...",
  "next_prompt": "...",
  "next_action_type": "revise_query | call_search_text | call_search_image | call_browser | answer_with_evidence | stop_and_format | reduce_scope",
  "should_retry_same_action": false,
  "memory_lesson": "...",
  "applicable_task_types": ["2wiki_text", "simplevqa_multimodal", "benchmark_text", "benchmark_multimodal"],
  "confidence": 0.0
}
"""


@dataclass
class ReflectionConfig:
    """Runtime switches for reflection."""

    enabled: bool = os.getenv("ENABLE_REFLECTION", "1") != "0"
    use_llm: bool = os.getenv("REFLECTION_USE_LLM", "0") == "1"
    use_light_llm: bool = os.getenv("REFLECTION_USE_LIGHT_LLM", "0") == "1"
    memory_path: str = os.getenv("REFLECTION_MEMORY_PATH", "reflection_memory/reflection_memory.jsonl")
    max_memory_items: int = int(os.getenv("REFLECTION_MAX_MEMORY", "4"))
    min_relevance: float = 0.12
    memory_reuse_threshold: float = float(os.getenv("REFLECTION_MEMORY_REUSE_THRESHOLD", "0.58"))
    text_memory_reuse_threshold: float = float(os.getenv("REFLECTION_TEXT_MEMORY_REUSE_THRESHOLD", "0.78"))
    critic_min_step: int = int(os.getenv("REFLECTION_CRITIC_MIN_STEP", "3"))
    critic_max_calls_per_task: int = int(os.getenv("REFLECTION_CRITIC_MAX_CALLS_PER_TASK", "2"))
    async_critic: bool = os.getenv("REFLECTION_ASYNC_CRITIC", "0") == "1"
    force_critic_failure_types: str = os.getenv(
        "REFLECTION_FORCE_CRITIC_ON_FAILURE_TYPES",
        "budget_exhausted,answer_format_error,insufficient_evidence",
    )
    critic_async_log_path: str = os.getenv(
        "REFLECTION_CRITIC_ASYNC_LOG_PATH",
        "reflection_memory/critic_async_log.jsonl",
    )
    reuse_critic_memory: bool = os.getenv("REFLECTION_REUSE_CRITIC_MEMORY", "1") != "0"
    write_critic_memory: bool = os.getenv("REFLECTION_WRITE_CRITIC_MEMORY", "1") != "0"
    isolated_critic_memory_models: str = os.getenv(
        "REFLECTION_ISOLATED_CRITIC_MEMORY_MODELS",
        "Qwen3-30B-A3B",
    )
    llm_model: Optional[str] = os.getenv("REFLECTION_MODEL", "Qwen3-30B-A3B")
    base_url: Optional[str] = os.getenv("REFLECTION_BASE_URL", "http://127.0.0.1:8001/v1") or None
    api_key: str = os.getenv("REFLECTION_API_KEY", "")
    light_model: str = os.getenv("REFLECTION_LIGHT_MODEL", os.getenv("MODEL_NAME", "Qwen3.5-9B"))
    light_base_url: str = os.getenv("REFLECTION_LIGHT_BASE_URL", os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    timeout: float = float(os.getenv("REFLECTION_TIMEOUT", "30"))
    light_timeout: float = float(os.getenv("REFLECTION_LIGHT_TIMEOUT", "8"))
    max_prompt_chars: int = int(os.getenv("REFLECTION_MAX_CONTEXT_CHARS", "6000"))
    allow_vision_context: bool = os.getenv("REFLECTION_ALLOW_VISION_CONTEXT", "1") != "0"


@dataclass
class FailureEvent:
    """Normalized representation of a failure observed by the harness."""

    task_id: str
    step_id: int
    trigger: str
    failure_type: str
    evidence: str
    tool_name: Optional[str] = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    instruction: str = ""
    task_type: str = "general"
    recent_messages: list[dict[str, Any]] = field(default_factory=list)
    tools_summary: list[dict[str, str]] = field(default_factory=list)
    image_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionRecord:
    """Structured reflection result stored and re-injected later."""

    task_id: str
    step_id: int
    trigger: str
    failure_type: str
    root_cause: str
    correction_strategy: str
    reusable_lesson: str
    next_prompt: str
    evidence: list[str] = field(default_factory=list)
    next_action_type: str = "revise_query"
    should_retry_same_action: bool = False
    memory_lesson: str = ""
    applicable_task_types: list[str] = field(default_factory=list)
    task_context: str = ""
    task_type: str = "general"
    critic_model: str = "rule_fallback"
    memory_id: str = ""
    source_task_id: str = ""
    use_count: int = 0
    memory_written: bool = False
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)


class ReflectionMemory:
    """Append-only JSONL memory with simple lexical retrieval."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ReflectionRecord) -> bool:
        if not self._is_worth_storing(record):
            return False
        if self._has_duplicate(record):
            return False

        row = self._record_to_memory_row(record)
        row["source_task_id"] = record.source_task_id or record.task_id
        row["use_count"] = int(record.use_count or 0)
        self._append_row_locked(row)
        return True

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    logger.warning("skip broken reflection memory line: %s", s[:160])
                    continue
                if isinstance(obj, dict):
                    rows.append(self._normalize_row(obj))
        return rows

    def retrieve(
        self,
        query: str,
        limit: int,
        min_score: float,
        task_type: str = "general",
        update_use_count: bool = True,
        exclude_critic_models: Optional[set[str]] = None,
    ) -> list[dict[str, Any]]:
        query_terms = _tokens(query)
        if not query_terms:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self.load():
            if exclude_critic_models and str(row.get("critic_model") or "") in exclude_critic_models:
                continue
            row_task_type = str(row.get("task_type") or "general")
            applicable = row.get("applicable_task_types") or []
            task_bonus = 0.0
            if task_type != "general" and (row_task_type == task_type or task_type in applicable):
                task_bonus = 0.35
            text = " ".join(
                str(row.get(k, ""))
                for k in (
                    "failure_type",
                    "root_cause",
                    "correction_strategy",
                    "memory_lesson",
                    "reusable_lesson",
                    "next_prompt",
                    "task_context",
                    "trigger",
                )
            )
            row_terms = _tokens(text)
            if not row_terms:
                continue
            overlap = len(query_terms & row_terms)
            score = overlap / max(len(query_terms), 1) + task_bonus
            if score >= min_score:
                row = dict(row)
                row["_score"] = round(score, 4)
                scored.append((score, row))

        scored.sort(key=lambda item: (item[0], item[1].get("created_at", 0)), reverse=True)
        hits = [row for _, row in scored[:limit]]
        if update_use_count and hits:
            self.increment_use_count(hits)
        return hits

    def increment_use_count(self, hits: list[dict[str, Any]]) -> None:
        if not self.path.exists():
            return
        hit_keys = {_memory_key(row) for row in hits}
        updated: list[dict[str, Any]] = []
        changed = False
        for row in self.load():
            if _memory_key(row) in hit_keys:
                row["use_count"] = int(row.get("use_count", 0) or 0) + 1
                changed = True
            updated.append(row)
        if not changed:
            return
        self._rewrite_rows_locked(updated)

    def _append_row_locked(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            _lock_file(f)
            try:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            finally:
                _unlock_file(f)

    def _rewrite_rows_locked(self, rows: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a+", encoding="utf-8") as f:
            _lock_file(f)
            try:
                f.seek(0)
                f.truncate()
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            finally:
                _unlock_file(f)

    def _is_worth_storing(self, record: ReflectionRecord) -> bool:
        lesson = record.memory_lesson or record.reusable_lesson
        return bool(
            str(record.root_cause or "").strip()
            and str(record.correction_strategy or "").strip()
            and str(record.next_prompt or "").strip()
            and str(lesson or "").strip()
            and float(record.confidence or 0.0) >= 0.4
        )

    def _has_duplicate(self, record: ReflectionRecord) -> bool:
        lesson_terms = _tokens(record.memory_lesson or record.reusable_lesson)
        if not lesson_terms:
            return False
        for row in self.load():
            if row.get("failure_type") != record.failure_type:
                continue
            score = _jaccard(lesson_terms, _tokens(str(row.get("memory_lesson") or row.get("reusable_lesson") or "")))
            if score >= 0.72:
                return True
        return False

    def _record_to_memory_row(self, record: ReflectionRecord) -> dict[str, Any]:
        return {
            "memory_id": record.memory_id or _make_memory_id(record),
            "source_task_id": record.source_task_id or record.task_id,
            "task_type": record.task_type or "general",
            "failure_type": record.failure_type,
            "root_cause": record.root_cause,
            "correction_strategy": record.correction_strategy,
            "memory_lesson": record.memory_lesson or record.reusable_lesson,
            "applicable_task_types": record.applicable_task_types or [record.task_type or "general"],
            "critic_model": record.critic_model or "rule_fallback",
            "confidence": float(record.confidence or 0.0),
            "created_at": record.created_at,
            "use_count": int(record.use_count or 0),
            "next_prompt": record.next_prompt,
            "next_action_type": record.next_action_type,
            "task_context": record.task_context,
            "reusable_lesson": record.reusable_lesson,
        }

    def _normalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        if not out.get("memory_id"):
            out["memory_id"] = _make_memory_id_from_row(out)
        if not out.get("source_task_id"):
            out["source_task_id"] = str(out.get("task_id", ""))
        if not out.get("task_type"):
            out["task_type"] = "general"
        if not out.get("memory_lesson"):
            out["memory_lesson"] = str(out.get("reusable_lesson", ""))
        if not out.get("applicable_task_types"):
            out["applicable_task_types"] = [out.get("task_type", "general")]
        if not out.get("critic_model"):
            out["critic_model"] = "legacy"
        if "use_count" not in out:
            out["use_count"] = 0
        if "created_at" not in out:
            out["created_at"] = 0
        return out


class ReflectionManager:
    """Detects failures, creates reflections, and formats them for the agent."""

    def __init__(self, config: ReflectionConfig, client: Any = None, model_name: str = ""):
        self.config = config
        self.client = client
        self.model_name = config.llm_model or "Qwen3-30B-A3B"
        self.memory = ReflectionMemory(config.memory_path)
        self.last_memory_hits: list[dict[str, Any]] = []
        self.critic_calls_by_task: dict[str, int] = {}
        self._async_pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="reflection_critic")
            if config.async_critic
            else None
        )

    def build_system_appendix(self, instruction: str, task_type: str = "general") -> str:
        self.last_memory_hits = []
        if not self.config.enabled:
            return ""
        lessons = self.memory.retrieve(
            instruction,
            limit=self.config.max_memory_items,
            min_score=self.config.min_relevance,
            task_type=task_type,
            exclude_critic_models=self._excluded_memory_critic_models(),
        )
        self.last_memory_hits = lessons
        if not lessons:
            return ""

        lines = [
            "",
            "## Reflection Memory",
            "以下是此前任务失败后总结出的可复用经验。遇到相似情形时先运用这些经验，再决定是否重试。",
        ]
        for i, row in enumerate(lessons, start=1):
            lesson = _clip(str(row.get("memory_lesson") or row.get("reusable_lesson", "")), 240)
            strategy = _clip(str(row.get("correction_strategy", "")), 240)
            lines.append(f"{i}. 经验：{lesson} 修正策略：{strategy}")
        return "\n".join(lines)

    def detect_assistant_failure(
        self,
        task_id: str,
        step_id: int,
        instruction: str,
        content: str,
        reasoning_content: str,
        tool_calls: Any,
        finish_reason: Optional[str],
        recent_messages: list[dict[str, Any]],
        task_type: str = "general",
        tools_summary: Optional[list[dict[str, str]]] = None,
        image_info: Optional[dict[str, Any]] = None,
    ) -> Optional[FailureEvent]:
        if not self.config.enabled:
            return None

        if not tool_calls and not content.strip():
            if "<tool_call>" in reasoning_content or "<function=" in reasoning_content:
                return FailureEvent(
                    task_id=task_id,
                    step_id=step_id,
                    trigger="assistant_empty_with_textual_tool_call",
                    failure_type="malformed_tool_call",
                    evidence=_clip(reasoning_content, 1200),
                    instruction=instruction,
                    task_type=task_type,
                    recent_messages=recent_messages,
                    tools_summary=tools_summary or [],
                    image_info=image_info or {},
                )
            return FailureEvent(
                task_id=task_id,
                step_id=step_id,
                trigger="assistant_empty_response",
                failure_type="no_action",
                evidence=f"finish_reason={finish_reason}, no content and no executable tool_calls",
                instruction=instruction,
                task_type=task_type,
                recent_messages=recent_messages,
                tools_summary=tools_summary or [],
                image_info=image_info or {},
            )

        if finish_reason == "length":
            return FailureEvent(
                task_id=task_id,
                step_id=step_id,
                trigger="assistant_context_or_generation_limit",
                failure_type="generation_truncated",
                evidence=_clip(content or reasoning_content, 1200),
                instruction=instruction,
                task_type=task_type,
                recent_messages=recent_messages,
                tools_summary=tools_summary or [],
                image_info=image_info or {},
            )
        return None

    def detect_tool_failure(
        self,
        task_id: str,
        step_id: int,
        instruction: str,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: str,
        recent_messages: list[dict[str, Any]],
        task_type: str = "general",
        tools_summary: Optional[list[dict[str, str]]] = None,
        image_info: Optional[dict[str, Any]] = None,
    ) -> Optional[FailureEvent]:
        if not self.config.enabled:
            return None
        normalized = str(tool_result)
        lowered = normalized.lower()
        if tool_name == "search_image" and not tool_args.get("image_url"):
            return FailureEvent(
                task_id=task_id,
                step_id=step_id,
                trigger="image_url_missing",
                failure_type="image_url_missing",
                evidence="search_image was called without an online image_url.",
                tool_name=tool_name,
                tool_args=tool_args,
                instruction=instruction,
                task_type=task_type,
                recent_messages=recent_messages,
                tools_summary=tools_summary or [],
                image_info=image_info or {},
            )
        if any(p.lower() in lowered for p in FAILURE_PATTERNS):
            return FailureEvent(
                task_id=task_id,
                step_id=step_id,
                trigger="tool_failure",
                failure_type=_classify_tool_failure(normalized),
                evidence=_clip(normalized, 1200),
                tool_name=tool_name,
                tool_args=tool_args,
                instruction=instruction,
                task_type=task_type,
                recent_messages=recent_messages,
                tools_summary=tools_summary or [],
                image_info=image_info or {},
            )
        if normalized.strip() in EMPTY_RESULT_PATTERNS:
            return FailureEvent(
                task_id=task_id,
                step_id=step_id,
                trigger="tool_empty_result",
                failure_type="empty_evidence",
                evidence=_clip(normalized, 1200),
                tool_name=tool_name,
                tool_args=tool_args,
                instruction=instruction,
                task_type=task_type,
                recent_messages=recent_messages,
                tools_summary=tools_summary or [],
                image_info=image_info or {},
            )
        return None

    def repeated_query_failure(
        self,
        task_id: str,
        step_id: int,
        instruction: str,
        tool_name: str,
        query: str,
        count: int,
        recent_messages: list[dict[str, Any]],
        task_type: str = "general",
        tools_summary: Optional[list[dict[str, str]]] = None,
        image_info: Optional[dict[str, Any]] = None,
    ) -> FailureEvent:
        return FailureEvent(
            task_id=task_id,
            step_id=step_id,
            trigger="repeated_query",
            failure_type="repeated_query",
            evidence=f"{tool_name} query repeated {count} times: {query}",
            tool_name=tool_name,
            tool_args={"query": query, "repeat_count": count},
            instruction=instruction,
            task_type=task_type,
            recent_messages=recent_messages,
            tools_summary=tools_summary or [],
            image_info=image_info or {},
        )

    def max_steps_failure(
        self,
        task_id: str,
        step_id: int,
        instruction: str,
        recent_messages: list[dict[str, Any]],
        task_type: str = "general",
        tools_summary: Optional[list[dict[str, str]]] = None,
        image_info: Optional[dict[str, Any]] = None,
    ) -> FailureEvent:
        return FailureEvent(
            task_id=task_id,
            step_id=step_id,
            trigger="max_steps_reached",
            failure_type="budget_exhausted",
            evidence="The task loop reached max_steps before a final answer.",
            instruction=instruction,
            task_type=task_type,
            recent_messages=recent_messages,
            tools_summary=tools_summary or [],
            image_info=image_info or {},
        )

    def reflect(self, event: FailureEvent) -> ReflectionRecord:
        """Memory-first reflection.

        Execution order:
        1. Match failure against the JSONL experience library.
        2. Reuse a high-confidence strategy without calling Qwen3-30B-A3B.
        3. Optionally use a fast Qwen3.5-9B critic for common failures.
        4. Call Qwen3-30B-A3B only for hard/novel failures and within budget.
        5. Fall back to deterministic rules.
        """
        if not self._should_bypass_memory(event):
            memory_record = self._reflect_from_memory(event)
            if memory_record is not None:
                return memory_record

        if self.config.use_light_llm:
            try:
                record = self._reflect_with_light_llm(event)
                record.memory_written = self._append_memory_if_allowed(record)
                return record
            except Exception as exc:  # noqa: BLE001
                logger.debug("Light reflection unavailable, continue gate: %s", exc)

        if self._should_call_critic(event) and self.config.async_critic:
            quick_record = self._reflect_with_rules(event)
            quick_record.critic_model = "rule_fallback_async_critic_pending"
            quick_record.memory_written = self._append_memory_if_allowed(quick_record)
            self._submit_async_critic(event)
            return quick_record

        if self._should_call_critic(event):
            try:
                self.critic_calls_by_task[event.task_id] = self.critic_calls_by_task.get(event.task_id, 0) + 1
                record = self._reflect_with_llm(event)
                record.memory_written = self._append_memory_if_allowed(record)
                self._write_critic_log(event, record, async_call=False)
                return record
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM reflection failed, falling back to rules: %s", exc)

        record = self._reflect_with_rules(event)
        record.memory_written = self._append_memory_if_allowed(record)
        return record

    def _submit_async_critic(self, event: FailureEvent) -> None:
        if self._async_pool is None:
            return
        self.critic_calls_by_task[event.task_id] = self.critic_calls_by_task.get(event.task_id, 0) + 1

        def job() -> None:
            try:
                record = self._reflect_with_llm(event)
                record.memory_written = self._append_memory_if_allowed(record)
                self._write_critic_log(event, record, async_call=True)
                logger.info(
                    "async reflection critic finished: task=%s failure=%s written=%s",
                    event.task_id,
                    event.failure_type,
                    record.memory_written,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("async reflection critic failed: %s", exc)

        self._async_pool.submit(job)

    def _isolated_critic_models(self) -> set[str]:
        return {
            item.strip()
            for item in str(self.config.isolated_critic_memory_models or "").split(",")
            if item.strip()
        }

    def _excluded_memory_critic_models(self) -> set[str]:
        return self._isolated_critic_models() if not self.config.reuse_critic_memory else set()

    def _append_memory_if_allowed(self, record: ReflectionRecord) -> bool:
        if (
            not self.config.write_critic_memory
            and record.critic_model in self._isolated_critic_models()
        ):
            return False
        return self.memory.append(record)

    def _should_bypass_memory(self, event: FailureEvent) -> bool:
        forced = {
            item.strip()
            for item in str(self.config.force_critic_failure_types or "").split(",")
            if item.strip()
        }
        return event.failure_type in forced and self._should_call_critic(event)

    def _write_critic_log(self, event: FailureEvent, record: ReflectionRecord, async_call: bool) -> None:
        if record.critic_model != self.model_name:
            return
        row = {
            "created_at": time.time(),
            "task_id": event.task_id,
            "step_id": event.step_id,
            "failure_type": event.failure_type,
            "critic_model": record.critic_model,
            "async": async_call,
            "memory_written": record.memory_written,
            "confidence": record.confidence,
            "memory_id": record.memory_id,
        }
        path = Path(self.config.critic_async_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            _lock_file(f)
            try:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            finally:
                _unlock_file(f)

    def _reflect_from_memory(self, event: FailureEvent) -> Optional[ReflectionRecord]:
        query = _event_memory_query(event)
        hits = self.memory.retrieve(
            query,
            limit=max(1, self.config.max_memory_items),
            min_score=self.config.min_relevance,
            task_type=_normalize_task_type(event.task_type),
            exclude_critic_models=self._excluded_memory_critic_models(),
        )
        usable = []
        for row in hits:
            score = float(row.get("_score", 0.0) or 0.0)
            if self._memory_row_usable(row, event, score):
                usable.append(row)
        if not usable:
            return None

        row = usable[0]
        score = float(row.get("_score", 0.0) or 0.0)
        lesson = str(row.get("memory_lesson") or row.get("reusable_lesson") or "")
        strategy = str(row.get("correction_strategy") or "")
        next_prompt = str(row.get("next_prompt") or strategy or lesson)
        return ReflectionRecord(
            task_id=event.task_id,
            step_id=event.step_id,
            trigger=event.trigger,
            failure_type=event.failure_type,
            root_cause=_clip(str(row.get("root_cause") or f"复用经验库中与 {event.failure_type} 匹配的失败归因。"), 500),
            correction_strategy=_clip(strategy, 700),
            reusable_lesson=_clip(lesson, 500),
            memory_lesson=_clip(lesson, 500),
            next_prompt=_clip(next_prompt, 700),
            evidence=[_clip(event.evidence, 240)] if event.evidence else [],
            next_action_type=str(row.get("next_action_type") or "revise_query"),
            applicable_task_types=_normalize_task_types(row.get("applicable_task_types"), event.task_type),
            task_context=_clip(event.instruction, 500),
            task_type=_normalize_task_type(event.task_type),
            critic_model="memory_reuse",
            memory_id=str(row.get("memory_id") or ""),
            source_task_id=str(row.get("source_task_id") or ""),
            confidence=min(0.98, max(float(row.get("confidence", 0.0) or 0.0), score)),
            memory_written=False,
        )

    def _memory_row_usable(self, row: dict[str, Any], event: FailureEvent, score: float) -> bool:
        task_type = _normalize_task_type(event.task_type)
        row_task_type = _normalize_task_type(str(row.get("task_type") or "general"))
        applicable_types = _normalize_task_types(row.get("applicable_task_types"), row_task_type)
        same_failure = row.get("failure_type") == event.failure_type
        same_task_type = row_task_type == task_type or task_type in applicable_types
        is_text = task_type in {"2wiki_text", "benchmark_text"}

        threshold = self.config.memory_reuse_threshold
        if is_text:
            threshold = max(threshold, self.config.text_memory_reuse_threshold)
        if event.failure_type in {"budget_exhausted", "repeated_query", "empty_evidence"}:
            threshold = max(threshold, 0.72)
        if score < threshold:
            return False

        # Prevent generic rule memories such as "avoid blind retry" from
        # suppressing the 30B critic on hard multi-hop text cases.
        if is_text and not same_task_type:
            return False
        if is_text and event.failure_type == "budget_exhausted" and _is_generic_memory(row):
            return False
        if not (same_failure or same_task_type):
            return False
        return True

    def _should_call_critic(self, event: FailureEvent) -> bool:
        if not (self.config.use_llm and self.model_name):
            return False
        if event.step_id < self.config.critic_min_step and event.failure_type not in {
            "generation_truncated",
            "budget_exhausted",
            "weak_visual_grounding",
        }:
            return False
        calls = self.critic_calls_by_task.get(event.task_id, 0)
        if calls >= self.config.critic_max_calls_per_task:
            return False
        if event.failure_type in {"malformed_tool_call", "image_url_missing", "repeated_query"} and calls > 0:
            return False
        return True

    def to_feedback_message(self, record: ReflectionRecord) -> str:
        return (
            "[Reflection]\n"
            f"触发原因：{record.trigger}\n"
            f"失败类型：{record.failure_type}\n"
            f"根因定位：{record.root_cause}\n"
            f"修正策略：{record.correction_strategy}\n"
            f"当前下一步：{record.next_prompt}\n"
            f"反思模式：{record.critic_model}\n"
            "请先按上述修正策略调整计划，再继续执行；不要简单重复刚才失败的动作。"
        )

    def _reflect_with_llm(self, event: FailureEvent) -> ReflectionRecord:
        client = self._get_critic_client()
        prompt = self._build_critic_payload(event)
        messages = [
            {"role": "system", "content": REFLECTION_CRITIC_PROMPT},
            {"role": "user", "content": _clip(json.dumps(prompt, ensure_ascii=False), self.config.max_prompt_chars)},
        ]
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=800,
            timeout=self.config.timeout,
        )
        raw = response.choices[0].message.content or "{}"
        data = validate_critic_output(raw)
        return ReflectionRecord(
            task_id=event.task_id,
            step_id=event.step_id,
            trigger=event.trigger,
            failure_type=str(data.get("failure_type") or event.failure_type),
            root_cause=_clip(str(data["root_cause"]), 500),
            evidence=[_clip(str(x), 220) for x in data.get("evidence", [])],
            correction_strategy=_clip(str(data["correction_strategy"]), 700),
            reusable_lesson=_clip(str(data["memory_lesson"]), 500),
            memory_lesson=_clip(str(data["memory_lesson"]), 500),
            next_prompt=_clip(str(data["next_prompt"]), 700),
            next_action_type=str(data.get("next_action_type") or "revise_query"),
            should_retry_same_action=bool(data.get("should_retry_same_action", False)),
            applicable_task_types=_normalize_task_types(data.get("applicable_task_types"), event.task_type),
            task_context=_clip(event.instruction, 500),
            task_type=_normalize_task_type(event.task_type),
            critic_model=self.model_name,
            memory_id=_make_memory_id_from_parts(event.task_id, data.get("failure_type", event.failure_type), data.get("memory_lesson", "")),
            source_task_id=event.task_id,
            confidence=_safe_float(data.get("confidence"), 0.0),
        )

    def _reflect_with_light_llm(self, event: FailureEvent) -> ReflectionRecord:
        """Use the 9B base model as a cheap failure analyst for common cases.

        This path is optional and still follows the same strict JSON protocol.
        It is meant for low-cost triage; hard/novel cases are handled by the
        30B critic through _should_call_critic().
        """
        client = self._get_light_client()
        prompt = self._build_critic_payload(event)
        light_system = (
            REFLECTION_CRITIC_PROMPT
            + "\nYou are the lightweight Qwen3.5-9B triage critic. Prefer short, operational fixes."
        )
        messages = [
            {"role": "system", "content": light_system},
            {"role": "user", "content": _clip(json.dumps(prompt, ensure_ascii=False), min(self.config.max_prompt_chars, 2500))},
        ]
        response = client.chat.completions.create(
            model=self.config.light_model,
            messages=messages,
            temperature=0.1,
            max_tokens=500,
            timeout=self.config.light_timeout,
        )
        raw = response.choices[0].message.content or "{}"
        data = validate_critic_output(raw)
        return ReflectionRecord(
            task_id=event.task_id,
            step_id=event.step_id,
            trigger=event.trigger,
            failure_type=str(data.get("failure_type") or event.failure_type),
            root_cause=_clip(str(data["root_cause"]), 500),
            evidence=[_clip(str(x), 220) for x in data.get("evidence", [])],
            correction_strategy=_clip(str(data["correction_strategy"]), 700),
            reusable_lesson=_clip(str(data["memory_lesson"]), 500),
            memory_lesson=_clip(str(data["memory_lesson"]), 500),
            next_prompt=_clip(str(data["next_prompt"]), 700),
            next_action_type=str(data.get("next_action_type") or "revise_query"),
            should_retry_same_action=bool(data.get("should_retry_same_action", False)),
            applicable_task_types=_normalize_task_types(data.get("applicable_task_types"), event.task_type),
            task_context=_clip(event.instruction, 500),
            task_type=_normalize_task_type(event.task_type),
            critic_model=f"light_{self.config.light_model}",
            memory_id=_make_memory_id_from_parts(event.task_id, data.get("failure_type", event.failure_type), data.get("memory_lesson", "")),
            source_task_id=event.task_id,
            confidence=min(_safe_float(data.get("confidence"), 0.0), 0.78),
        )

    def _get_critic_client(self) -> Any:
        if self.client is not None and not self.config.base_url and not self.config.api_key:
            return self.client
        try:
            from openai import OpenAI  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"openai SDK unavailable for reflection critic: {exc}") from exc
        kwargs: dict[str, Any] = {"api_key": self.config.api_key or "EMPTY", "timeout": self.config.timeout}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return OpenAI(**kwargs)

    def _get_light_client(self) -> Any:
        try:
            from openai import OpenAI  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"openai SDK unavailable for light reflection: {exc}") from exc
        return OpenAI(
            base_url=self.config.light_base_url,
            api_key=os.getenv("LLM_API_KEY", "EMPTY"),
            timeout=self.config.light_timeout,
        )

    def _build_critic_payload(self, event: FailureEvent) -> dict[str, Any]:
        image_info = event.image_info if self.config.allow_vision_context else {}
        return {
            "task_id": event.task_id,
            "task_type": _normalize_task_type(event.task_type),
            "instruction": _strip_gold_like_text(event.instruction),
            "failure_event": {
                "failure_type": event.failure_type,
                "error": event.evidence,
                "step_id": event.step_id,
                "trigger": event.trigger,
                "tool_name": event.tool_name,
                "tool_args": event.tool_args,
            },
            "recent_trajectory_window": summarize_recent(event.recent_messages, limit=8),
            "available_tools_schema_summary": event.tools_summary,
            "image_info": image_info,
            "forbidden": [
                "Do not answer the original task.",
                "Do not use gold answer.",
                "Do not modify pred.",
                "Only propose a correction for the main Qwen agent.",
            ],
        }

    def _reflect_with_rules(self, event: FailureEvent) -> ReflectionRecord:
        task_type = _normalize_task_type(event.task_type)
        if event.failure_type == "malformed_tool_call":
            root_cause = "模型把工具调用写进 reasoning 文本，但没有产生可执行的 OpenAI tool_calls，主循环无法调度工具。"
            strategy = "下一步必须重新发起规范 function call；不要用 XML 或自然语言伪造工具调用。"
            lesson = "若出现空 assistant 内容且 reasoning 中有 <tool_call>，应提示模型改用真实工具调用格式。"
            next_prompt = "请把刚才计划中的查询改为一次或多次真实工具调用，或在已有证据足够时直接给出最终答案。"
            next_action_type = "call_search_image" if task_type == "simplevqa_multimodal" else "call_search_text"
        elif event.failure_type == "no_action":
            root_cause = "模型既没有回答，也没有发起工具调用，导致一步预算被空转消耗。"
            strategy = "要求模型在下一步二选一：调用一个具体工具，或基于已有证据输出最终答案。"
            lesson = "空动作需要立即打断，避免持续消耗 max_steps。"
            next_prompt = "请立刻选择一个可执行动作：调用工具获取缺失证据，或给出最终答案。"
            next_action_type = "call_search_image" if task_type == "simplevqa_multimodal" else "call_search_text"
        elif event.failure_type == "generation_truncated":
            root_cause = "模型输出被截断，可能因为推理过长或上下文过大。"
            strategy = "压缩中间推理，只保留关键实体、证据和下一步动作；必要时缩小搜索范围。"
            lesson = "长推理任务失败时应先压缩状态，再继续。"
            next_prompt = "请用不超过 5 行总结当前证据，然后继续最小必要动作。"
            next_action_type = "reduce_scope"
        elif event.failure_type == "budget_exhausted":
            root_cause = "任务在步数预算内没有收敛，常见原因是查询链过长、工具调用未执行或迟迟未形成最终答案。"
            strategy = "后续相似任务应优先并行查询关键实体，拿到足够证据后立即回答，避免继续扩展无关搜索。"
            lesson = "多跳问答应先拆实体、并行搜证、尽早比较，不要把最终答案推迟到预算耗尽。"
            next_prompt = "在相似任务中限制每个子问题一次查询，证据足够后立即输出简短答案。"
            next_action_type = "reduce_scope"
        elif event.failure_type == "tool_timeout":
            root_cause = f"{event.tool_name or '工具'} 调用超时或连接失败，重复同一参数大概率继续失败。"
            strategy = "降低抓取量或换用搜索摘要/浏览器工具；必要时改写查询并减少 top_k、max_chars。"
            lesson = "工具超时后应减少请求复杂度或换工具，而不是原样重试。"
            next_prompt = "请用更短查询或更小 fetch 参数重试一次；若仍失败，换用另一个工具。"
            next_action_type = "revise_query"
        elif event.failure_type == "tool_schema_or_args":
            root_cause = "工具参数不符合 JSON 或 schema，调度层无法正确执行。"
            strategy = "重新构造最小合法参数，只保留 schema 中声明的字段。"
            lesson = "工具参数失败时先修 schema，而不是改变任务目标。"
            next_prompt = "请重新调用同一工具，但只传入合法 JSON 参数。"
            next_action_type = "call_search_image" if task_type == "simplevqa_multimodal" else "call_search_text"
        elif event.failure_type == "empty_evidence":
            root_cause = "工具没有返回可用证据，可能是关键词过窄、实体歧义或来源不可访问。"
            strategy = "改写查询，加入实体类型、别名、年份或地点限定；必要时换搜索源。"
            lesson = "空结果后应扩展或消歧查询，而不是重复同一关键词。"
            next_prompt = "请改写查询，加入别名/年份/地点中的至少一个限定词后再搜。"
            next_action_type = "revise_query"
        elif event.failure_type == "image_url_missing":
            root_cause = "多模态任务缺少在线 image_url，但模型仍尝试调用 search_image，导致图搜无法执行。"
            strategy = "不要继续调用 search_image；先根据主模型图像理解提取人物/物体/场景关键词，再使用 search_text 或 browser 验证。"
            lesson = "没有在线 image_url 时，图像题应转为视觉描述到文本检索的流程。"
            next_prompt = "请停止调用 search_image，先用一句话描述图中关键实体/场景，再把这些关键词用于 search_text 或 browser 验证。"
            next_action_type = "call_search_text"
        elif event.failure_type == "repeated_query":
            root_cause = "模型重复使用相同查询，说明没有根据失败反馈改变信息获取策略。"
            strategy = "改写查询，拆分实体或关系，加入年份、地点、别名等限定；必要时换浏览器打开关键页面。"
            lesson = "重复查询应触发查询改写或工具切换，而不是继续消耗步骤预算。"
            next_prompt = "请停止重复同一查询，先列出缺失实体/关系，再用不同关键词或浏览器验证证据。"
            next_action_type = "revise_query"
        else:
            root_cause = f"{event.tool_name or '当前步骤'} 返回错误信号：{_clip(event.evidence, 240)}"
            strategy = "先解析错误含义，再选择缩小参数、改写查询或更换工具。"
            lesson = "任何失败信号都应转化为参数或策略变更，禁止盲目重试。"
            next_prompt = "请根据错误信息修改下一步动作，并说明改变了哪个参数或工具。"
            next_action_type = "revise_query"

        if task_type == "2wiki_text":
            next_prompt += " 对 2Wiki 文本多跳题，先找实体 A，再找实体 B，再验证二者关系或比较条件。"
            applicable = ["2wiki_text"]
        elif task_type == "benchmark_text":
            next_prompt += " 对 benchmark 文本题，先拆解题目约束，再用搜索/浏览器补足关键证据，最后只输出简洁答案。"
            applicable = ["benchmark_text"]
        elif task_type in {"simplevqa_multimodal", "benchmark_multimodal"}:
            if event.failure_type in {"no_action", "budget_exhausted", "empty_evidence"}:
                root_cause = root_cause + " 对多模态任务，还需要确认图像识别和外部证据是否对齐。"
            next_prompt += " 对多模态题，先识别图像关键实体/场景；若 image_url 缺失，不要调用 search_image，改用文字搜索或浏览器验证。"
            applicable = [task_type]
        else:
            applicable = ["general"]

        return ReflectionRecord(
            task_id=event.task_id,
            step_id=event.step_id,
            trigger=event.trigger,
            failure_type=event.failure_type,
            root_cause=root_cause,
            evidence=[_clip(event.evidence, 240)] if event.evidence else [],
            correction_strategy=strategy,
            reusable_lesson=lesson,
            memory_lesson=lesson,
            next_prompt=next_prompt,
            next_action_type=next_action_type,
            should_retry_same_action=False,
            applicable_task_types=applicable,
            task_context=_clip(event.instruction, 500),
            task_type=task_type,
            critic_model="rule_fallback",
            memory_id=_make_memory_id_from_parts(event.task_id, event.failure_type, lesson),
            source_task_id=event.task_id,
            confidence=0.72,
        )


def summarize_recent(messages: Iterable[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    """Keep only compact recent messages for reflection prompts."""
    rows = list(messages)[-limit:]
    out: list[dict[str, Any]] = []
    for msg in rows:
        compact = dict(msg)
        if "content" in compact:
            compact["content"] = _clip(str(compact.get("content", "")), 800)
        if "tool_calls" in compact:
            compact["tool_calls"] = compact["tool_calls"][:2]
        out.append(compact)
    return out


def validate_critic_output(raw: str) -> dict[str, Any]:
    data = _extract_json(raw)
    required = {
        "failure_type",
        "root_cause",
        "evidence",
        "correction_strategy",
        "next_prompt",
        "next_action_type",
        "should_retry_same_action",
        "memory_lesson",
        "applicable_task_types",
        "confidence",
    }
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"critic output missing required fields: {sorted(missing)}")

    failure_type = str(data.get("failure_type", ""))
    if failure_type not in FAILURE_TYPES:
        raise ValueError(f"invalid failure_type: {failure_type}")

    next_action_type = str(data.get("next_action_type", ""))
    if next_action_type not in NEXT_ACTION_TYPES:
        raise ValueError(f"invalid next_action_type: {next_action_type}")

    confidence = _safe_float(data.get("confidence"), -1.0)
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be in [0, 1]")
    data["confidence"] = confidence

    for key in ["root_cause", "correction_strategy", "next_prompt", "memory_lesson"]:
        if not str(data.get(key, "")).strip():
            raise ValueError(f"critic output field is empty: {key}")

    text_blob = json.dumps(data, ensure_ascii=False).lower()
    forbidden_answer_patterns = [
        r"\bthe answer is\b",
        r"\bfinal answer\b",
        r"最终答案",
        r'"answer"\s*:',
        r'"gold"\s*:',
        r"\bgold answer\b",
    ]
    if any(re.search(pattern, text_blob) for pattern in forbidden_answer_patterns):
        raise ValueError("critic output appears to answer the task or reference gold/answer fields")

    evidence = data.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("evidence must be a list")
    evidence = [str(item).strip() for item in evidence if str(item).strip()]
    if not evidence and confidence >= 0.5:
        raise ValueError("evidence must be non-empty unless confidence < 0.5")
    data["evidence"] = evidence
    data["applicable_task_types"] = _normalize_task_types(data.get("applicable_task_types"), "general")
    data["should_retry_same_action"] = bool(data.get("should_retry_same_action"))
    return data


def _classify_tool_failure(text: str) -> str:
    lowered = text.lower()
    if "timeout" in lowered or "timed out" in lowered:
        return "tool_timeout"
    if "browser" in lowered and ("navigation" in lowered or "navigate" in lowered):
        return "browser_navigation_error"
    if "json" in lowered or "schema" in lowered or "arguments" in lowered:
        return "tool_schema_or_args"
    if "not found" in lowered or "no results" in lowered:
        return "empty_evidence"
    return "tool_error"


def _event_memory_query(event: FailureEvent) -> str:
    tool_args = " ".join(f"{k}={v}" for k, v in (event.tool_args or {}).items())
    image_bits = " ".join(f"{k}={v}" for k, v in (event.image_info or {}).items() if k in {"has_image", "image_url", "benchmark_index"})
    return " ".join(
        [
            event.task_type,
            event.failure_type,
            event.trigger,
            event.tool_name or "",
            tool_args,
            image_bits,
            _clip(event.evidence, 600),
            _clip(event.instruction, 500),
        ]
    )


def _tokens(text: str) -> set[str]:
    return {
        t.lower()
        for t in re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]{2,}", text)
        if t.lower()
        not in {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "请",
            "一个",
            "工具",
            "任务",
        }
    }


def _is_generic_memory(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(row.get(k, ""))
        for k in ("memory_lesson", "reusable_lesson", "correction_strategy", "next_prompt")
    ).lower()
    task_terms = {
        "multi-hop",
        "multihop",
        "2wiki",
        "benchmark",
        "entity",
        "entities",
        "decomposition",
        "evidence chain",
        "query",
        "search",
        "浏览器",
        "搜索",
        "实体",
        "多跳",
        "证据链",
        "拆解",
    }
    generic_terms = {
        "avoid blind retry",
        "不要盲目重试",
        "禁止盲目重试",
        "change the next action",
        "修改下一步动作",
        "立即打断",
    }
    has_task_terms = any(term in text for term in task_terms)
    has_generic_terms = any(term in text for term in generic_terms)
    return has_generic_terms and not has_task_terms


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _memory_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("memory_id") or row.get("source_task_id") or row.get("task_id") or ""),
        str(row.get("failure_type") or ""),
        str(row.get("memory_lesson") or row.get("correction_strategy") or ""),
    )


def _normalize_task_type(task_type: str | None) -> str:
    value = str(task_type or "general")
    return value if value in TASK_TYPES else "general"


def _normalize_task_types(value: Any, fallback: str) -> list[str]:
    if isinstance(value, list):
        out = [_normalize_task_type(str(item)) for item in value]
    else:
        out = [_normalize_task_type(fallback)]
    out = [item for item in out if item in TASK_TYPES]
    return sorted(set(out)) or [_normalize_task_type(fallback)]


def _strip_gold_like_text(text: str) -> str:
    # Defensive cleanup in case a caller accidentally embeds answer metadata.
    lines = []
    for line in str(text).splitlines():
        if re.search(r"\b(gold|answer|label|target)\s*[:=]", line, flags=re.I):
            continue
        lines.append(line)
    return "\n".join(lines)


def _make_memory_id(record: ReflectionRecord) -> str:
    return _make_memory_id_from_parts(record.task_id, record.failure_type, record.memory_lesson or record.reusable_lesson)


def _make_memory_id_from_parts(task_id: str, failure_type: str, lesson: str) -> str:
    seed = f"{task_id}:{failure_type}:{lesson[:120]}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def _make_memory_id_from_row(row: dict[str, Any]) -> str:
    return _make_memory_id_from_parts(
        str(row.get("source_task_id") or row.get("task_id") or ""),
        str(row.get("failure_type") or ""),
        str(row.get("memory_lesson") or row.get("reusable_lesson") or row.get("correction_strategy") or ""),
    )


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated at {max_chars} chars]"


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, out))


def _lock_file(handle: Any) -> None:
    try:
        import fcntl  # noqa: PLC0415

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except Exception:  # noqa: BLE001
        return


def _unlock_file(handle: Any) -> None:
    try:
        import fcntl  # noqa: PLC0415

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception:  # noqa: BLE001
        return
