"""Reflection and loop-control helpers for the harness agent."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


_PUNCT_RE = re.compile(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()\[\]{}<>《》【】]+")


def normalize_answer(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    return _PUNCT_RE.sub("", text.lower()).strip()


def judge_with_gold(prediction: str, gold_answer: str) -> bool:
    pred = normalize_answer(prediction)
    gold = normalize_answer(gold_answer)
    return bool(gold and (pred == gold or gold in pred))


def _safe_json(args: Any) -> str:
    try:
        return json.dumps(args, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(args)


def _tool_signature(fn_name: str, fn_args: dict[str, Any]) -> str:
    normalized = dict(fn_args or {})
    if fn_name == "search_text":
        normalized = {
            "query": str(normalized.get("query", "")).strip().lower(),
            "fetch": bool(normalized.get("fetch", True)),
        }
    elif fn_name == "search_image":
        normalized = {
            "image": str(normalized.get("image") or normalized.get("image_url") or "").strip(),
            "fetch": bool(normalized.get("fetch", True)),
        }
    elif fn_name in {"browser_navigate", "browser_parallel"}:
        if "url" in normalized:
            normalized = {"url": str(normalized.get("url", "")).strip().lower()}
        elif "urls" in normalized:
            urls = [str(u).strip().lower() for u in normalized.get("urls", [])]
            normalized = {"urls": urls, "mode": normalized.get("mode", "navigate")}
    return f"{fn_name}:{_safe_json(normalized)}"


@dataclass
class RunSignals:
    max_steps_reached: bool = False
    empty_assistant_turns: int = 0
    repeated_tool_calls: int = 0
    tool_errors: int = 0
    reflection_hints: int = 0
    final_answer_empty: bool = False
    reasons: list[str] = field(default_factory=list)

    def has_failure_signal(self) -> bool:
        return any(
            [
                self.max_steps_reached,
                self.final_answer_empty,
                self.empty_assistant_turns >= 2,
                self.repeated_tool_calls >= 2,
                self.tool_errors >= 2,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_text(self) -> str:
        parts = []
        if self.max_steps_reached:
            parts.append("达到最大轮数")
        if self.final_answer_empty:
            parts.append("最终答案为空")
        if self.empty_assistant_turns:
            parts.append(f"空回复 {self.empty_assistant_turns} 次")
        if self.repeated_tool_calls:
            parts.append(f"重复工具调用 {self.repeated_tool_calls} 次")
        if self.tool_errors:
            parts.append(f"工具错误 {self.tool_errors} 次")
        parts.extend(self.reasons[:3])
        return "；".join(parts) if parts else "未检测到明显失败信号"


class ReflectionMonitor:
    """Tracks repeated actions and emits compact self-reflection hints."""

    def __init__(
        self,
        *,
        repeat_threshold: int = 2,
        error_threshold: int = 2,
        max_hints: int = 3,
        task_family: str = "",
    ) -> None:
        self.repeat_threshold = max(1, int(repeat_threshold))
        self.error_threshold = max(1, int(error_threshold))
        self.max_hints = max(0, int(max_hints))
        self.task_family = task_family
        self.empty_assistant_turns = 0
        self.tool_errors = 0
        self.repeated_tool_calls = 0
        self.reflection_hints = 0
        self._call_counts: dict[str, int] = {}
        self._error_streak = 0
        self._pending_reasons: list[str] = []
        self._all_reasons: list[str] = []
        self._tool_history: list[str] = []
        self._evidence_gathered: list[str] = []

    def record_assistant(self, content: str, has_tool_calls: bool) -> None:
        if not content and not has_tool_calls:
            self.empty_assistant_turns += 1
            self._queue_reason("模型产生空回复，需要直接给出答案或明确下一步工具调用")

    def record_tool(self, fn_name: str, fn_args: dict[str, Any], tool_result: str) -> None:
        signature = _tool_signature(fn_name, fn_args)
        count = self._call_counts.get(signature, 0) + 1
        self._call_counts[signature] = count
        if count > 1:
            self.repeated_tool_calls += 1
        if count >= self.repeat_threshold:
            self._queue_reason(
                f"重复调用 {fn_name} 且参数基本相同：{_safe_json(fn_args)[:180]}"
            )

        # Track tool history for progress awareness
        self._tool_history.append(f"{fn_name}({_safe_json(fn_args)[:60]})")

        # Extract evidence snippets from successful results
        result_text = str(tool_result or "")
        lowered = result_text.lower()
        is_error = (
            "[error]" in lowered
            or '"ok": false' in lowered
            or "[proxy-error]" in lowered
            or "timeout" in lowered[:500]
            or "traceback" in lowered[:500]
        )
        if is_error:
            self.tool_errors += 1
            self._error_streak += 1
            if self._error_streak >= self.error_threshold:
                self._queue_reason(f"{fn_name} 连续出现工具错误或低质量返回")
        else:
            self._error_streak = 0
            # Record brief evidence from successful tool calls
            snippet = result_text[:120].replace("\n", " ").strip()
            if snippet and len(self._evidence_gathered) < 5:
                self._evidence_gathered.append(f"{fn_name}: {snippet}")

    def _queue_reason(self, reason: str) -> None:
        if reason not in self._pending_reasons:
            self._pending_reasons.append(reason)
        if reason not in self._all_reasons:
            self._all_reasons.append(reason)

    def consume_hint(self, step: int = 0, max_steps: int = 0) -> str:
        if not self._pending_reasons or self.reflection_hints >= self.max_hints:
            self._pending_reasons.clear()
            return ""
        reason = self._pending_reasons.pop(0)
        self.reflection_hints += 1

        progress_info = ""
        urgency = ""
        if step and max_steps:
            remaining = max_steps - step
            progress_info = f"（当前第{step}步/共{max_steps}步，剩余{remaining}步）"
            if remaining <= 3:
                urgency = "⚠️ 步数即将耗尽！"
            elif remaining <= max_steps // 2:
                urgency = "注意：已过半程。"

        evidence_summary = ""
        if self._evidence_gathered:
            evidence_summary = "\n已获取的关键信息：" + "；".join(self._evidence_gathered[-3:])

        strategy_hint = ""
        if self.task_family == "simplevqa":
            if self.repeated_tool_calls >= 2:
                strategy_hint = "\n策略：停止重复搜索，基于已有信息直接给出最可能的答案。"
            else:
                strategy_hint = "\n策略：确认图中主体后，用search_text查询具体属性即可作答。"
        elif self.task_family == "2wiki":
            if self.repeated_tool_calls >= 2:
                strategy_hint = "\n策略：已有足够线索，请直接比较/组合已知信息给出答案。"
            else:
                strategy_hint = "\n策略：分别查询两个实体的关键属性，然后比较或组合得出答案。"
        else:
            if self.repeated_tool_calls >= 2:
                strategy_hint = "\n策略：停止重复操作，基于已有证据直接作答。"

        actions = (
            "请立即选择：\n"
            "1) 如果证据足够，立即用<answer>答案</answer>输出最终短答案；\n"
            "2) 换一个更精确的关键词重新搜索；\n"
            "3) 换工具（如从search切换到browser）获取补充信息。\n"
            "禁止再次用相同参数调用同一个工具。"
        )

        return (
            f"{urgency}HARNESS_REFLECTION{progress_info}：检测到推理效率问题。\n"
            f"问题：{reason}\n"
            f"{evidence_summary}"
            f"{strategy_hint}\n"
            f"{actions}"
        )

    def signals(self, *, max_steps_reached: bool, final_answer: str) -> RunSignals:
        return RunSignals(
            max_steps_reached=max_steps_reached,
            empty_assistant_turns=self.empty_assistant_turns,
            repeated_tool_calls=self.repeated_tool_calls,
            tool_errors=self.tool_errors,
            reflection_hints=self.reflection_hints,
            final_answer_empty=not bool((final_answer or "").strip()),
            reasons=list(self._all_reasons),
        )


def _tags_from_metadata(metadata: dict[str, Any] | None) -> list[str]:
    if not metadata:
        return []
    tags: list[str] = []
    for key in (
        "language",
        "question_type",
        "task_category",
        "subject_category",
        "entity_class",
        "source",
    ):
        value = metadata.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            tags.append(str(value).strip().lower())
    return tags


def build_query_tags(task_family: str, metadata: dict[str, Any] | None) -> list[str]:
    return sorted({task_family, *_tags_from_metadata(metadata)})


def heuristic_lessons(
    *,
    task_family: str,
    instruction: str,
    metadata: dict[str, Any] | None,
    final_answer: str,
    signals: RunSignals,
    correct: bool | None,
) -> list[dict[str, Any]]:
    """Create abstract, non-answer-leaking lessons from a run."""

    task_family = task_family or "generic"
    metadata = metadata or {}
    tags = build_query_tags(task_family, metadata)
    lessons: list[dict[str, Any]] = []
    failure = signals.has_failure_signal() or correct is False
    success = correct is True

    if task_family == "simplevqa":
        category = "visual_grounding"
        if failure:
            lessons.append(
                {
                    "task_family": task_family,
                    "category": category,
                    "outcome": "failure",
                    "lesson": "视觉问答失败时，常见原因是没有先确认图中主体，直接搜索了问题中的抽象属性。",
                    "strategy": "先用图像本身识别主体；若主体不确定，优先 search_image；确认主体后再 search_text 查询年份、国家、作品、成因等属性。",
                    "avoid": "不要在未确认图中实体时反复搜索原问题全文；最终答案只保留题目要求的短答案。",
                    "tags": tags,
                    "confidence": 0.74,
                }
            )
        elif success:
            lessons.append(
                {
                    "task_family": task_family,
                    "category": category,
                    "outcome": "success",
                    "lesson": "SimpleVQA 高效路径是先识别图中实体，再只查询问题要求的单一属性。",
                    "strategy": "图搜用于定位实体，文搜用于查属性；答案用最短规范形式。",
                    "avoid": "不要输出解释性长句。",
                    "tags": tags,
                    "confidence": 0.62,
                }
            )

    elif task_family == "2wiki":
        question_type = str(metadata.get("question_type", "") or "").lower()
        if "comparison" in question_type or any(k in instruction.lower() for k in ("earlier", "later", "larger", "smaller", "older", "younger")):
            lessons.append(
                {
                    "task_family": task_family,
                    "category": "multi_hop_comparison",
                    "outcome": "failure" if failure else "success",
                    "lesson": "2Wiki 比较题必须分别抽取两个实体的同一属性，再比较并回答实体名或题目要求的对象。",
                    "strategy": "把问题拆成 A 的属性值、B 的属性值、比较规则三步；搜索时使用实体名加属性词，不要只搜索整句问题。",
                    "avoid": "不要把日期、地点等中间属性误当作最终答案；不要重复同一个搜索 query。",
                    "tags": tags + ["comparison"],
                    "confidence": 0.78 if failure else 0.66,
                }
            )
        else:
            lessons.append(
                {
                    "task_family": task_family,
                    "category": "multi_hop_bridge",
                    "outcome": "failure" if failure else "success",
                    "lesson": "2Wiki 组合题通常需要先找桥接实体，再查桥接实体的目标属性。",
                    "strategy": "先用第一个关系定位中间实体，再用中间实体加目标属性搜索；回答最后一跳的值。",
                    "avoid": "不要跳过中间实体直接猜测最终属性。",
                    "tags": tags + ["compositional"],
                    "confidence": 0.75 if failure else 0.64,
                }
            )

    if signals.repeated_tool_calls >= 1:
        lessons.append(
            {
                "task_family": task_family,
                "category": "loop_control",
                "outcome": "failure",
                "lesson": "重复调用同一工具和同一参数通常不会产生新信息，会浪费轮数和 token。",
                "strategy": "同一 query 或 URL 最多尝试一次；第二次需要改关键词、降低抓取范围、换 search/browser，或基于已有证据作答。",
                "avoid": "不要无变化地重复 search_text、search_image 或 browser_navigate。",
                "tags": tags + ["loop"],
                "confidence": 0.82,
            }
        )

    if signals.tool_errors >= 1:
        lessons.append(
            {
                "task_family": task_family,
                "category": "tool_recovery",
                "outcome": "failure",
                "lesson": "工具返回错误后应改变操作，而不是原样重试。",
                "strategy": "搜索失败时缩短 query、关闭 fetch 或换关键词；浏览器失败时先使用搜索摘要或 browser_parallel 打开多个候选页。",
                "avoid": "不要连续两次以上用相同参数重试失败工具。",
                "tags": tags + ["tool_error"],
                "confidence": 0.8,
            }
        )

    if signals.max_steps_reached or not final_answer.strip():
        lessons.append(
            {
                "task_family": task_family,
                "category": "stopping",
                "outcome": "failure",
                "lesson": "达到最大轮数通常说明没有及时收敛到可回答证据。",
                "strategy": "每两次工具调用后检查是否已有足够证据；足够时立即给最终短答案，不足时只补一个最关键缺口。",
                "avoid": "不要为了寻找完美证据而无限扩展搜索范围。",
                "tags": tags + ["stopping"],
                "confidence": 0.78,
            }
        )

    return lessons


def compact_tool_result(text: str, max_chars: int) -> tuple[str, bool]:
    text = str(text or "")
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip() + f"\n...[tool result truncated at {max_chars} chars]", True
    return text, False


def summarize_entries(entries: Iterable[dict[str, Any]], max_chars: int = 6000) -> str:
    parts: list[str] = []
    total = 0
    for entry in entries:
        role = entry.get("role", "")
        content = entry.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)[:1200]
        content = content.strip()
        if not content:
            continue
        line = f"{role}: {content[:800]}"
        if total + len(line) > max_chars:
            break
        parts.append(line)
        total += len(line)
    return "\n".join(parts)

