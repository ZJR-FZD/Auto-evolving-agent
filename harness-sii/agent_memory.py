"""Lightweight long-term memory for the harness agent.

The memory is intentionally file-backed and dependency-free so it can run in
the evaluation workers without an embedding service. Retrieval uses lexical
overlap plus family/tag/recency/confidence weights, which is good enough for
agent strategy lessons such as "do not repeat the same search query".
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MEMORY_DIR = Path(__file__).resolve().parent / "memory"
DEFAULT_MEMORY_PATH = DEFAULT_MEMORY_DIR / "lessons.jsonl"

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def _now() -> float:
    return time.time()


def _clip(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip() + f"\n...[truncated at {max_chars} chars]"
    return text


def _tokens(text: str) -> set[str]:
    text = (text or "").lower()
    return {m.group(0) for m in _WORD_RE.finditer(text) if len(m.group(0).strip()) > 0}


def _fingerprint(parts: Iterable[str]) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update((part or "").strip().lower().encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:16]


@dataclass
class MemoryEntry:
    id: str
    task_family: str
    category: str
    outcome: str
    lesson: str
    strategy: str = ""
    avoid: str = ""
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.55
    source_task_id: str = ""
    source: str = "heuristic"
    created_at: float = field(default_factory=_now)
    last_accessed: float = field(default_factory=_now)
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=str(data.get("id", "")),
            task_family=str(data.get("task_family", "generic") or "generic"),
            category=str(data.get("category", "strategy") or "strategy"),
            outcome=str(data.get("outcome", "unknown") or "unknown"),
            lesson=str(data.get("lesson", "") or ""),
            strategy=str(data.get("strategy", "") or ""),
            avoid=str(data.get("avoid", "") or ""),
            tags=[str(t) for t in data.get("tags", []) if str(t).strip()],
            confidence=float(data.get("confidence", 0.55) or 0.55),
            source_task_id=str(data.get("source_task_id", "") or ""),
            source=str(data.get("source", "heuristic") or "heuristic"),
            created_at=float(data.get("created_at", _now()) or _now()),
            last_accessed=float(data.get("last_accessed", _now()) or _now()),
            access_count=int(data.get("access_count", 0) or 0),
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def text_blob(self) -> str:
        tags = " ".join(self.tags)
        return " ".join(
            [
                self.task_family,
                self.category,
                self.outcome,
                self.lesson,
                self.strategy,
                self.avoid,
                tags,
            ]
        )


class EvolutionMemory:
    """JSONL-backed strategy memory with safe append/update operations."""

    def __init__(
        self,
        memory_path: str | Path | None = None,
        *,
        max_entries: int = 600,
    ) -> None:
        path = Path(memory_path or os.getenv("AGENT_MEMORY_PATH", "") or DEFAULT_MEMORY_PATH)
        if path.suffix:
            self.path = path
            self.dir = path.parent
        else:
            self.dir = path
            self.path = path / "lessons.jsonl"
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.max_entries = max_entries

    @contextmanager
    def _locked(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def load_all(self) -> list[MemoryEntry]:
        if not self.path.exists():
            return []
        entries: list[MemoryEntry] = []
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, dict):
                        entry = MemoryEntry.from_dict(data)
                        if entry.lesson:
                            entries.append(entry)
        except OSError:
            return []
        return entries

    def _write_all(self, entries: list[MemoryEntry]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        entries = sorted(entries, key=lambda e: (e.confidence, e.last_accessed), reverse=True)
        entries = entries[: self.max_entries]
        with self.path.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def add(
        self,
        *,
        task_family: str,
        category: str,
        outcome: str,
        lesson: str,
        strategy: str = "",
        avoid: str = "",
        tags: Iterable[str] | None = None,
        confidence: float = 0.55,
        source_task_id: str = "",
        source: str = "heuristic",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        lesson = _clip(lesson, 700)
        strategy = _clip(strategy, 700)
        avoid = _clip(avoid, 500)
        clean_tags = sorted({str(t).strip().lower() for t in (tags or []) if str(t).strip()})
        if len(lesson) < 8 and len(strategy) < 8:
            return ""

        memory_id = _fingerprint([task_family, category, outcome, lesson, strategy, avoid])
        confidence = max(0.05, min(float(confidence), 0.98))
        now = _now()

        with self._locked():
            entries = self.load_all()
            # Check for exact duplicate
            for entry in entries:
                if entry.id == memory_id:
                    entry.confidence = min(0.98, entry.confidence + 0.05)
                    entry.last_accessed = now
                    entry.access_count += 1
                    entry.tags = sorted(set(entry.tags).union(clean_tags))
                    if source_task_id:
                        entry.source_task_id = source_task_id
                    if metadata:
                        entry.metadata.update(metadata)
                    self._write_all(entries)
                    return entry.id

            # Check for similar lessons (same family+category+outcome) and consolidate
            for entry in entries:
                if (entry.task_family == task_family
                    and entry.category == category
                    and entry.outcome == outcome):
                    overlap = len(_tokens(lesson) & _tokens(entry.lesson))
                    similarity = overlap / max(6, min(len(_tokens(lesson)), len(_tokens(entry.lesson))))
                    if similarity > 0.6:
                        # Consolidate: keep the longer/better lesson
                        if len(lesson) > len(entry.lesson):
                            entry.lesson = lesson
                            entry.strategy = strategy or entry.strategy
                            entry.avoid = avoid or entry.avoid
                        entry.confidence = min(0.98, max(entry.confidence, confidence) + 0.03)
                        entry.last_accessed = now
                        entry.access_count += 1
                        entry.tags = sorted(set(entry.tags).union(clean_tags))
                        self._write_all(entries)
                        return entry.id

            entry = MemoryEntry(
                id=memory_id,
                task_family=task_family or "generic",
                category=category or "strategy",
                outcome=outcome or "unknown",
                lesson=lesson,
                strategy=strategy,
                avoid=avoid,
                tags=clean_tags,
                confidence=confidence,
                source_task_id=source_task_id,
                source=source,
                metadata=metadata or {},
            )
            entries.append(entry)
            self._write_all(entries)
            return entry.id

    def recall(
        self,
        query: str,
        *,
        task_family: str = "",
        tags: Iterable[str] | None = None,
        top_k: int = 4,
        min_score: float = 0.10,
    ) -> list[tuple[MemoryEntry, float]]:
        entries = self.load_all()
        if not entries:
            return []

        q_tokens = _tokens(query)
        q_tags = {str(t).strip().lower() for t in (tags or []) if str(t).strip()}
        now = _now()
        scored: list[tuple[MemoryEntry, float]] = []

        for entry in entries:
            e_tokens = _tokens(entry.text_blob())
            overlap = len(q_tokens & e_tokens)
            lexical = overlap / max(6, min(len(q_tokens) or 1, len(e_tokens) or 1))
            if q_tokens and not overlap and entry.task_family not in (task_family, "generic"):
                lexical = 0.0

            family_score = 0.0
            if task_family and entry.task_family == task_family:
                family_score = 0.30
            elif entry.task_family == "generic":
                family_score = 0.10

            tag_score = 0.0
            if q_tags and entry.tags:
                tag_score = 0.08 * len(q_tags & set(entry.tags))

            # Time-based confidence decay (half-life = 30 days)
            age_days = max(0.0, (now - entry.created_at) / 86400.0)
            recency = math.exp(-age_days * math.log(2) / 30.0)
            effective_confidence = entry.confidence * recency

            # Access frequency bonus (capped)
            access_bonus = min(entry.access_count, 8) * 0.015

            # Composite score with balanced weights
            score = (
                0.40 * lexical
                + family_score
                + tag_score
                + 0.10 * recency
                + 0.20 * effective_confidence
                + access_bonus
            )
            if score >= min_score:
                scored.append((entry, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(0, int(top_k))]

    def mark_recalled(self, entry_ids: Iterable[str]) -> None:
        ids = {entry_id for entry_id in entry_ids if entry_id}
        if not ids or not self.path.exists():
            return
        now = _now()
        with self._locked():
            entries = self.load_all()
            changed = False
            for entry in entries:
                if entry.id in ids:
                    entry.access_count += 1
                    entry.last_accessed = now
                    changed = True
            if changed:
                self._write_all(entries)

    @staticmethod
    def format_for_prompt(
        recalled: list[tuple[MemoryEntry, float]],
        *,
        max_chars: int = 1800,
        short_term: list[dict] | None = None,
    ) -> str:
        if not recalled and not short_term:
            return ""
        lines = [
            "以下是与当前任务相关的经验教训，仅用于改进策略，不要在最终答案中复述："
        ]
        total = len(lines[0])

        # Short-term memory (cross-task within batch)
        if short_term:
            for idx, stm in enumerate(short_term[:3], start=1):
                line = f"[近期] {stm.get('lesson', '')}"
                if stm.get("strategy"):
                    line += f" 建议：{stm['strategy']}"
                if total + len(line) > max_chars:
                    break
                lines.append(line)
                total += len(line)

        # Long-term memory
        for idx, (entry, score) in enumerate(recalled, start=1):
            parts = [
                f"{idx}. [{entry.task_family}/{entry.category}]",
                f"经验：{entry.lesson}",
            ]
            if entry.strategy:
                parts.append(f"建议：{entry.strategy}")
            if entry.avoid:
                parts.append(f"避免：{entry.avoid}")
            line = " ".join(parts)
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Short-term memory buffer (cross-task within batch)
# ---------------------------------------------------------------------------

class ShortTermMemory:
    """In-memory buffer for cross-task learning within a single batch run."""

    def __init__(self, max_entries: int = 10) -> None:
        self.max_entries = max_entries
        self._buffer: list[dict] = []

    def add(self, lesson: str, strategy: str = "", task_family: str = "") -> None:
        if not lesson or len(lesson) < 8:
            return
        self._buffer.append({
            "lesson": lesson[:300],
            "strategy": strategy[:200],
            "task_family": task_family,
            "timestamp": _now(),
        })
        if len(self._buffer) > self.max_entries:
            self._buffer = self._buffer[-self.max_entries:]

    def recall(self, task_family: str = "", top_k: int = 3) -> list[dict]:
        if not self._buffer:
            return []
        relevant = [
            e for e in self._buffer
            if not task_family or e.get("task_family") in (task_family, "generic", "")
        ]
        return relevant[-top_k:]

    def clear(self) -> None:
        self._buffer.clear()

