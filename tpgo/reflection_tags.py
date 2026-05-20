"""Failure-tag inference for routed search tasks.

The functions in this module inspect the question, route metadata, and an
optional trace summary. They never use gold answers.
"""

from __future__ import annotations

import re
from typing import Any

from tpgo.task_router import TaskRoute


FAILURE_TAGS = (
    "entity_identification_failure",
    "source_retrieval_failure",
    "ambiguous_name_collision",
    "insufficient_primary_evidence",
    "numeric_extraction_error",
    "temporal_constraint_mismatch",
    "overlong_reasoning_chain",
    "answer_format_error",
)


def _add(tags: list[str], tag: str) -> None:
    """Append a known tag once."""
    if tag in FAILURE_TAGS and tag not in tags:
        tags.append(tag)


def _trace_value(trace: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    """Read a trace key safely."""
    if not isinstance(trace, dict):
        return default
    return trace.get(key, default)


def infer_failure_tags(
    question: str,
    route: TaskRoute,
    trace: dict | None = None,
) -> list[str]:
    """Infer likely failure tags without using a gold answer.

    Args:
        question: Original question text.
        route: TaskRoute generated from the question.
        trace: Optional run summary, e.g. search count, low-signal count, or
            final-answer status.

    Returns:
        Ordered list of failure tags.
    """
    q = (question or "").lower()
    tags: list[str] = []

    if route.task_type == "general_multihop" or route.confidence < 0.5:
        _add(tags, "entity_identification_failure")

    for tag in route.risk_tags:
        _add(tags, tag)

    if re.search(r"\b(before|after|between|prior to|inclusive|earlier|later)\b", q):
        _add(tags, "temporal_constraint_mismatch")
    if re.search(r"\b\d+(?:st|nd|rd|th)?\b", q):
        _add(tags, "numeric_extraction_error")
    if any(term in q for term in ("first and last", "exact", "only", "format")):
        _add(tags, "answer_format_error")

    if route.task_type in {"company_finance", "science_species", "archive_permit_history"}:
        _add(tags, "insufficient_primary_evidence")
    if route.task_type in {"person_bio_family", "film_tv_game", "music_artist"}:
        _add(tags, "ambiguous_name_collision")

    search_calls = int(_trace_value(trace, "search_calls", 0) or 0)
    low_signal = int(_trace_value(trace, "low_signal_tool_results", 0) or 0)
    critic_bad = int(_trace_value(trace, "critic_bad", 0) or 0)
    steps = int(_trace_value(trace, "steps", 0) or 0)
    has_final = bool(_trace_value(trace, "has_final_answer", True))

    if search_calls >= 8 or steps >= 14 or critic_bad >= 5:
        _add(tags, "overlong_reasoning_chain")
    if low_signal >= 2 or (search_calls and low_signal / max(search_calls, 1) >= 0.4):
        _add(tags, "source_retrieval_failure")
    if not has_final:
        _add(tags, "answer_format_error")

    return tags
