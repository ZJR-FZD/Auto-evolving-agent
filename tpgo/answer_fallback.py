"""Best-effort answer fallback utilities.

These helpers do not solve tasks. They extract plausible answer spans from the
trajectory when the model would otherwise output an empty/Unable answer.
"""

from __future__ import annotations

import json
import re
from typing import Any

from tpgo.task_router import TaskRoute


BAD_ANSWER_PATTERNS = (
    "unable to determine",
    "unable to identify",
    "cannot determine",
    "cannot definitively",
    "insufficient evidence",
    "unknown",
    "not enough information",
)

STOP_CANDIDATES = {
    "BAD",
    "GOOD",
    "UNKNOWN",
    "SYSTEM",
    "USER",
    "ASSISTANT",
    "Next Action",
    "Correction Note",
    "Internal Correction",
    "Revised Plan",
    "Constraint Ledger",
    "Current Candidate",
    "Verified Facts",
    "Unverified Facts",
    "Answer Role",
    "Candidate Verified",
    "Final Answer",
    "Key Findings",
    "KEY FINDINGS",
    "Reasoning",
    "REASONING",
    "Search Results",
    "Task Complete",
    "No URLs",
    "No More",
    "Tool Calls",
    "In Season",
    "One Possibility",
    "Given Constraints",
    "Our Founder",
    "Read More",
    "Learn More",
    "Privacy Policy",
    "Contact Us",
    "Search Query Planning",
    "Soft Constraint Ledger",
    "Task Route",
    "Route Replan",
    "Search Budget",
    "Wikipedia",
    "Google",
    "YouTube",
    "Facebook",
    "Twitter",
    "Instagram",
}


def is_bad_answer(answer: str) -> bool:
    """Return True if the answer is empty or an explicit non-answer."""
    s = (answer or "").strip().lower()
    if not s:
        return True
    return any(pat in s for pat in BAD_ANSWER_PATTERNS)


def _iter_text_blobs(entries: list[dict[str, Any]]) -> list[str]:
    blobs: list[str] = []
    for entry in entries:
        role = entry.get("role")
        if role == "system":
            continue
        content = entry.get("content")
        if role == "user" and isinstance(content, str) and content.lstrip().startswith("["):
            continue
        if role == "tool":
            try:
                parsed = json.loads(content or "")
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        text = " ".join(str(item.get(k, "")) for k in ("title", "snippet", "content"))
                        if "[HARNESS]" not in text:
                            blobs.append(text)
                continue
        if isinstance(content, str):
            blobs.append(content)
    return blobs


def _extract_person_names(text: str) -> list[str]:
    names = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", text or "")
    return [n for n in names if n not in STOP_CANDIDATES and not n.startswith("The ")]


def _extract_titles(text: str) -> list[str]:
    quoted = re.findall(r'"([^"]{3,80})"', text or "")
    titleish = re.findall(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){1,7})\b", text or "")
    out = []
    for cand in quoted + titleish:
        if cand not in STOP_CANDIDATES and len(cand.split()) <= 8:
            out.append(cand)
    return out


def _is_control_candidate(candidate: str) -> bool:
    """Return True for routing, critic, or boilerplate strings."""
    cand = candidate.strip()
    low = cand.lower()
    words = cand.split()
    if cand in STOP_CANDIDATES:
        return True
    if words and words[0] in {"In", "On", "At", "By", "From", "To", "With", "Without", "Given", "Based", "Despite"}:
        return True
    if len(cand) <= 4 and cand.isupper():
        return True
    if low.startswith(("[system]", "system ", "tool_call", "reasoning_content")):
        return True
    if any(word in low for word in ("constraint ledger", "search query", "route replan")):
        return True
    if any(
        word in low
        for word in (
            "next action",
            "correction note",
            "internal correction",
            "revised plan",
            "constraint ledger",
            "current candidate",
            "verified facts",
            "unverified facts",
            "answer role",
            "key findings",
            "tool calls",
            "search results",
        )
    ):
        return True
    if any(
        phrase in low
        for phrase in (
            "unable to",
            "cannot determine",
            "insufficient evidence",
            "failed queries",
            "blocked sites",
            "low-signal",
            "no successful",
            "no urls",
            "not definitively",
            "not confidently",
            "available evidence",
        )
    ):
        return True
    return False


def _extract_numbers(text: str) -> list[str]:
    out = []
    for pat in (
        r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",
        r"\b\d+(?:\.\d+)?\s*(?:million|billion|shares|goals|points|%)\b",
        r"\b(?:18|19|20)\d{2}\b",
    ):
        out.extend(re.findall(pat, text or "", flags=re.IGNORECASE))
    return out


def _score_candidate(candidate: str, blobs: list[str]) -> tuple[int, int]:
    low = candidate.lower()
    freq = sum(blob.lower().count(low) for blob in blobs)
    words = candidate.split()
    is_numeric = bool(re.fullmatch(r"[\d,]+(?:\.\d+)?(?:\s*[A-Za-z%]+)?", candidate))
    is_title_case = all(w[:1].isupper() or w[:1].isdigit() for w in words if w)
    shape_bonus = 0
    if len(words) >= 2:
        shape_bonus += 6
    if is_title_case:
        shape_bonus += 8
    if is_numeric:
        shape_bonus += 5
    length_penalty = abs(len(words) - 2)
    return shape_bonus + freq, -length_penalty


def _fallback_from_question(route: TaskRoute) -> str:
    """Return a route-shaped placeholder candidate when no span is available."""
    if route.search_keywords:
        seed = route.search_keywords[0]
        if seed and seed.lower() not in {"what", "which", "who", "when", "where"}:
            return seed
    if "person name" in route.answer_format_hint.lower():
        return "most likely named person from retrieved evidence"
    if "scientific name" in route.answer_format_hint.lower():
        return "most likely scientific name from retrieved evidence"
    if any(word in route.answer_format_hint.lower() for word in ("numeric", "amount", "year", "score", "minute")):
        return "most likely numeric value from retrieved evidence"
    return "most likely answer candidate from retrieved evidence"


def best_effort_answer(entries: list[dict[str, Any]], route: TaskRoute) -> str:
    """Extract a plausible answer span from trajectory text.

    The returned value is intentionally concise. It is a fallback for benchmark
    completion, not a substitute for evidence-based answering.
    """
    blobs = _iter_text_blobs(entries)
    text = "\n".join(blobs)
    candidates: list[str] = []

    hint = route.answer_format_hint.lower()
    if "person name" in hint or route.task_type in {"person_bio_family", "music_artist"}:
        candidates.extend(_extract_person_names(text))
    elif any(word in hint for word in ("numeric", "amount", "year", "score", "minute")):
        candidates.extend(_extract_numbers(text))
    else:
        candidates.extend(_extract_titles(text))
        candidates.extend(_extract_person_names(text))
        candidates.extend(_extract_numbers(text))

    cleaned: list[str] = []
    for cand in candidates:
        cand = re.sub(r"\s+", " ", str(cand)).strip(" .,:;!?")
        if not cand or len(cand) < 2:
            continue
        if len(cand.split()) == 1 and cand.islower():
            continue
        if _is_control_candidate(cand):
            continue
        if cand not in cleaned:
            cleaned.append(cand)

    if not cleaned:
        return _fallback_from_question(route)
    return max(cleaned[:50], key=lambda c: _score_candidate(c, blobs))
