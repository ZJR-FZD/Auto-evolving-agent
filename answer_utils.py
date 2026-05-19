"""Answer extraction and normalization helpers.

These functions are deliberately gold-free: they only inspect the model
prediction and, optionally, the question type.  Gold answers may be used by
evaluation scripts to score the cleaned prediction, but never to create it.
"""

from __future__ import annotations

import re
import string
from typing import Any


UNCERTAIN_PATTERNS = (
    r"unable to determine",
    r"cannot determine",
    r"insufficient evidence",
    r"not enough evidence",
    r"not definitively identify",
    r"does not definitively identify",
    r"could not find",
    r"无法确定",
    r"证据不足",
)


def strip_answer_tag(text: Any) -> str:
    s = "" if text is None else str(text).strip()
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", s, flags=re.S | re.I)
    return match.group(1).strip() if match else s


def remove_markdown(text: str) -> str:
    s = str(text or "").strip()
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    return s.strip(" \t\r\n\"'“”‘’")


def normalize_yes_no(question: str, pred: str) -> str:
    q = str(question or "").strip().lower()
    s = str(pred or "").strip()
    low = s.lower()
    is_yes_no = q.startswith(
        (
            "are ",
            "is ",
            "was ",
            "were ",
            "do ",
            "does ",
            "did ",
            "can ",
            "could ",
            "would ",
            "has ",
            "have ",
        )
    )
    if not is_yes_no:
        return s
    if re.match(r"^\s*yes\b", low):
        return "yes"
    if re.match(r"^\s*no\b", low):
        return "no"
    if re.match(r"^\s*(否|不|不是|并非)", s):
        return "no"
    if re.match(r"^\s*(是|对|相同|同一)", s):
        return "yes"
    if "not the same" in low or "different" in low or "not both" in low:
        return "no"
    if any(x in s for x in ("不同", "不是同", "并非同", "不在同", "不是来自同")):
        return "no"
    if any(x in s for x in ("相同", "同一个国家", "同一国家", "都是")) and "不" not in s:
        return "yes"
    if "both" in low and "same" in low and "not" not in low:
        return "yes"
    return s


def extract_comparison_answer(question: str, text: str) -> str:
    """Extract the entity selected by comparison-style answers."""
    q = str(question or "").lower()
    s = str(text or "").strip()
    if not s:
        return s

    # "Nancy Ditz was born later. ..." -> "Nancy Ditz"
    direct_patterns = (
        r"^(.{2,120}?)\s+(?:was|were|is|are)\s+born\s+(?:later|earlier|first|last)\b",
        r"^(.{2,120}?)\s+(?:died|passed away)\s+(?:first|earlier|later|last)\b",
        r"^(.{2,120}?)\s+(?:was|were|is|are)\s+(?:established|founded|created|released|published)\s+(?:first|earlier|later|last)\b",
        r"^(.{2,120}?)\s+\([^)]{1,40}\)\s+(?:was|were|is|are)\s+(?:established|founded|created|released|published)\s+(?:first|earlier|later|last)\b",
    )
    if any(token in q for token in ("born later", "born earlier", "died first", "died earlier", "established first", "founded first")):
        for pattern in direct_patterns:
            match = re.search(pattern, s, flags=re.I)
            if match:
                return _strip_parenthetical_year(match.group(1).strip())

    # "X (1952) was established first, before Y (1968)." even when the
    # question says "Which school was established first".
    if re.search(r"\bwhich\b.*\b(first|later|earlier|older|younger)\b", q):
        for pattern in direct_patterns:
            match = re.search(pattern, s, flags=re.I)
            if match:
                return _strip_parenthetical_year(match.group(1).strip())

    return s


def extract_place_answer(question: str, text: str) -> str:
    q = str(question or "").lower()
    s = str(text or "").strip()
    if not s:
        return s
    # "What is the place of birth..." in these datasets often expects the
    # primary town/city.  For "Where was X born?", keep the full location span
    # because the gold may be a district/state after the first comma.
    asks_birth_place = "place of birth" in q or "birthplace" in q
    if asks_birth_place and "," in s:
        first = s.split(",", 1)[0].strip()
        if 1 <= len(first) <= 80:
            return first
    return s


def _strip_parenthetical_year(text: str) -> str:
    return re.sub(r"\s*\(\s*(?:c\.\s*)?\d{3,4}[^)]*\)\s*$", "", text).strip()


def extract_bold_or_quoted_entity(text: str) -> str:
    s = str(text or "").strip()
    match = re.search(
        r"(?:answer|likely|appears|is)[^*\n]{0,40}\*\*([^*]{1,100})\*\*",
        s,
        flags=re.I | re.S,
    )
    if match:
        return match.group(1).strip()
    quotes = re.findall(r"[\"“”']([^\"“”']{2,100})[\"“”']", s)
    if quotes:
        quotes = sorted(quotes, key=len, reverse=True)
        return quotes[0].strip()
    return s


def extract_bold_entity(text: str) -> str:
    """Prefer a bolded entity only when it is clearly introduced as answer.

    This must run before markdown stripping.  It avoids losing candidates like
    "the most likely answer is **Evergrande** (...)" while not blindly taking
    every bold phrase from an evidence-chain explanation.
    """
    s = str(text or "").strip()
    match = re.search(
        r"(?:answer|likely answer|most likely answer|答案|最终答案)[^*\n]{0,60}\*\*([^*\n]{1,100})\*\*",
        s,
        flags=re.I | re.S,
    )
    if match:
        return match.group(1).strip()
    return s


def extract_after_answer_cue(text: str) -> str:
    s = str(text or "").strip()
    patterns = (
        r"(?:final answer|the answer is|answer is|answer)\s*[:：]?\s*(.+)$",
        r"(?:答案是|最终答案是|答案[:：])\s*(.+)$",
        r"(?:most likely answer is|likely answer is)\s*(.+)$",
        r"(?:is likely|appears to be)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, s, flags=re.I | re.S)
        if match:
            return match.group(1).strip()
    return s


def extract_person_or_entity_sentence_start(question: str, text: str) -> str:
    """Clean common verbose final answers without using gold.

    The benchmark often receives generations such as:
    "Based on the search results, the teen ... was Jayben Camacho, a 16-year-old..."
    or "Nancy Ditz was born later...".  These should become the selected span.
    """
    q = str(question or "").lower()
    s = str(text or "").strip()
    if not s:
        return s

    asks_entity = any(
        token in q
        for token in (
            "who",
            "which person",
            "what is the name",
            "full name",
            "teen",
            "player",
            "ceo",
            "founder",
        )
    )
    if asks_entity or re.search(r"\bbased on\b|\bsearch results\b|\bevidence\b", s, flags=re.I):
        patterns = (
            r"\b(?:was|is|were|are)\s+([A-Z][A-Za-z0-9.'’&-]*(?:\s+[A-Z][A-Za-z0-9.'’&-]*){0,5})(?:,|\s+(?:who|from|of|in|born|died|won|founded|served|became)\b|\.|$)",
            r"^([A-Z][A-Za-z0-9.'’&-]*(?:\s+[A-Z][A-Za-z0-9.'’&-]*){0,5})\s+(?:was|is|were|are|won|died|founded|served|became)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, s)
            if match:
                candidate = match.group(1).strip()
                if candidate.lower() not in {"based", "the answer", "answer"}:
                    return candidate
    return s


def truncate_explanation(text: str) -> str:
    s = str(text or "").strip()
    lines = [line.strip() for line in s.splitlines() if line.strip()]
    if len(lines) > 1:
        short_lines = [
            line
            for line in lines
            if len(line) <= 90 and not re.search(r"based on|because|however|therefore|分析|证据", line, re.I)
        ]
        s = short_lines[-1] if short_lines else lines[0]

    splitters = (
        r"\s+because\s+",
        r"\s+however\s+",
        r"\s+although\s+",
        r"\s+but\s+",
        r"\s+which\s+",
        r"\s+who\s+",
        r"\s+where\s+",
        r"\s+according to\s+",
        r"\s+based on\s+",
        r"\s+但",
        r"\s+因为",
    )
    for splitter in splitters:
        parts = re.split(splitter, s, maxsplit=1, flags=re.I)
        if len(parts) > 1 and 1 <= len(parts[0].strip()) <= 100:
            s = parts[0].strip()
            break

    if len(s) > 120:
        for sep in (". ", "; ", "；", "。"):
            if sep in s:
                candidate = s.split(sep)[0].strip()
                if 1 <= len(candidate) <= 120:
                    s = candidate
                    break
    return s.strip(" .;；。,:：")


def clean_pred_for_submit(pred: Any, question: str = "") -> str:
    """Extract a concise final answer span without using gold answers."""
    s = strip_answer_tag(pred)

    if any(re.search(pattern, s, flags=re.I) for pattern in UNCERTAIN_PATTERNS):
        return "Insufficient evidence"

    s = normalize_yes_no(question, s)
    if s.lower() in {"yes", "no"}:
        return s.lower()

    s = extract_bold_entity(s)
    s = remove_markdown(s)
    s = extract_bold_or_quoted_entity(s)
    s = extract_after_answer_cue(s)
    s = remove_markdown(s)
    s = extract_comparison_answer(question, s)
    s = extract_person_or_entity_sentence_start(question, s)
    s = extract_place_answer(question, s)
    s = truncate_explanation(s)
    s = remove_markdown(s)
    s = _strip_parenthetical_year(s)
    s = re.sub(
        r"\s*\((?:based on|born|founded|according|not enough|evidence|分析|证据).*?\)\s*$",
        "",
        s,
        flags=re.I,
    )
    s = re.sub(r"\s+", " ", s).strip(" .;；。,:：")
    return s or "Insufficient evidence"


def infer_answer_type(question: str) -> str:
    q = str(question or "").strip().lower()
    if q.startswith(("are ", "is ", "was ", "were ", "do ", "does ", "did ", "has ", "have ")):
        return "yes_no"
    if q.startswith("when "):
        return "date"
    if q.startswith("where "):
        return "location"
    if q.startswith("who "):
        return "person"
    if q.startswith("how many ") or "number of" in q:
        return "number"
    if "country" in q or "nationality" in q:
        return "country_or_nationality"
    if "which film" in q or "which movie" in q or "episode" in q or "book" in q:
        return "title"
    return "entity"


def normalize_for_metric(text: Any) -> str:
    s = clean_pred_for_submit(text)
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\s+", " ", s).strip()
    return s
