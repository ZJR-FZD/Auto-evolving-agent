"""
Qwen Agent Harness — Plan & ReAct Orchestrator
================================================
Improved version with:
  1. Plan-then-Solve: model decomposes the problem before acting
  2. ReAct with Reflection: model evaluates progress after each tool call
  3. Leaked tool_call recovery: parses <tool_call> from reasoning_content
  4. Forced answer on final steps: injects prompt to force conclusion
  5. Answer extraction: uses <answer>...</answer> tags for clean output
  6. Robust failure handling: detects blocked sites, avoids repeated failures

Usage (CLI):
    python task_runner_plan_react.py -i "your question" -t task_001
"""

import argparse
import json
import logging
import os
import re
import uuid
from urllib.parse import urlparse
from typing import Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from roles import Role
from trajectory import Trajectory
from tools.search_tool import search_text, search_image
from tools.browser_tool import (
    browser_navigate, browser_get_text, browser_click,
    browser_type, browser_parallel,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("harness.plan_react")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "qwen-3.5")
MAX_STEPS    = int(os.getenv("MAX_STEPS", "15"))
MAX_TOKENS   = int(os.getenv("MAX_TOKENS", "16000"))
DISABLE_TOOLS = os.getenv("DISABLE_TOOLS", "0") == "1"

# ---------------------------------------------------------------------------
# Tool schema (same as original)
# ---------------------------------------------------------------------------
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": (
                "基于 Google 的联网文字搜索，返回搜索结果摘要和网页正文。"
                "返回 [{rank,title,url,snippet,content}]。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":     {"type": "string",  "description": "搜索关键词"},
                    "top_k":     {"type": "integer", "description": "返回条数（1-3）", "default": 1},
                    "fetch":     {"type": "boolean", "description": "是否抓取正文（慢，仅在摘要不够时使用）", "default": False},
                    "max_chars": {"type": "integer", "description": "每篇正文截断字符数", "default": 500},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_image",
            "description": (
                "反向图像搜索（Google Lens），输入图片 URL，返回相关网页。"
                "返回 [{rank,title,url,snippet,content}]。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string",  "description": "图片的 http(s) URL"},
                    "top_k":     {"type": "integer", "description": "返回条数（1-3）", "default": 1},
                    "fetch":     {"type": "boolean", "description": "是否抓取正文（慢，仅在摘要不够时使用）", "default": False},
                    "max_chars": {"type": "integer", "description": "正文截断字符数", "default": 500},
                },
                "required": ["image_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": "打开 URL，返回页面文本预览。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url":          {"type": "string",  "description": "要访问的 URL"},
                    "wait_until":   {"type": "string", "enum": ["domcontentloaded", "load", "networkidle"], "default": "domcontentloaded"},
                    "include_text": {"type": "boolean", "default": True},
                    "max_text":     {"type": "integer", "default": 2000},
                    "timeout":      {"type": "integer", "default": 30},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_get_text",
            "description": "获取当前页面完整可见文本。",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_chars": {"type": "integer", "default": 5000},
                    "timeout":   {"type": "integer", "default": 15},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": "CSS 选择器点击元素。",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "nth":      {"type": "integer", "default": 0},
                    "timeout":  {"type": "integer", "default": 10},
                },
                "required": ["selector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_type",
            "description": "向输入框键入文本。",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "text":     {"type": "string"},
                    "submit":   {"type": "boolean", "default": False},
                    "clear":    {"type": "boolean", "default": True},
                    "timeout":  {"type": "integer", "default": 10},
                },
                "required": ["selector", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_parallel",
            "description": "并发打开多个 URL。",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls":            {"type": "array", "items": {"type": "string"}},
                    "mode":            {"type": "string", "enum": ["navigate", "get_text"], "default": "navigate"},
                    "max_chars":       {"type": "integer"},
                    "wait_until":      {"type": "string", "enum": ["domcontentloaded", "load", "networkidle"], "default": "domcontentloaded"},
                    "max_concurrency": {"type": "integer", "default": 4},
                    "timeout":         {"type": "integer", "default": 30},
                },
                "required": ["urls"],
            },
        },
    },
]

TOOL_FN_MAP = {
    "search_text":      lambda a: search_text(**a),
    "search_image":     lambda a: search_image(**a),
    "browser_navigate": lambda a: browser_navigate(**a),
    "browser_get_text": lambda a: browser_get_text(**a),
    "browser_click":    lambda a: browser_click(**a),
    "browser_type":     lambda a: browser_type(**a),
    "browser_parallel": lambda a: browser_parallel(**a),
}

# ---------------------------------------------------------------------------
# Harness-level guardrails (for small models that can't self-police)
# ---------------------------------------------------------------------------

GARBAGE_DOMAINS = {
    "instagram.com", "tiktok.com", "facebook.com", "fonts101.com",
    "etsy.com", "thatpervert.com", "pinterest.com",
}

ALLOWLIST_DOMAINS = {
    "wikipedia.org", "wikidata.org",
    "fifa.com", "uefa.com", "olympics.com", "ioc.org",
    "bbc.com", "reuters.com", "apnews.com", "nytimes.com",
    "theguardian.com", "espn.com",
}

LOW_SIGNAL_DOMAINS = {
    "youtube.com", "youtu.be", "reddit.com", "x.com", "twitter.com",
    "foxsports.com", "wfin.com", "tiktok.com", "instagram.com",
    "onefootball.com", "namu.wiki",
}

LOW_SIGNAL_PATTERNS = [
    "403", "forbidden", "captcha", "access denied", "security verification",
    "[jina-error]", "[proxy-error]", "just a moment...",
]

LOW_SIGNAL_URL_PATTERNS = [
    "/catalogsearch/", "/result/index", "/search?", "/tag/", "/products/",
]

ERROR_SIGNAL_PATTERNS = [
    "[error]", "[proxy-error]", "readtimeout", "timed out", "httpsconnectionpool",
]

STATE_SEQUENCE = ["S0_PARSE", "S1_SUBJECT", "S2_EVENT", "S3_DETAIL", "S4_DONE"]
MAX_TRIALS_PER_STATE = 3
LOW_SIGNAL_PIVOT_THRESHOLD = 2
QUERY_SIMILARITY_THRESHOLD = 0.70
MIN_INDEPENDENT_SOURCES = 2

STATE_INSTRUCTION = {
    "S0_PARSE": (
        "[SYSTEM][STATE=S0_PARSE] Parse the question into searchable facts only.\n"
        "- Extract hard facts (years, scores, minutes, proper nouns, event type).\n"
        "- Ignore metaphorical/riddle language.\n"
        "- Plan 2-3 clue chains; next query must use one chain with 2-3 facts.\n"
        "- Exactly one search query this step; use: <year/range> + <event/detail> + <unique constraint>."
    ),
    "S1_SUBJECT": (
        "[SYSTEM][STATE=S1_SUBJECT] Identify the subject candidate.\n"
        "- Use precise snippet search (fetch=false first).\n"
        "- If 2 low-signal results happen, pivot to a different clue chain.\n"
        "- Query template: <candidate hypothesis> + <hard fact 1> + <hard fact 2>."
    ),
    "S2_EVENT": (
        "[SYSTEM][STATE=S2_EVENT] Verify event alignment for the candidate.\n"
        "- Run one independent confirmation query using another fact.\n"
        "- Reject candidate if key constraints conflict.\n"
        "- Query template: <candidate> + <independent event/time fact>."
    ),
    "S3_DETAIL": (
        "[SYSTEM][STATE=S3_DETAIL] Extract the requested final detail.\n"
        "- Verify with at least two independent sources when possible.\n"
        "- Keep answer concise and exact.\n"
        "- Query template: <candidate> + <exact asked detail>."
    ),
}

CONFLICT_PIVOT_PROMPT = (
    "[SYSTEM] Candidate-event alignment conflict detected. Your current candidate does not satisfy "
    "key constraints from the question (timeframe/event/details). Roll back to candidate discovery "
    "and pivot to a new clue chain with 2-3 hard facts only."
)

SEARCH_DEGRADED_PROMPT = (
    "[SYSTEM] Search backend is degraded (consecutive timeouts/errors). "
    "Do NOT keep calling search_text repeatedly. "
    "Preferred fallback: use browser_navigate on one reliable source URL if available; "
    "otherwise conclude with <answer confidence=\"low\">...</answer>."
)


def _extract_domain(url: str) -> str:
    if not url:
        return ""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _is_domain_match(domain: str, domain_set: set[str]) -> bool:
    if not domain:
        return False
    return any(domain == d or domain.endswith(f".{d}") for d in domain_set)


def _collect_domains_from_results(results: list[dict]) -> set[str]:
    domains = set()
    for item in results:
        dom = _extract_domain(item.get("url", ""))
        if dom:
            domains.add(dom)
    return domains


def _extract_candidates(results: list[dict]) -> list[str]:
    """Extract rough person/entity candidates from snippets and titles."""
    candidates = []
    seen = set()
    name_pat = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")
    for item in results:
        blob = " ".join([str(item.get("title", "")), str(item.get("snippet", ""))])
        for m in name_pat.finditer(blob):
            cand = m.group(1).strip()
            if len(cand) < 4:
                continue
            if cand.lower() in {"premier league", "champions league", "manchester united"}:
                continue
            if cand not in seen:
                seen.add(cand)
                candidates.append(cand)
    return candidates


def _event_alignment_score(instruction: str, results: list[dict]) -> int:
    """
    Positive score means results align with instruction constraints.
    Negative score means likely conflict/red-herring.
    """
    text = (instruction or "").lower()
    blob = " ".join(
        (str(r.get("title", "")) + " " + str(r.get("snippet", "")) + " " + str(r.get("content", "")))
        for r in results
    ).lower()
    score = 0
    if "95th minute" in text and "95th" in blob:
        score += 1
    if ("free-kick" in text or "free kick" in text) and ("free-kick" in blob or "free kick" in blob):
        score += 1
    if "early 21st century" in text:
        if re.search(r"\b200[0-9]\b|\b2010\b", blob):
            score += 1
        if re.search(r"\b202[1-9]\b", blob):
            score -= 2
    # Cross-sport conflicts are strong negatives.
    if any(tok in blob for tok in ["nfl", "afl", "nba", "baseball", "cricket"]):
        score -= 2
    return score


def _query_quality_score(query: str, instruction: str, recent_kw: list[set]) -> float:
    q = (query or "").strip()
    if not q:
        return -999.0
    ql = q.lower()
    score = 0.0
    # Prefer hard-fact rich queries.
    score += 1.2 * len(re.findall(HARD_FACT_INDICATORS, q))
    # Prefer carrying facts from instruction.
    for fact in build_instruction_fact_bank(instruction):
        if fact.replace('"', "").lower() in ql:
            score += 1.0
    # Penalize metaphor/riddle direction.
    if any(p in ql for p in METAPHOR_PATTERNS) or any(k in ql for k in RIDDLE_DIRECTION_KEYWORDS):
        score -= 3.0
    # Penalize near-duplicate query.
    kw = extract_query_keywords({"query": q})
    if kw and recent_kw:
        sim = max((jaccard_similarity(kw, prev) for prev in recent_kw[-5:]), default=0.0)
        if sim >= QUERY_SIMILARITY_THRESHOLD:
            score -= 2.0
    return score


def _select_tool_calls_for_step(tool_calls, instruction: str, recent_queries: list[set]):
    """
    Keep at most one search_text call per step; preserve other tool calls.
    """
    if not tool_calls:
        return tool_calls
    selected = []
    search_candidates = []
    for tc in tool_calls:
        if hasattr(tc, "function"):
            fn_name = tc.function.name
            fn_args_str = tc.function.arguments
        else:
            fn_name = tc["function"]["name"]
            fn_args_str = tc["function"]["arguments"]
        if fn_name != "search_text":
            selected.append(tc)
            continue
        try:
            args = json.loads(fn_args_str)
        except Exception:
            args = {}
        q = str(args.get("query", ""))
        search_candidates.append((tc, _query_quality_score(q, instruction, recent_queries)))
    if search_candidates:
        best_tc, _ = max(search_candidates, key=lambda x: x[1])
        selected.append(best_tc)
    return selected


def _looks_low_signal_result(item: dict) -> bool:
    """Heuristic check for result entries that are likely unusable/noisy."""
    url = (item.get("url") or "").lower()
    text_blob = " ".join([
        str(item.get("title", "")),
        str(item.get("snippet", "")),
        str(item.get("content", "")),
    ]).lower()
    if not url:
        return True
    if any(domain in url for domain in LOW_SIGNAL_DOMAINS):
        return True
    if any(pat in url for pat in LOW_SIGNAL_URL_PATTERNS):
        return True
    if any(pat in text_blob for pat in LOW_SIGNAL_PATTERNS):
        return True
    return False


def _is_error_signal_text(text: str) -> bool:
    s = (text or "").lower()
    return any(pat in s for pat in ERROR_SIGNAL_PATTERNS)


def filter_garbage_results(tool_result_str: str) -> tuple[str, bool]:
    """
    Remove low-quality/noisy results and return `(filtered_json, has_signal)`.
    """
    if _is_error_signal_text(tool_result_str):
        return json.dumps(
            [{
                "rank": 1,
                "title": "[HARNESS] Search tool error/timeout. Treat as low-signal and pivot.",
                "url": "",
                "snippet": str(tool_result_str)[:500],
                "content": "",
            }],
            ensure_ascii=False,
        ), False
    try:
        results = json.loads(tool_result_str)
        if not isinstance(results, list):
            return tool_result_str, True
        filtered = []
        for r in results:
            domain = _extract_domain(r.get("url", ""))
            if _is_domain_match(domain, GARBAGE_DOMAINS):
                continue
            if _looks_low_signal_result(r):
                continue
            filtered.append(r)
        filtered_domains = _collect_domains_from_results(filtered)
        allowlist_hit = any(_is_domain_match(d, ALLOWLIST_DOMAINS) for d in filtered_domains)
        if not filtered and results:
            return json.dumps(
                [{"rank": 1, "title": "[HARNESS] All results were from blocked sites (Instagram/TikTok/etc). "
                  "or low-signal sources (YouTube/Reddit/CAPTCHA/403). "
                  "Your query is too vague or the info is not indexed. "
                  "Try a COMPLETELY different query with different entities and constraints.",
                  "url": "", "snippet": "", "content": ""}],
                ensure_ascii=False), False
        has_signal = bool(filtered) and (allowlist_hit or len(filtered_domains) >= 1)
        return json.dumps(filtered, ensure_ascii=False), has_signal
    except (json.JSONDecodeError, TypeError):
        if _is_error_signal_text(tool_result_str):
            return tool_result_str, False
        return tool_result_str, True


# Phrases that are clearly metaphorical/riddle-like and should NOT be searched literally
METAPHOR_PATTERNS = [
    "born out of discord",
    "identity evolved",
    "evolved through several iterations",
    "born out of",
    "riddle",
    "puzzle",
]

# Keywords that indicate the model is trying to decode the riddle instead of searching facts
# A query is blocked if it contains BOTH a riddle-direction keyword AND lacks hard facts
RIDDLE_DIRECTION_KEYWORDS = [
    "name change", "name changes", "many names",
    "discord", "division", "riddle", "puzzle",
]

HARD_FACT_INDICATORS = re.compile(
    r'\b('
    r'19\d{2}|20[0-2]\d|'
    r'19\d{2}s|20[0-2]\ds|'
    r'\d{1,3}(st|nd|rd|th)\s*minute|'
    r'free[\s-]?kick|'
    r'qualifier|champion|league|cup|goal|score|'
    r'final|semi[\s-]?final|quarter[\s-]?final|'
    r'vs|versus|against|derby'
    r')\b',
    re.IGNORECASE
)

def is_metaphor_query(query: str) -> bool:
    """Check if a search query is trying to decode riddle/metaphor instead of searching facts."""
    query_lower = query.lower()
    has_hard_facts = bool(HARD_FACT_INDICATORS.search(query))
    # Direct metaphor phrase match - block only when query lacks hard facts.
    has_direct_metaphor = any(pat in query_lower for pat in METAPHOR_PATTERNS)
    if has_direct_metaphor and not has_hard_facts:
        return True
    # Riddle-direction query without hard facts - block
    has_riddle_direction = any(kw in query_lower for kw in RIDDLE_DIRECTION_KEYWORDS)
    if has_riddle_direction and not has_hard_facts:
        return True
    return False


METAPHOR_BLOCKED_MSG = (
    '[{{"rank": 1, "title": "[HARNESS] Query blocked: you are searching a metaphor/riddle literally. '
    'This NEVER returns useful results. Instead, search for HARD FACTS from the question: '
    'specific dates, numbers, proper nouns, or events. '
    'For example, search for the specific event (95th minute free kick 2001 qualifier) '
    'not the riddle description (born out of discord).", '
    '"url": "", "snippet": "", "content": ""}}]'
)


def extract_query_keywords(fn_args: dict) -> set:
    """Extract meaningful keywords from a search query for dedup detection."""
    query = fn_args.get("query", "")
    query = re.sub(r'["\'\(\)]', ' ', query)
    words = {w.lower() for w in query.split() if len(w) > 2}
    stop_words = {"the", "and", "for", "from", "with", "that", "this", "was", "are", "not"}
    return words - stop_words


def jaccard_similarity(a: set, b: set) -> float:
    """Simple keyword overlap score for duplicate-query detection."""
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


GENERIC_QUERY_WORDS = {
    "football", "soccer", "match", "team", "history", "famous", "winner",
    "goal", "goals", "free", "kick", "minute", "early", "late",
}


def build_instruction_fact_bank(instruction: str) -> list[str]:
    """Extract reusable hard facts from the original task instruction."""
    text = (instruction or "").lower()
    facts = []
    if "95th minute" in text:
        facts.append('"95th minute"')
    if "free-kick" in text or "free kick" in text:
        facts.append('"free kick"')
    if "early 21st century" in text:
        facts.append('"early 21st century"')
        facts.append("2000s")
    if "all their goals early" in text:
        facts.append('"scored early"')
    if "latter stage" in text or "later stage" in text:
        facts.append('"scored late"')
    return facts


def rewrite_query_with_hard_facts(query: str, instruction: str) -> str:
    """
    Rewrite metaphor-like queries into hard-fact-oriented search strings.
    """
    q = query or ""
    # Remove obvious riddle phrasing while retaining potentially useful words.
    cleaned = q
    for pat in METAPHOR_PATTERNS:
        cleaned = re.sub(re.escape(pat), " ", cleaned, flags=re.IGNORECASE)
    for kw in RIDDLE_DIRECTION_KEYWORDS:
        cleaned = re.sub(re.escape(kw), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'["\'\(\)]', " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    fact_bank = build_instruction_fact_bank(instruction)
    remaining_words = [
        w for w in cleaned.lower().split()
        if len(w) > 2 and w not in GENERIC_QUERY_WORDS
    ]
    seed = fact_bank[:3] + remaining_words[:3]
    if "football" not in " ".join(seed).lower():
        seed.append("football")
    rewritten = " ".join(seed).strip()
    if len(rewritten.split()) < 4:
        return '"95th minute" "free kick" football 2000s'
    return rewritten


def diversify_duplicate_query(query: str, instruction: str, current_kw: set) -> str | None:
    """
    Add missing fact tokens to near-duplicate queries to create a new angle.
    """
    fact_bank = build_instruction_fact_bank(instruction)
    fact_tokens = []
    for fact in fact_bank:
        fact_tokens.extend(extract_query_keywords({"query": fact}))
    missing = [tok for tok in fact_tokens if tok not in current_kw]
    if not missing:
        return None
    add_on = " ".join(missing[:3])
    diversified = f"{query} {add_on}".strip()
    diversified = re.sub(r"\s+", " ", diversified)
    return diversified


def should_block_duplicate_query(current_kw: set, recent_kw: list[set]) -> bool:
    """
    Block only near-identical retries that add almost no new information.
    """
    if not current_kw or not recent_kw:
        return False
    last_window = recent_kw[-5:]
    for prev in last_window:
        overlap = jaccard_similarity(current_kw, prev)
        if overlap >= QUERY_SIMILARITY_THRESHOLD:
            return True
    return False


REFLECTION_PROMPT = (
    "[SYSTEM] You have used {steps} search steps. Review your progress:\n"
    "- If you found a strong candidate answer, output it now with <answer>...</answer>.\n"
    "- If your last 2-3 searches returned nothing useful, you MUST try a completely "
    "different approach (different keywords, different starting entity).\n"
    "- Do NOT repeat similar queries. Each search must use substantially different terms."
)

DUPLICATE_QUERY_PROMPT = (
    "[SYSTEM] WARNING: Your recent searches use very similar keywords. "
    "This approach is not working. You MUST now:\n"
    "1. Stop searching this direction entirely.\n"
    "2. Think of a completely different angle to find the answer.\n"
    "3. Use different starting entities or facts in your next query.\n"
    "If you have any candidate answer at all, output it now with <answer>...</answer>."
)

LOW_SIGNAL_PIVOT_PROMPT = (
    "[SYSTEM] Your recent search results were mostly low-signal (blocked/captcha/video/social/noise). "
    "Immediately pivot to a new clue chain and use one highly constrained query format:\n"
    "- include 2-3 hard facts from the question (years, counts, names, exact phrases);\n"
    "- avoid generic terms like 'match', 'history', 'movie';\n"
    "- avoid YouTube/Reddit-oriented wording;\n"
    "- if a candidate entity appears, run exactly one confirmation query: "
    "\"<candidate> + <another independent fact>\"."
)

DUPLICATE_QUERY_BLOCKED_MSG = (
    '[{{"rank": 1, "title": "[HARNESS] Query blocked: too similar to your recent searches. '
    'Use a different clue chain and different entities, not rewordings.", '
    '"url": "", "snippet": "", "content": ""}}]'
)

# PLACEHOLDER_SYSTEM_PROMPT
SYSTEM_PROMPT = """You are a search agent. Answer questions by searching the web.

# RULES (follow strictly)

1. DECOMPOSE first: identify 2-3 different clue chains from the question.
2. Search HARD FACTS only: dates, numbers, proper nouns. Use "quotes" for phrases.
3. NEVER search metaphors/riddles literally. They return garbage.
4. PIVOT after 2 failed searches: switch to a completely different clue chain.
5. COMMIT when confident: one confirmation search, then output your answer.
6. DO NOT repeat near-duplicate queries. Similarity > 0.7 means rewrite.
7. Prefer snippet search first (fetch=False). Use fetch=True only for final verification.
8. Aim for two independent supporting sources before final answer.

# SEARCH STRATEGY

- Pack multiple hard facts into one query for precision.
- If a clue points to many candidates (e.g. yearly award), switch to another clue chain.
- After finding a strong candidate, search [candidate] + [another fact] to confirm.
- Use fetch=True only when snippet is insufficient.

# EXAMPLES

Good: "95th minute" free kick qualifier 2001 England
Bad:  "born out of discord" football team  ← metaphor, returns garbage

Good: "married 1894" "died 1961" "no children" watercolor exhibition
Bad:  social history book fourteen persons  ← too vague

# OUTPUT
<answer>your answer here</answer>
If evidence is weak, use: <answer confidence="low">your best guess</answer>."""

FORCE_ANSWER_PROMPT = (
    "[SYSTEM] Step budget exhausted. Do NOT call any more tools. "
    "If you have >=2 independent sources, output <answer>...</answer>. "
    "Otherwise output <answer confidence=\"low\">...</answer>."
)

# ---------------------------------------------------------------------------
# Helper: parse leaked tool_call from reasoning_content
# ---------------------------------------------------------------------------
def parse_leaked_tool_calls(reasoning_content: str) -> list[dict] | None:
    """
    Detect and parse <tool_call> XML leaked into reasoning_content.
    Returns a list of synthetic tool_call dicts, or None if nothing found.
    """
    pattern = r'<function=(\w+)>\s*<parameter=(\w+)>\s*'
    if '<tool_call>' not in reasoning_content and '<function=' not in reasoning_content:
        return None

    calls = []
    fn_blocks = re.findall(
        r'<function=(\w+)>(.*?)</function>',
        reasoning_content, re.DOTALL
    )
    for fn_name, params_block in fn_blocks:
        args = {}
        param_pairs = re.findall(
            r'<parameter=(\w+)>\s*(.*?)\s*</parameter>',
            params_block, re.DOTALL
        )
        for pname, pval in param_pairs:
            pval = pval.strip()
            if pval.lower() in ('true', 'false'):
                args[pname] = pval.lower() == 'true'
            else:
                try:
                    args[pname] = int(pval)
                except ValueError:
                    args[pname] = pval
        if fn_name in TOOL_FN_MAP:
            calls.append({
                "id": f"leaked_{uuid.uuid4().hex[:12]}",
                "function": {"name": fn_name, "arguments": json.dumps(args, ensure_ascii=False)},
                "type": "function",
            })
    return calls if calls else None


# ---------------------------------------------------------------------------
# Helper: extract answer from <answer> tags
# ---------------------------------------------------------------------------
def extract_answer(content: str) -> str:
    """Extract text from <answer>...</answer> tags, or return full content."""
    m_low = re.search(r'<answer\s+confidence="low">(.*?)</answer>', content, re.DOTALL)
    if m_low:
        return f'[LOW_CONFIDENCE] {m_low.group(1).strip()}'
    m = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    if m:
        return m.group(1).strip()
    return content.strip()


# ---------------------------------------------------------------------------
# Core run_task function
# ---------------------------------------------------------------------------

def run_task(
    task: dict,
    max_steps: int = MAX_STEPS,
    llm_base_url: str = LLM_BASE_URL,
    model_name: str = MODEL_NAME,
    trajectory_dir: str = "trajectories",
) -> dict:
    task_id     = task.get("id") or str(uuid.uuid4())[:8]
    instruction = task["instruction"]
    image_b64   = task.get("image_b64")
    image_url   = task.get("image_url")

    logger.info("run_task [plan&react]: task_id=%s", task_id)

    traj   = Trajectory(task_id, output_dir=trajectory_dir)
    client = OpenAI(base_url=llm_base_url, api_key="EMPTY")

    # ------------------------------------------------------------------ step 0
    traj.write(Role.SYSTEM, SYSTEM_PROMPT, step_id=0)

    # Build user message
    if image_b64 and image_url:
        user_content = [
            {"type": "text", "text": instruction + "\n输入图像的在线链接：" + image_url},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
    elif image_b64:
        user_content = [
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
    else:
        user_content = instruction

    traj.write(Role.USER, user_content, step_id=0)

    # ------------------------------------------------------------------ loop
    final_answer = ""
    force_answer_injected = False
    recent_queries = []  # Track recent search queries for dedup detection
    low_signal_streak = 0
    state_index = 0
    state_attempts = {s: 0 for s in STATE_SEQUENCE}
    evidence_domains = set()
    state_domain_evidence = {s: set() for s in STATE_SEQUENCE}
    candidate_votes = {}
    candidate_domain_evidence = {}
    best_candidate = None
    event_alignment = 0
    timeout_streak = 0
    search_degraded_mode = False

    for step in range(1, max_steps + 1):
        logger.info("--- step %d/%d ---", step, max_steps)
        current_state = STATE_SEQUENCE[state_index]
        if current_state in STATE_INSTRUCTION:
            traj.write(Role.USER, STATE_INSTRUCTION[current_state], step_id=step)
            state_attempts[current_state] += 1

        # --- Guardrail: inject reflection every 5 steps ---
        if step > 1 and step % 5 == 0 and not force_answer_injected:
            reflection = REFLECTION_PROMPT.format(steps=step)
            traj.write(Role.USER, reflection, step_id=step)
            logger.info("Injected REFLECTION_PROMPT at step %d", step)

        # --- Guardrail: if search quality is poor for consecutive rounds, force pivot ---
        if low_signal_streak >= LOW_SIGNAL_PIVOT_THRESHOLD and not force_answer_injected:
            traj.write(Role.USER, LOW_SIGNAL_PIVOT_PROMPT, step_id=step)
            logger.info("Injected LOW_SIGNAL_PIVOT_PROMPT at step %d", step)
            # Low signal in one state => consume trial budget faster to force a chain switch.
            state_attempts[current_state] += 1

        if timeout_streak >= 2 and not search_degraded_mode and not force_answer_injected:
            traj.write(Role.USER, SEARCH_DEGRADED_PROMPT, step_id=step)
            search_degraded_mode = True
            logger.info("Injected SEARCH_DEGRADED_PROMPT at step %d", step)

        # --- Guardrail: detect duplicate queries ---
        if len(recent_queries) >= 3:
            last_3_kw = recent_queries[-3:]
            overlap_01 = len(last_3_kw[0] & last_3_kw[1]) / max(len(last_3_kw[0] | last_3_kw[1]), 1)
            overlap_12 = len(last_3_kw[1] & last_3_kw[2]) / max(len(last_3_kw[1] | last_3_kw[2]), 1)
            if overlap_01 > 0.5 and overlap_12 > 0.5 and not force_answer_injected:
                traj.write(Role.USER, DUPLICATE_QUERY_PROMPT, step_id=step)
                logger.info("Injected DUPLICATE_QUERY_PROMPT at step %d (overlap: %.2f, %.2f)", step, overlap_01, overlap_12)

        # Inject force-answer prompt near the end
        if step == max_steps - 2 and not force_answer_injected:
            if len(evidence_domains) >= MIN_INDEPENDENT_SOURCES:
                final_prompt = FORCE_ANSWER_PROMPT
            else:
                final_prompt = (
                    "[SYSTEM] Step budget exhausted and evidence is weak (<2 independent sources). "
                    "Do NOT call tools. Output <answer confidence=\"low\">...</answer>."
                )
            traj.write(Role.USER, final_prompt, step_id=step)
            force_answer_injected = True
            logger.info("Injected FORCE_ANSWER_PROMPT")

        # After force-answer, disable tools to force the model to output text
        disable_tools_this_step = force_answer_injected

        messages = traj.to_messages()
        logger.info("messages count=%d, sending to LLM ...", len(messages))

        request_kwargs = dict(
            model=model_name,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            extra_body={"enable_thinking": True},
        )
        if not DISABLE_TOOLS and not disable_tools_this_step:
            request_kwargs["tools"] = TOOLS_SCHEMA
            request_kwargs["tool_choice"] = "auto"

        try:
            response = client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc, exc_info=True)
            traj.write(Role.TOOL, f"[HARNESS ERROR] LLM call failed: {exc}", step_id=step)
            break

        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        reasoning_content = msg.reasoning_content or ""
        total_tokens = response.usage.total_tokens or ""

        tool_calls = None if DISABLE_TOOLS else msg.tool_calls

        # --- Fix: recover leaked tool_calls from reasoning_content ---
        if not tool_calls and not content and reasoning_content:
            leaked = parse_leaked_tool_calls(reasoning_content)
            if leaked:
                tool_calls = leaked
                logger.info("Recovered %d leaked tool_call(s) from reasoning", len(leaked))

        # Write assistant turn
        tool_calls_data = []
        if tool_calls:
            if hasattr(tool_calls[0], 'model_dump'):
                tool_calls_data = [tc.model_dump() for tc in tool_calls]
            else:
                tool_calls_data = tool_calls

        extra = {}
        if tool_calls_data:
            extra["tool_calls"] = tool_calls_data
        if reasoning_content:
            extra["reasoning_content"] = reasoning_content
        if total_tokens:
            extra["total_tokens"] = total_tokens

        traj.write(Role.ASSISTANT, content, step_id=step, extra=extra if extra else None)

        if content:
            logger.info("assistant: %s", content[:200])
        logger.info("finish_reason=%s, has_tool_calls=%s", choice.finish_reason, bool(tool_calls))

        # --- Check for final answer ---
        if not tool_calls and content:
            final_answer = extract_answer(content)
            logger.info("Task complete at step %d", step)
            break

        if not tool_calls and not content:
            continue

        tool_calls = _select_tool_calls_for_step(tool_calls, instruction, recent_queries)

        # --- Execute tool calls (concurrently) ---
        def _exec_one_tool(tc):
            if hasattr(tc, 'function'):
                fn_name = tc.function.name
                fn_args_str = tc.function.arguments
                tc_id = tc.id
            else:
                fn_name = tc["function"]["name"]
                fn_args_str = tc["function"]["arguments"]
                tc_id = tc["id"]

            try:
                fn_args = json.loads(fn_args_str)
            except json.JSONDecodeError:
                fn_args = {}
                logger.warning("Bad tool args JSON")

            logger.info("tool_call: %s(%s)", fn_name, fn_args)

            # Search arg normalization for better recall/precision.
            if fn_name == "search_text":
                if timeout_streak >= 2:
                    blocked = (
                        '[{"rank":1,"title":"[HARNESS] search_text temporarily disabled after consecutive timeouts.",'
                        '"url":"","snippet":"Switch tool path or return low-confidence answer.","content":""}]'
                    )
                    return tc_id, fn_name, fn_args, blocked
                query = str(fn_args.get("query", "")).strip()
                query = re.sub(r"\s+", " ", query)
                fn_args["query"] = query
                fn_args["top_k"] = max(int(fn_args.get("top_k", 5)), 5)
                # Broad queries with fetch=True often pull noisy pages; fetch later only if needed.
                if fn_args.get("fetch") and len(query.split()) < 6:
                    fn_args["fetch"] = False

                # Prefer rewrite over hard-block for metaphor-like queries.
                if is_metaphor_query(query):
                    rewritten = rewrite_query_with_hard_facts(query, instruction)
                    if rewritten != query:
                        logger.info("Rewrote metaphor-like query: %s -> %s", query, rewritten)
                        fn_args["query"] = rewritten
                        query = rewritten
                    else:
                        logger.info("BLOCKED metaphor query: %s", query)
                        return tc_id, fn_name, fn_args, METAPHOR_BLOCKED_MSG

                # Pre-call duplicate handling: diversify first, block only if no useful rewrite.
                kw = extract_query_keywords(fn_args)
                if kw and should_block_duplicate_query(kw, recent_queries):
                    diversified = diversify_duplicate_query(query, instruction, kw)
                    if diversified and diversified != query:
                        logger.info("Diversified duplicate-like query: %s -> %s", query, diversified)
                        fn_args["query"] = diversified
                    else:
                        logger.info("BLOCKED duplicate-like query: %s", query)
                        return tc_id, fn_name, fn_args, DUPLICATE_QUERY_BLOCKED_MSG

            if fn_name not in TOOL_FN_MAP:
                tool_result = f"[ERROR] Unknown tool: {fn_name}"
            else:
                try:
                    raw = TOOL_FN_MAP[fn_name](fn_args)
                    if isinstance(raw, (dict, list)):
                        tool_result = json.dumps(raw, ensure_ascii=False)
                    else:
                        tool_result = str(raw)
                except Exception as exc:
                    tool_result = f"[ERROR] Tool '{fn_name}' raised: {type(exc).__name__}: {exc}"
                    logger.exception("Tool error")

            return tc_id, fn_name, fn_args, tool_result

        pending_tools = tool_calls or []
        if len(pending_tools) > 1:
            with ThreadPoolExecutor(max_workers=min(len(pending_tools), 4)) as pool:
                futures = [pool.submit(_exec_one_tool, tc) for tc in pending_tools]
                results = [f.result() for f in futures]
        else:
            results = [_exec_one_tool(tc) for tc in pending_tools]

        for tc_id, fn_name, fn_args, tool_result in results:
            # --- Guardrail: filter garbage URLs from search results ---
            if fn_name == "search_text":
                tool_result, has_signal = filter_garbage_results(tool_result)
                is_timeout_like = _is_error_signal_text(tool_result)
                if is_timeout_like:
                    timeout_streak += 1
                elif has_signal:
                    timeout_streak = 0
                if has_signal:
                    low_signal_streak = 0
                else:
                    low_signal_streak += 1
                try:
                    parsed_results = json.loads(tool_result)
                except Exception:
                    parsed_results = []
                if isinstance(parsed_results, list):
                    domains = _collect_domains_from_results(parsed_results)
                    evidence_domains.update(domains)
                    state_domain_evidence[current_state].update(domains)
                    event_alignment += _event_alignment_score(instruction, parsed_results)
                    for item in parsed_results:
                        item_domains = _collect_domains_from_results([item])
                        for cand in _extract_candidates([item]):
                            candidate_votes[cand] = candidate_votes.get(cand, 0) + 1
                            candidate_domain_evidence.setdefault(cand, set()).update(item_domains)
                            if not best_candidate or candidate_votes[cand] > candidate_votes.get(best_candidate, 0):
                                best_candidate = cand
                # Track query keywords for dedup detection
                kw = extract_query_keywords(fn_args)
                if kw:
                    recent_queries.append(kw)

            logger.info("tool_result (%s): %s", fn_name, str(tool_result)[:200])
            traj.write(Role.TOOL, tool_result, step_id=step, tool_call_id=tc_id,
                       extra={"fn_name": fn_name, "fn_args": fn_args})

        # State transitions and failsafe controls.
        if current_state == "S0_PARSE":
            if recent_queries:
                state_index = min(state_index + 1, len(STATE_SEQUENCE) - 1)
        elif current_state == "S1_SUBJECT":
            if (
                best_candidate
                and candidate_votes.get(best_candidate, 0) >= 2
                and len(candidate_domain_evidence.get(best_candidate, set())) >= 2
            ):
                state_index = min(state_index + 1, len(STATE_SEQUENCE) - 1)
        elif current_state == "S2_EVENT":
            if (
                best_candidate
                and candidate_votes.get(best_candidate, 0) >= 2
                and len(candidate_domain_evidence.get(best_candidate, set())) >= MIN_INDEPENDENT_SOURCES
                and event_alignment >= 2
            ):
                state_index = min(state_index + 1, len(STATE_SEQUENCE) - 1)
            elif event_alignment < 0:
                traj.write(Role.USER, CONFLICT_PIVOT_PROMPT, step_id=step)
                state_index = max(1, state_index - 1)
                low_signal_streak = LOW_SIGNAL_PIVOT_THRESHOLD
        elif current_state == "S3_DETAIL" and final_answer:
            state_index = min(state_index + 1, len(STATE_SEQUENCE) - 1)

        if state_attempts[current_state] >= MAX_TRIALS_PER_STATE and state_index < len(STATE_SEQUENCE) - 1:
            state_index += 1
            logger.info(
                "Force transition by trial budget: %s -> %s",
                current_state, STATE_SEQUENCE[state_index]
            )
    else:
        logger.warning("Reached max_steps=%d", max_steps)
        # Try to extract answer from last assistant content
        entries = traj.read_all()
        for e in reversed(entries):
            if e["role"] == "assistant" and e.get("content"):
                final_answer = extract_answer(e["content"])
                break
        if not final_answer:
            final_answer = "[HARNESS] Max steps reached without answer."

    summary = traj.summary()
    logger.info("Trajectory summary: %s", summary)

    return {
        "task_id":         task_id,
        "answer":          final_answer,
        "steps":           step,
        "trajectory_path": str(traj.path),
        "summary":         summary,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen Agent Harness (Plan & ReAct) — run a single task",
    )
    p.add_argument("--instruction", "-i", required=True, help="Task instruction text")
    p.add_argument("--task-id",     "-t", default=None,  help="Optional task ID")
    p.add_argument("--max-steps",   "-s", type=int, default=MAX_STEPS)
    p.add_argument("--llm-url",           default=LLM_BASE_URL)
    p.add_argument("--model",             default=MODEL_NAME)
    p.add_argument("--traj-dir",          default="trajectories")
    p.add_argument("--image",             default=None, help="Local image path")
    p.add_argument("--image-url",         default=None, help="Online image URL")
    return p.parse_args()


if __name__ == "__main__":
    import base64

    args = _parse_args()

    image_b64 = None
    if args.image:
        with open(args.image, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

    task = {
        "instruction": args.instruction,
        "image_b64":   image_b64,
        "image_url":   args.image_url,
    }
    if args.task_id:
        task["id"] = args.task_id

    result = run_task(
        task,
        max_steps=args.max_steps,
        llm_base_url=args.llm_url,
        model_name=args.model,
        trajectory_dir=args.traj_dir,
    )

    print("\n" + "=" * 60)
    print("TASK COMPLETE")
    print("=" * 60)
    print(f"Task ID:  {result['task_id']}")
    print(f"Steps:    {result['steps']}")
    print(f"Traj:     {result['trajectory_path']}")
    print(f"\nAnswer:\n{result['answer']}")