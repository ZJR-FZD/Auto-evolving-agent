"""
Qwen Agent Harness — Main Orchestrator
======================================

Drives the agent loop:
  1. Writes system + user turn to trajectory
  2. Calls Qwen-3.5 via Sglang OpenAI-compat API
  3. Dispatches tool calls and records results
  4. Loops until finish_reason == 'stop' or max_steps reached

Usage (CLI):
    python -m task_runner \
        --instruction "请帮我查询上海创智学院谢源老师的相关信息，并获取其代表作。" \
        --task-id my_task_010
    python -m task_runner \
        --instruction "请先帮我分析图像的内容，再调用search_image工具进行图像搜索。" \
        --image "/inspire/qb-ilm2/project/26summer-camp-01/qiaojingyang-240208120192/harness-sii/datasets/simpleVQA/CCSimpleQA/0.jpg" \
        --image-url "https://datasets-server.huggingface.co/cached-assets/ohjoonhee/SimpleVQA/--/8fefe22e2775a6ac0a73ac22edba8a01536b8a59/--/default/test/0/image/image.jpg?Expires=1779081093&Signature=cHN23HVLSGpna8jlbFRnpt90RruGsgAjpRTot1IArVYgZrUFTz2Fl5Gn7OSU6QVmxQMZFc8csXss9g9-8sh9fAPpRbOAwgdlVdH8yg1fr4pIGLneUXz8swhhSlSECAbYyDi-r2we7kizYjnuvlfDa45BsRU32c7sPVLttqVWbNH8vWrYi9rTajYAdbCn9l2zYMN~zpSp~8b4T2OwMGw6feZl3fBdZxMPWmuyf2GTaIAiisDTQd2b6-8Yq3CsIzjfmW6M4nN0T5O8FXLR-yTd5ve9Pj40U13410vyqUbcOGDC~R7hCtrXDhxpg4aivRPLcjcHPTbKgu10K09cWSTZAQ__&Key-Pair-Id=K204OQ5RWQVDLD" \
        --task-id my_task_011
"""

import argparse
import base64
import json
import logging
import os
import re
import time
import uuid
from typing import Optional
from pathlib import Path

from openai import OpenAI

from agent_memory import EvolutionMemory
from reflection import (
    ReflectionMonitor,
    build_query_tags,
    compact_tool_result,
    heuristic_lessons,
    judge_with_gold,
    summarize_entries,
)
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
logger = logging.getLogger("harness.task_runner")

# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_QWEN_ANSWER_RE = re.compile(r"<function=answer>\s*(.*?)\s*</(?:parameter|function)", re.DOTALL)


_TAG_ARTIFACT_RE = re.compile(r"</?(parameter|function|tool_call|tool_response)\b[^>]*>", re.IGNORECASE)


def extract_answer(content: str) -> str:
    """Extract clean answer from assistant content."""
    if not content:
        return ""
    content = _THINK_RE.sub("", content).strip()
    match = _ANSWER_TAG_RE.search(content)
    if match:
        answer = match.group(1).strip()
        answer = _TAG_ARTIFACT_RE.sub("", answer).strip()
        return answer
    # Reject raw tool call content as answer
    if content.lstrip().startswith(("<tool_call>", "<function=")):
        return ""
    return content


# ---------------------------------------------------------------------------
# Tool call leakage detection in reasoning_content
# ---------------------------------------------------------------------------
_TOOL_CALL_PATTERNS = [
    re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL),
    re.compile(r'"name"\s*:\s*"(search_text|search_image|browser_navigate|browser_get_text|browser_click|browser_type|browser_parallel)".*?"arguments"\s*:\s*(\{[^}]*\})', re.DOTALL),
]
_QWEN_FUNC_RE = re.compile(
    r'<(?:tool_call|function).*?<function=(search_text|search_image|browser_navigate|browser_get_text|browser_click|browser_type|browser_parallel)>\s*(.*?)\s*</function>',
    re.DOTALL,
)
_QWEN_PARAM_RE = re.compile(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', re.DOTALL)


def detect_leaked_tool_calls(reasoning_content: str) -> list[dict]:
    """Detect tool calls leaked into reasoning/thinking content."""
    if not reasoning_content:
        return []
    leaked = []
    for pattern in _TOOL_CALL_PATTERNS:
        for match in pattern.finditer(reasoning_content):
            try:
                if match.lastindex == 1:
                    data = json.loads(match.group(1))
                    if isinstance(data, dict) and "name" in data:
                        fn_name = data["name"]
                        fn_args = data.get("arguments") or data.get("parameters") or {}
                        if isinstance(fn_args, str):
                            fn_args = json.loads(fn_args)
                        leaked.append({"name": fn_name, "arguments": fn_args})
                elif match.lastindex == 2:
                    fn_name = match.group(1)
                    fn_args = json.loads(match.group(2))
                    leaked.append({"name": fn_name, "arguments": fn_args})
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    # Qwen-style: <function=name><parameter=key>value</parameter>...</function>
    for match in _QWEN_FUNC_RE.finditer(reasoning_content):
        fn_name = match.group(1)
        params_block = match.group(2)
        fn_args = {}
        for pm in _QWEN_PARAM_RE.finditer(params_block):
            key = pm.group(1)
            val = pm.group(2).strip()
            if val.lower() in ("true", "false"):
                fn_args[key] = val.lower() == "true"
            else:
                try:
                    fn_args[key] = int(val)
                except ValueError:
                    fn_args[key] = val
        if fn_name in TOOL_FN_MAP:
            leaked.append({"name": fn_name, "arguments": fn_args})
    return leaked


# ---------------------------------------------------------------------------
# Context compression for long conversations
# ---------------------------------------------------------------------------
COMPRESS_AFTER_STEP = int(os.getenv("COMPRESS_AFTER_STEP", "5"))
COMPRESSED_TOOL_CHARS = int(os.getenv("COMPRESSED_TOOL_CHARS", "300"))
COMPRESSED_TOOL_CHARS_OLD = int(os.getenv("COMPRESSED_TOOL_CHARS_OLD", "150"))


def compress_messages(messages: list[dict], current_step: int) -> list[dict]:
    """Progressively compress old tool results to save tokens."""
    if current_step < COMPRESS_AFTER_STEP:
        return messages
    compressed = []
    n = len(messages)
    for i, msg in enumerate(messages):
        if msg["role"] == "tool" and i < n - 6:
            content = msg.get("content", "")
            if isinstance(content, str):
                limit = COMPRESSED_TOOL_CHARS_OLD if i < n - 12 else COMPRESSED_TOOL_CHARS
                if len(content) > limit:
                    msg = dict(msg)
                    msg["content"] = content[:limit] + "\n...[已压缩]"
        compressed.append(msg)
    return compressed

# ---------------------------------------------------------------------------
# LLM connection (Sglang OpenAI-compat, two nodes behind Nginx)
# ---------------------------------------------------------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "qwen-3.5")
MAX_STEPS    = int(os.getenv("MAX_STEPS", "20"))
MAX_TOKENS   = int(os.getenv("MAX_TOKENS", "16000"))

# 调试开关：True = 不向 LLM 注册 tools，纯文本对话，便于先验证 LLM 通路
# 工具实现接好后默认关闭；如需调试 LLM 通路，export DISABLE_TOOLS=1
DISABLE_TOOLS = os.getenv("DISABLE_TOOLS", "0") == "1"

# Evolution / reflection knobs.  They are deliberately environment-driven so
# baseline and evolved runs can be compared by changing run commands only.
ENABLE_MEMORY_DEFAULT = os.getenv("ENABLE_MEMORY", "1") != "0"
ENABLE_REFLECTION_DEFAULT = os.getenv("ENABLE_REFLECTION", "1") != "0"
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "4"))
MEMORY_MAX_CHARS = int(os.getenv("MEMORY_MAX_CHARS", "1800"))
MEMORY_UPDATE_MODE = os.getenv("MEMORY_UPDATE_MODE", "heuristic").lower()
MAX_TOOL_RESULT_CHARS = int(os.getenv("MAX_TOOL_RESULT_CHARS", "5000"))
REFLECTION_REPEAT_THRESHOLD = int(os.getenv("REFLECTION_REPEAT_THRESHOLD", "2"))
REFLECTION_ERROR_THRESHOLD = int(os.getenv("REFLECTION_ERROR_THRESHOLD", "2"))
REFLECTION_MAX_HINTS = int(os.getenv("REFLECTION_MAX_HINTS", "3"))
DEBUG_LLM_MESSAGES = os.getenv("DEBUG_LLM_MESSAGES", "0") == "1"
FORCE_ANSWER_STEPS_BEFORE_END = int(os.getenv("FORCE_ANSWER_STEPS_BEFORE_END", "2"))
ENABLE_CONFIDENCE_RETRY = os.getenv("ENABLE_CONFIDENCE_RETRY", "0") == "1"
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))

# ---------------------------------------------------------------------------
# Tool schema (OpenAI function-calling format)
# ---------------------------------------------------------------------------
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": (
                "基于 Serper (Google) 的联网文字搜索，并用 Jina Reader 抽取每个结果页面的正文"
                "返回 [{rank,title,url,snippet,content}]。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":     {"type": "string",  "description": "搜索关键词"},
                    "top_k":     {"type": "integer", "description": "返回条数（1-3）", "default": 1},
                    "fetch":     {"type": "boolean", "description": "是否抓取正文，false 时只返回摘要", "default": True},
                    "max_chars": {"type": "integer", "description": "每篇正文截断的最大字符数", "default": 500},
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
                "图搜文：基于 Google Lens (Serper /lens) 的反向图像搜索，并用 "
                "Jina Reader 抽取结果页面正文。输入必须是 http(s) 图片 URL 。"
                "返回 [{rank,title,url,snippet,content}]。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string",  "description": "图片的 http(s) URL"},
                    "top_k":     {"type": "integer", "description": "返回条数（1-3）", "default": 1},
                    "fetch":     {"type": "boolean", "description": "是否抓取正文", "default": True},
                    "max_chars": {"type": "integer", "description": "每篇正文截断的最大字符数", "default": 500},
                },
                "required": ["image_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": (
                "在沙盒浏览器中打开一个 URL。默认顺带返回前若干字符的页面文本预览，"
                "需要完整正文请再调 browser_get_text。返回 "
                "{ok,url,title,wait_until,text_preview?,truncated?}。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url":          {"type": "string",  "description": "要访问的 URL（可省略协议头）"},
                    "wait_until":   {"type": "string",  "description": "Playwright 等待策略",
                                     "enum": ["domcontentloaded", "load", "networkidle"],
                                     "default": "domcontentloaded"},
                    "include_text": {"type": "boolean", "description": "是否返回 text_preview", "default": True},
                    "max_text":     {"type": "integer", "description": "text_preview 字符上限", "default": 2000},
                    "timeout":      {"type": "integer", "description": "导航超时秒数", "default": 30},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_get_text",
            "description": "返回当前页面清洗后的可见文本。返回 {ok,url,title,text,truncated,total_chars}。",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_chars": {"type": "integer", "description": "正文最大字符数", "default": 5000},
                    "timeout":   {"type": "integer", "description": "抽取超时秒数", "default": 15},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": (
                "用 CSS 选择器点击当前页的元素。selector 接受任意合法 CSS，例如 "
                "'#login', 'button.primary', \"button:has-text('确定')\"。返回 "
                "{ok,selector,current_url,current_title,navigated}。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string",  "description": "CSS 选择器"},
                    "nth":      {"type": "integer", "description": "命中多个时取第几个（0 表示用 .first）", "default": 0},
                    "timeout":  {"type": "integer", "description": "点击超时秒数", "default": 10},
                },
                "required": ["selector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_type",
            "description": (
                "向一个 CSS 选择器选中的输入框键入文本，可选按回车提交。"
                "返回 {ok,selector,submitted,current_url,current_title}。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string",  "description": "CSS 选择器（输入框）"},
                    "text":     {"type": "string",  "description": "要输入的文本"},
                    "submit":   {"type": "boolean", "description": "输入完是否按 Enter", "default": False},
                    "clear":    {"type": "boolean", "description": "输入前是否清空字段", "default": True},
                    "timeout":  {"type": "integer", "description": "操作超时秒数", "default": 10},
                },
                "required": ["selector", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_parallel",
            "description": (
                "在沙盒浏览器中**并发**打开多个 URL。"
                "mode='navigate' 每个返回 {url,title,text_preview,truncated}；"
                "mode='get_text' 每个返回 {url,title,text,truncated,total_chars}。"
                "返回值是一个列表，单个 URL 失败不影响其他。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "urls":            {"type": "array", "items": {"type": "string"}, "description": "URL 列表"},
                    "mode":            {"type": "string", "enum": ["navigate", "get_text"], "default": "navigate"},
                    "max_chars":       {"type": "integer", "description": "每条结果文本上限；缺省时 navigate=2000，get_text=5000"},
                    "wait_until":      {"type": "string",
                                        "enum": ["domcontentloaded", "load", "networkidle"],
                                        "default": "domcontentloaded"},
                    "max_concurrency": {"type": "integer", "description": "同时打开的标签页数（1-8）", "default": 4},
                    "timeout":         {"type": "integer", "description": "单页超时秒数", "default": 30},
                },
                "required": ["urls"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool function dispatch map
# ---------------------------------------------------------------------------
def _call_search_image(args: dict) -> list[dict]:
    """Accept both schema names: image_url (tool schema) and image (tool impl)."""
    normalized = dict(args or {})
    if "image" not in normalized and "image_url" in normalized:
        normalized["image"] = normalized.pop("image_url")
    return search_image(**normalized)


TOOL_FN_MAP = {
    "search_text":      lambda a: search_text(**a),
    "search_image":     _call_search_image,
    "browser_navigate": lambda a: browser_navigate(**a),
    "browser_get_text": lambda a: browser_get_text(**a),
    "browser_click":    lambda a: browser_click(**a),
    "browser_type":     lambda a: browser_type(**a),
    "browser_parallel": lambda a: browser_parallel(**a),
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一个高效、严谨的任务执行 Agent，运行在配备多工具的自动化框架中。

## 核心原则
1. 每一步先在 <think>...</think> 中简述推理（不超过3句），再决定调用工具或直接回答。
2. 任务完成后用 <answer>答案</answer> 包裹最终答案，无需再调用工具。
3. 工具返回 ok=False 时，分析 error，最多重试 1 次；仍失败则换工具或方法。
4. search_image 必须使用图像的在线 URL。
5. 每一步要么调用工具，要么输出最终答案，不能两者都不做。
6. 禁止用完全相同的参数重复调用同一工具。

## 高效推理策略
- 收到问题后立即判断类型，规划最短路径（通常2-3步即可）。
- 每次工具返回后立即评估：证据是否足够？足够则直接输出答案。
- 搜索时使用精确关键词（实体名+属性词），不要搜索整句问题。
- 第一次搜索结果已包含答案时，立即停止，不要追求更多证据。
- SEC.gov等政府网站无法直接访问，用搜索引擎获取摘要。

## 任务策略
- SimpleVQA：(1)识别图中主体→(2)search_text查询属性→(3)输出短答案。主体不确定时先search_image。
- 2Wiki：(1)识别两个实体→(2)分别查询各自属性→(3)比较/组合得出答案。比较题必须分别查两个实体的同一属性。
- 通用：先搜索最关键的事实，得到后立即作答。

## 工具使用效率
- search_text: 用精确关键词，top_k=1 通常足够，fetch=True 获取正文。
- search_image: 用于识别图中未知实体，只需调用一次。
- browser_navigate: 仅在搜索摘要不够时使用，优先用搜索。
- browser_parallel: 需要对比多个页面时使用，减少轮数。

## 答案格式
- 最终答案必须简短、直接，只包含题目要求的信息（人名、地名、数字、年份等）。
- 用 <answer>你的答案</answer> 包裹。
- 不要在答案中包含解释、推理过程或多余文字。
"""


def _build_system_prompt(memory_context: str = "") -> str:
    if not memory_context:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\n\n## 长期记忆\n{memory_context}\n"


# ---------------------------------------------------------------------------
# Core run_task function
# ---------------------------------------------------------------------------

def _image_data_url(image_b64: str) -> str:
    mime = "image/jpeg"
    try:
        raw = base64.b64decode(image_b64, validate=False)
    except Exception:  # noqa: BLE001
        raw = b""

    if raw.startswith(b"\x89PNG"):
        mime = "image/png"
    elif raw.startswith(b"GIF8"):
        mime = "image/gif"
    elif raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        mime = "image/webp"
    elif raw.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"

    return f"data:{mime};base64,{image_b64}"


def _task_bool(task: dict, key: str, default: bool) -> bool:
    if key not in task:
        return default
    value = task.get(key)
    if isinstance(value, str):
        return value.lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _task_family(task: dict) -> str:
    return str(task.get("task_family") or task.get("dataset") or "generic").lower()


def _task_metadata(task: dict) -> dict:
    meta = task.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _memory_for_task(task: dict) -> EvolutionMemory:
    memory_dir = task.get("memory_dir") or os.getenv("AGENT_MEMORY_DIR", "")
    return EvolutionMemory(memory_dir or None)


def _recall_memory(task: dict, instruction: str, short_term_memory=None) -> tuple[str, list[str]]:
    if not _task_bool(task, "enable_memory", ENABLE_MEMORY_DEFAULT):
        return "", []
    store = _memory_for_task(task)
    family = _task_family(task)
    metadata = _task_metadata(task)
    query = " ".join(
        [
            instruction,
            json.dumps(metadata, ensure_ascii=False, sort_keys=True)[:1000],
        ]
    )
    recalled = store.recall(
        query,
        task_family=family,
        tags=build_query_tags(family, metadata),
        top_k=MEMORY_TOP_K,
    )
    ids = [entry.id for entry, _ in recalled]
    if ids:
        store.mark_recalled(ids)
    stm_entries = None
    if short_term_memory is not None:
        stm_entries = short_term_memory.recall(task_family=family, top_k=3)
    return EvolutionMemory.format_for_prompt(
        recalled, max_chars=MEMORY_MAX_CHARS, short_term=stm_entries
    ), ids


def _post_task_memory_update(
    *,
    task: dict,
    instruction: str,
    final_answer: str,
    signals,
    stats: dict,
    traj=None,
) -> list[str]:
    if not _task_bool(task, "enable_memory", ENABLE_MEMORY_DEFAULT):
        return []

    mode = str(task.get("memory_update_mode") or MEMORY_UPDATE_MODE or "heuristic").lower()
    if mode in {"0", "false", "off", "none", "disabled"}:
        return []

    gold_answer = str(task.get("gold_answer") or "")
    allow_gold = _task_bool(task, "allow_gold_feedback", False) or mode == "gold"
    correct = None
    if allow_gold and gold_answer:
        correct = judge_with_gold(final_answer, gold_answer)

    # In heuristic mode, avoid writing noisy "success" memories without labels.
    if correct is None and not signals.has_failure_signal():
        return []

    # Try LLM-based lesson extraction first (higher quality)
    lessons = []
    if signals.has_failure_signal() or correct is False:
        try:
            from llm_judge import extract_lessons_llm, JUDGE_ENABLED
            if JUDGE_ENABLED and traj:
                traj_summary = summarize_entries(traj.read_all(), max_chars=3000)
                llm_lessons = extract_lessons_llm(
                    question=instruction,
                    trajectory_summary=traj_summary,
                    final_answer=final_answer,
                    task_family=_task_family(task),
                    correct=correct,
                )
                for ll in llm_lessons:
                    lessons.append({
                        "task_family": _task_family(task),
                        "category": ll.get("category", "strategy"),
                        "outcome": "failure" if correct is False else "unknown",
                        "lesson": ll.get("lesson", ""),
                        "strategy": ll.get("strategy", ""),
                        "avoid": ll.get("avoid", ""),
                        "tags": build_query_tags(_task_family(task), _task_metadata(task)),
                        "confidence": float(ll.get("confidence", 0.7)),
                    })
        except Exception as exc:
            logger.debug("LLM lesson extraction failed: %s", exc)

    # Fallback to heuristic lessons
    if not lessons:
        lessons = heuristic_lessons(
            task_family=_task_family(task),
            instruction=instruction,
            metadata=_task_metadata(task),
            final_answer=final_answer,
            signals=signals,
            correct=correct,
        )
    if not lessons:
        return []

    store = _memory_for_task(task)
    written: list[str] = []
    for lesson in lessons:
        metadata = {
            "stats": {
                "steps": stats.get("steps"),
                "tool_calls": stats.get("tool_calls"),
                "tool_errors": stats.get("tool_errors"),
                "total_tokens": stats.get("total_tokens"),
            },
            "feedback_mode": mode,
            "correct": correct,
        }
        memory_id = store.add(
            task_family=lesson["task_family"],
            category=lesson["category"],
            outcome=lesson["outcome"],
            lesson=lesson["lesson"],
            strategy=lesson.get("strategy", ""),
            avoid=lesson.get("avoid", ""),
            tags=lesson.get("tags", []),
            confidence=float(lesson.get("confidence", 0.55)),
            source_task_id=str(task.get("id") or ""),
            source="gold" if correct is not None else "heuristic",
            metadata=metadata,
        )
        if memory_id:
            written.append(memory_id)
    return written


def run_task(
    task: dict,
    max_steps: int = MAX_STEPS,
    llm_base_url: str = LLM_BASE_URL,
    model_name: str = MODEL_NAME,
    trajectory_dir: str = "trajectories",
    short_term_memory=None,
) -> dict:
    """
    Execute a task with the Qwen agent loop.

    Args:
        task:            Dict with keys:
                           - "instruction" (str, required): task description
                           - "id"          (str, optional): task identifier
                           - "image_b64"   (str, optional): base64 image for vision input
                           - "image_url"   (str, optional): online image url for vision input
        max_steps:       Maximum agent loop iterations.
        llm_base_url:    Sglang / OpenAI-compat endpoint.
        model_name:      Model identifier served by Sglang.
        trajectory_dir:  Directory to write JSONL trajectories.
        short_term_memory: Optional ShortTermMemory for cross-task learning within batch.

    Returns:
        Dict with keys: task_id, answer, steps, trajectory_path, summary
    """
    task_id     = task.get("id") or str(uuid.uuid4())[:8]
    instruction = task["instruction"]
    image_b64   = task.get("image_b64")
    image_url   = task.get("image_url")

    logger.info("run_task: task_id=%s", task_id)

    traj   = Trajectory(task_id, output_dir=trajectory_dir)
    client = OpenAI(base_url=llm_base_url, api_key="EMPTY")
    started_at = time.time()
    memory_context, recalled_memory_ids = _recall_memory(task, instruction, short_term_memory)
    monitor = ReflectionMonitor(
        repeat_threshold=REFLECTION_REPEAT_THRESHOLD,
        error_threshold=REFLECTION_ERROR_THRESHOLD,
        max_hints=REFLECTION_MAX_HINTS,
        task_family=_task_family(task),
    )
    stats = {
        "llm_calls": 0,
        "tool_calls": 0,
        "tool_errors": 0,
        "repeated_tool_calls": 0,
        "reflection_hints": 0,
        "memory_recalled": len(recalled_memory_ids),
        "memory_written": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tool_result_truncated": 0,
    }

    # ------------------------------------------------------------------ step 0
    # Write system turn
    traj.write(
        Role.SYSTEM,
        _build_system_prompt(memory_context),
        step_id=0,
        extra={"memory_ids": recalled_memory_ids} if recalled_memory_ids else None,
    )

    # Build user message (optionally include image)
    user_text = instruction
    if image_url:
        user_text = f"{instruction}\n输入图像的在线链接：{image_url}"

    if image_b64:
        user_content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": _image_data_url(image_b64)}},
        ]
    elif image_url:
        user_content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    else:
        user_content = instruction

    traj.write(Role.USER, user_content, step_id=0)

    # ------------------------------------------------------------------ loop
    final_answer = ""
    last_assistant_content = ""
    max_steps_reached = False

    for step in range(1, max_steps + 1):
        logger.info("--- step %d ---", step)

        messages = traj.to_messages()
        messages = compress_messages(messages, step)

        # Progressive force-answer injection
        steps_remaining = max_steps - step
        if steps_remaining <= FORCE_ANSWER_STEPS_BEFORE_END and not DISABLE_TOOLS:
            if steps_remaining <= 0:
                force_msg = (
                    "HARNESS强制：这是最后一步（第%d步/共%d步）。"
                    "你必须立即输出最终答案，禁止调用任何工具。"
                    "用 <answer>你的答案</answer> 包裹。" % (step, max_steps)
                )
            else:
                force_msg = (
                    "HARNESS提示：剩余%d步（当前第%d步/共%d步）。"
                    "请优先基于已有证据输出答案。如果证据不足，最多再调用1次工具后必须作答。"
                    "用 <answer>你的答案</answer> 包裹最终答案。" % (steps_remaining, step, max_steps)
                )
            traj.write(Role.USER, force_msg, step_id=step, extra={"force_answer": True})
            messages.append({"role": "user", "content": force_msg})

        logger.info("messages count=%d, sending to LLM ...", len(messages))

        request_kwargs = dict(
            model=model_name,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=1.0,
            extra_body={"enable_thinking": True},
        )
        if not DISABLE_TOOLS:
            if steps_remaining <= 1:
                request_kwargs["tools"] = TOOLS_SCHEMA
                request_kwargs["tool_choice"] = "none"
            else:
                request_kwargs["tools"] = TOOLS_SCHEMA
                request_kwargs["tool_choice"] = "auto"

        try:
            response = client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc, exc_info=True)
            traj.write(
                Role.TOOL,
                f"[HARNESS ERROR] LLM call failed at step {step}: {exc}",
                step_id=step,
            )
            break

        choice  = response.choices[0]
        msg     = choice.message
        content = msg.content or ""
        if content:
            last_assistant_content = content
        if DEBUG_LLM_MESSAGES:
            print(msg)
        reasoning_content = getattr(msg, "reasoning_content", "") or ""
        usage = getattr(response, "usage", None)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        stats["llm_calls"] += 1
        stats["total_tokens"] += total_tokens
        stats["prompt_tokens"] += prompt_tokens
        stats["completion_tokens"] += completion_tokens

        # Tool call leakage detection in reasoning_content
        tool_calls = None if DISABLE_TOOLS else msg.tool_calls
        leaked_calls = []
        if not tool_calls and reasoning_content and step < max_steps - 1:
            leaked_calls = detect_leaked_tool_calls(reasoning_content)
            if leaked_calls:
                logger.info("Detected %d leaked tool call(s) in reasoning", len(leaked_calls))
                stats.setdefault("leaked_tool_calls", 0)
                stats["leaked_tool_calls"] += len(leaked_calls)

        monitor.record_assistant(content, bool(tool_calls or leaked_calls))

        # Write assistant turn
        tool_calls_data = (
            [tc.model_dump() for tc in tool_calls]
            if tool_calls else []
        )

        extra = {}
        if tool_calls_data:
            extra["tool_calls"] = tool_calls_data
        if reasoning_content:
            extra["reasoning_content"] = reasoning_content
        if total_tokens:
            extra["total_tokens"] = total_tokens
        if leaked_calls:
            extra["leaked_tool_calls"] = leaked_calls

        traj.write(
            Role.ASSISTANT,
            content,
            step_id=step,
            extra=extra if extra else None,
        )

        if content:
            logger.info("assistant: %s", content[:200])
        logger.info("finish_reason=%s, has_tool_calls=%s", choice.finish_reason, bool(tool_calls))

        # Done? Use improved answer extraction - also check for answer in content even with tool_calls
        if content:
            answer_match = _ANSWER_TAG_RE.search(_THINK_RE.sub("", content))
            if answer_match and not tool_calls and not leaked_calls:
                final_answer = answer_match.group(1).strip()
                logger.info("Task complete at step %d", step)
                break
            elif answer_match and (tool_calls or leaked_calls):
                final_answer = answer_match.group(1).strip()
                logger.info("Answer found in content alongside tool_calls at step %d, using answer", step)
                break

        # Check reasoning_content for leaked <answer> or <function=answer> when content is empty
        if not content and not tool_calls and not leaked_calls and reasoning_content:
            answer_match = _ANSWER_TAG_RE.search(reasoning_content)
            if not answer_match:
                answer_match = _QWEN_ANSWER_RE.search(reasoning_content)
            if answer_match:
                candidate = answer_match.group(1).strip()
                candidate = _TAG_ARTIFACT_RE.sub("", candidate).strip()
                if len(candidate) >= 2:
                    final_answer = candidate
                    logger.info("Answer found in reasoning_content at step %d", step)
                    break

        if not tool_calls and not leaked_calls and choice.finish_reason and content != "":
            final_answer = extract_answer(content)
            logger.info("Task complete at step %d (finish_reason)", step)
            break

        if not tool_calls and not leaked_calls and content == "":
            hint = monitor.consume_hint(step=step, max_steps=max_steps)
            if hint and _task_bool(task, "enable_reflection", ENABLE_REFLECTION_DEFAULT):
                stats["reflection_hints"] = monitor.reflection_hints
                traj.write(
                    Role.USER,
                    hint,
                    step_id=step,
                    extra={"harness_reflection": True},
                )
            continue

        # -------------------------------------------------------- tool calls
        # Merge official tool_calls with leaked ones
        effective_tool_calls = []
        if tool_calls:
            for tc in tool_calls:
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}
                effective_tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": fn_args,
                })
        elif leaked_calls:
            for i, lc in enumerate(leaked_calls):
                effective_tool_calls.append({
                    "id": f"leaked_{step}_{i}",
                    "name": lc["name"],
                    "args": lc.get("arguments", {}),
                })

        pending_hint = ""
        for tc_info in effective_tool_calls:
            fn_name = tc_info["name"]
            fn_args = tc_info["args"]
            tc_id = tc_info["id"]

            logger.info("tool_call: %s(%s)", fn_name, fn_args)
            stats["tool_calls"] += 1

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

            tool_result, truncated = compact_tool_result(tool_result, MAX_TOOL_RESULT_CHARS)
            if truncated:
                stats["tool_result_truncated"] += 1
            monitor.record_tool(fn_name, fn_args, tool_result)
            stats["tool_errors"] = monitor.tool_errors
            stats["repeated_tool_calls"] = monitor.repeated_tool_calls
            logger.info("tool_result (%s): %s", fn_name, str(tool_result)[:200])

            traj.write(
                Role.TOOL,
                tool_result,
                step_id=step,
                tool_call_id=tc_id,
                extra={"fn_name": fn_name, "fn_args": fn_args, "truncated": truncated},
            )
            pending_hint = monitor.consume_hint(step=step, max_steps=max_steps) or pending_hint

        if pending_hint and _task_bool(task, "enable_reflection", ENABLE_REFLECTION_DEFAULT):
            stats["reflection_hints"] = monitor.reflection_hints
            traj.write(
                Role.USER,
                pending_hint,
                step_id=step,
                extra={"harness_reflection": True},
            )
    else:
        logger.warning("Reached max_steps=%d without finish_reason=stop", max_steps)
        max_steps_reached = True
        raw_answer = last_assistant_content or ""
        final_answer = extract_answer(raw_answer) if raw_answer else "[HARNESS] Max steps reached without final answer."

    stats["steps"] = step
    stats["elapsed_seconds"] = time.time() - started_at
    stats["reflection_hints"] = monitor.reflection_hints
    signals = monitor.signals(max_steps_reached=max_steps_reached, final_answer=final_answer)

    # Update short-term memory for cross-task learning
    if short_term_memory is not None and signals.has_failure_signal():
        failure_summary = signals.summary_text()
        short_term_memory.add(
            lesson=f"任务失败：{failure_summary}",
            strategy="避免重复相同错误模式，尽早收敛作答",
            task_family=_task_family(task),
        )

    # LLM-based post-task memory update (enhanced with judge)
    written_memory_ids = _post_task_memory_update(
        task=task,
        instruction=instruction,
        final_answer=final_answer,
        signals=signals,
        stats=stats,
        traj=traj,
    )
    stats["memory_written"] = len(written_memory_ids)

    summary = traj.summary()
    summary["stats"] = stats
    summary["signals"] = signals.to_dict()
    summary["memory"] = {
        "recalled": recalled_memory_ids,
        "written": written_memory_ids,
    }
    traj.write(
        Role.TOOL,
        "[HARNESS SUMMARY]",
        step_id=step,
        extra={
            "harness_summary": True,
            "stats": stats,
            "signals": signals.to_dict(),
            "memory_recalled": recalled_memory_ids,
            "memory_written": written_memory_ids,
        },
    )
    logger.info("Trajectory summary: %s", summary)

    return {
        "task_id":         task_id,
        "answer":          final_answer,
        "steps":           step,
        "trajectory_path": str(traj.path),
        "summary":         summary,
        "stats":           stats,
        "signals":         signals.to_dict(),
        "memory_recalled": recalled_memory_ids,
        "memory_written":  written_memory_ids,
    }


# ---------------------------------------------------------------------------
# Confidence-based retry wrapper (LLM-as-Judge)
# ---------------------------------------------------------------------------

def run_task_with_retry(
    task: dict,
    max_steps: int = MAX_STEPS,
    llm_base_url: str = LLM_BASE_URL,
    model_name: str = MODEL_NAME,
    trajectory_dir: str = "trajectories",
    short_term_memory=None,
    enable_retry: bool = ENABLE_CONFIDENCE_RETRY,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """Run task with optional LLM-as-Judge confidence retry.

    If the judge scores the answer below threshold, re-run with a hint
    about what went wrong. This does NOT use gold answers.
    """
    result = run_task(
        task,
        max_steps=max_steps,
        llm_base_url=llm_base_url,
        model_name=model_name,
        trajectory_dir=trajectory_dir,
        short_term_memory=short_term_memory,
    )

    if not enable_retry or max_retries <= 0:
        return result

    answer = result.get("answer", "")
    if not answer.strip() or answer.startswith("[HARNESS]"):
        return result

    try:
        from llm_judge import judge_answer, JUDGE_ENABLED
        if not JUDGE_ENABLED:
            return result

        traj_summary = summarize_entries(
            Trajectory(result["task_id"], output_dir=trajectory_dir).read_all(),
            max_chars=2000,
        )
        judgment = judge_answer(
            question=task["instruction"],
            trajectory_summary=traj_summary,
            final_answer=answer,
            task_family=_task_family(task),
        )
        conf = judgment.get("confidence", 1.0)
        logger.info("Judge confidence=%.2f for task %s", conf, result["task_id"])
        result["judge_confidence"] = conf
        result["judge_reasoning"] = judgment.get("reasoning", "")

        if conf >= confidence_threshold:
            return result

        logger.info("Low confidence (%.2f < %.2f), retrying task %s",
                    conf, confidence_threshold, result["task_id"])
        suggestion = judgment.get("suggestion", "")
        retry_task = dict(task)
        retry_instruction = task["instruction"]
        if suggestion:
            retry_instruction += f"\n\n[提示：上次尝试的问题是：{suggestion}。请换一种策略重试。]"
        retry_task["instruction"] = retry_instruction
        retry_task["id"] = f"{task.get('id', '')}_retry"

        retry_result = run_task(
            retry_task,
            max_steps=max_steps,
            llm_base_url=llm_base_url,
            model_name=model_name,
            trajectory_dir=trajectory_dir,
            short_term_memory=short_term_memory,
        )

        retry_traj_summary = summarize_entries(
            Trajectory(retry_result["task_id"], output_dir=trajectory_dir).read_all(),
            max_chars=2000,
        )
        retry_judgment = judge_answer(
            question=task["instruction"],
            trajectory_summary=retry_traj_summary,
            final_answer=retry_result.get("answer", ""),
            task_family=_task_family(task),
        )
        retry_conf = retry_judgment.get("confidence", 0.0)
        logger.info("Retry judge confidence=%.2f", retry_conf)

        if retry_conf > conf:
            retry_result["judge_confidence"] = retry_conf
            retry_result["judge_reasoning"] = retry_judgment.get("reasoning", "")
            retry_result["retry_of"] = result["task_id"]
            return retry_result
        return result

    except Exception as exc:
        logger.debug("Confidence retry failed: %s", exc)
        return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen Agent Harness — run a single task from the command line",
    )
    p.add_argument("--instruction", "-i", required=True, help="Task instruction text")
    p.add_argument("--task-id",     "-t", default=None,  help="Optional task ID (auto-generated if omitted)")
    p.add_argument("--max-steps",   "-s", type=int, default=MAX_STEPS, help="Max agent loop steps")
    p.add_argument("--llm-url",           default=LLM_BASE_URL, help="Sglang base URL")
    p.add_argument("--model",             default=MODEL_NAME,   help="Model name")
    p.add_argument("--traj-dir",          default="trajectories", help="Trajectory output directory")
    p.add_argument("--image",             default=None, help="Local path to input image (optional)")
    p.add_argument("--image-url",         default=None, help="Online path to input image (optional)")
    p.add_argument("--task-family",       default="generic", help="Memory namespace, e.g. simplevqa or 2wiki")
    p.add_argument("--memory-dir",        default=None, help="Directory or JSONL path for long-term memory")
    p.add_argument("--memory-update-mode", default=MEMORY_UPDATE_MODE, help="heuristic, gold, or disabled")
    p.add_argument("--gold-answer",       default="", help="Optional gold answer for offline training/evolution only")
    p.add_argument("--allow-gold-feedback", action="store_true", help="Allow gold answer to judge this task for memory updates")
    p.add_argument("--disable-memory",    action="store_true", help="Disable memory recall/update for this run")
    p.add_argument("--disable-reflection", action="store_true", help="Disable in-run reflection hints")
    return p.parse_args()


if __name__ == "__main__":
    import base64

    args = _parse_args()

    image_b64 = None
    if args.image:
        with open(args.image, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
    image_url = None
    if args.image_url:
        image_url = args.image_url

    task = {
        "instruction": args.instruction,
        "image_b64":   image_b64,
        "image_url":   image_url,
        "task_family": args.task_family,
        "memory_dir": args.memory_dir,
        "memory_update_mode": args.memory_update_mode,
        "gold_answer": args.gold_answer,
        "allow_gold_feedback": args.allow_gold_feedback,
        "enable_memory": not args.disable_memory,
        "enable_reflection": not args.disable_reflection,
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
