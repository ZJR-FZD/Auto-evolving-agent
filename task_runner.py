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
import json
import logging
import os
import re
import time
import uuid
from typing import Optional
from pathlib import Path

from openai import OpenAI

from answer_utils import clean_pred_for_submit
from roles import Role
from trajectory import Trajectory
from reflection_module import ReflectionConfig, ReflectionManager
from reflection_module.core import summarize_recent
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
# LLM connection (Sglang OpenAI-compat, two nodes behind Nginx)
# ---------------------------------------------------------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "qwen-3.5")
MAX_STEPS    = int(os.getenv("MAX_STEPS", "20"))
MAX_TOKENS   = int(os.getenv("MAX_TOKENS", "16000"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.3"))
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "1") != "0"
ENABLE_TEXTUAL_TOOL_RESCUE = os.getenv("ENABLE_TEXTUAL_TOOL_RESCUE", "1") != "0"
TASK_TIME_BUDGET_SEC = float(os.getenv("TASK_TIME_BUDGET_SEC", "0") or 0)
DEBUG_PRINT_MESSAGES = os.getenv("DEBUG_PRINT_MESSAGES", "0") == "1"
FAST_TOOL_MODE = os.getenv("FAST_TOOL_MODE", "0") == "1"
FAST_SEARCH_TOP_K = int(os.getenv("FAST_SEARCH_TOP_K", "2"))
FAST_SEARCH_MAX_CHARS = int(os.getenv("FAST_SEARCH_MAX_CHARS", "700"))
FAST_SEARCH_FETCH = os.getenv("FAST_SEARCH_FETCH", "1") != "0"
FAST_BROWSER_TIMEOUT = int(os.getenv("FAST_BROWSER_TIMEOUT", "6"))
FAST_BROWSER_MAX_TEXT = int(os.getenv("FAST_BROWSER_MAX_TEXT", "900"))
ENABLE_TOOL_CACHE = os.getenv("ENABLE_TOOL_CACHE", "1") != "0"
BLOCK_REPEATED_TOOL_EXECUTION = os.getenv("BLOCK_REPEATED_TOOL_EXECUTION", "1") != "0"
MAX_REPEATED_ACTIONS = int(os.getenv("MAX_REPEATED_ACTIONS", "2"))
TEXT_BOUNDED_MODE = os.getenv("TEXT_BOUNDED_MODE", "1") != "0"
TEXT_FORCE_ANSWER_STEP = int(os.getenv("TEXT_FORCE_ANSWER_STEP", "12"))
TEXT_MAX_TOOL_CALLS = int(os.getenv("TEXT_MAX_TOOL_CALLS", "10"))
TEXT_MAX_SEARCH_CALLS = int(os.getenv("TEXT_MAX_SEARCH_CALLS", "8"))
TEXT_MAX_BROWSER_CALLS = int(os.getenv("TEXT_MAX_BROWSER_CALLS", "1"))
TEXT_FINAL_SYNTHESIS = os.getenv("TEXT_FINAL_SYNTHESIS", "1") != "0"
TEXT_FINAL_MAX_TOKENS = int(os.getenv("TEXT_FINAL_MAX_TOKENS", "1200"))
TEXT_FINAL_TIMEOUT = float(os.getenv("TEXT_FINAL_TIMEOUT", "25"))
TEXT_SEARCH_FAILURE_LIMIT = int(os.getenv("TEXT_SEARCH_FAILURE_LIMIT", "2"))
TEXT_UNCERTAIN_RETRY = os.getenv("TEXT_UNCERTAIN_RETRY", "1") != "0"
TEXT_UNCERTAIN_RETRY_MIN_SEARCH = int(os.getenv("TEXT_UNCERTAIN_RETRY_MIN_SEARCH", "4"))
MULTIMODAL_BOUNDED_MODE = os.getenv("MULTIMODAL_BOUNDED_MODE", "1") != "0"
MULTIMODAL_FORCE_VISUAL_NOTES = os.getenv("MULTIMODAL_FORCE_VISUAL_NOTES", "1") != "0"
MULTIMODAL_FORCE_ANSWER_STEP = int(os.getenv("MULTIMODAL_FORCE_ANSWER_STEP", "12"))
MULTIMODAL_MAX_TOOL_CALLS = int(os.getenv("MULTIMODAL_MAX_TOOL_CALLS", "10"))
MULTIMODAL_MAX_SEARCH_CALLS = int(os.getenv("MULTIMODAL_MAX_SEARCH_CALLS", "6"))
MULTIMODAL_MAX_BROWSER_CALLS = int(os.getenv("MULTIMODAL_MAX_BROWSER_CALLS", "2"))
MULTIMODAL_MIN_SEARCH_CALLS = int(os.getenv("MULTIMODAL_MIN_SEARCH_CALLS", "2"))
MULTIMODAL_SEARCH_FAILURE_LIMIT = int(os.getenv("MULTIMODAL_SEARCH_FAILURE_LIMIT", "2"))
MULTIMODAL_UNCERTAIN_RETRY = os.getenv("MULTIMODAL_UNCERTAIN_RETRY", "1") != "0"
MULTIMODAL_FINAL_SYNTHESIS = os.getenv("MULTIMODAL_FINAL_SYNTHESIS", "1") != "0"

# 调试开关：True = 不向 LLM 注册 tools，纯文本对话，便于先验证 LLM 通路
# 工具实现接好后默认关闭；如需调试 LLM 通路，export DISABLE_TOOLS=1
DISABLE_TOOLS = os.getenv("DISABLE_TOOLS", "0") == "1"

ENABLE_REFLECTION = os.getenv("ENABLE_REFLECTION", "1") != "0"
REFLECTION_USE_LLM = os.getenv("REFLECTION_USE_LLM", "0") == "1"
REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "Qwen3-30B-A3B")
REFLECTION_BASE_URL = os.getenv("REFLECTION_BASE_URL", "http://127.0.0.1:8001/v1")
REFLECTION_API_KEY = os.getenv("REFLECTION_API_KEY", "")
REFLECTION_TIMEOUT = float(os.getenv("REFLECTION_TIMEOUT", "30"))
REFLECTION_MAX_CONTEXT_CHARS = int(os.getenv("REFLECTION_MAX_CONTEXT_CHARS", "6000"))
REFLECTION_ALLOW_VISION_CONTEXT = os.getenv("REFLECTION_ALLOW_VISION_CONTEXT", "1") != "0"
REFLECTION_MEMORY_PATH = os.getenv("REFLECTION_MEMORY_PATH", "reflection_memory/reflection_memory.jsonl")
REFLECTION_MAX_MEMORY = int(os.getenv("REFLECTION_MAX_MEMORY", "4"))
REFLECTION_USE_LIGHT_LLM = os.getenv("REFLECTION_USE_LIGHT_LLM", "0") == "1"
REFLECTION_MEMORY_REUSE_THRESHOLD = float(os.getenv("REFLECTION_MEMORY_REUSE_THRESHOLD", "0.58"))
REFLECTION_TEXT_MEMORY_REUSE_THRESHOLD = float(os.getenv("REFLECTION_TEXT_MEMORY_REUSE_THRESHOLD", "0.78"))
REFLECTION_CRITIC_MIN_STEP = int(os.getenv("REFLECTION_CRITIC_MIN_STEP", "3"))
REFLECTION_CRITIC_MAX_CALLS_PER_TASK = int(os.getenv("REFLECTION_CRITIC_MAX_CALLS_PER_TASK", "2"))
REFLECTION_ASYNC_CRITIC = os.getenv("REFLECTION_ASYNC_CRITIC", "0") == "1"
REFLECTION_FORCE_CRITIC_ON_FAILURE_TYPES = os.getenv(
    "REFLECTION_FORCE_CRITIC_ON_FAILURE_TYPES",
    "budget_exhausted,answer_format_error,insufficient_evidence",
)
REFLECTION_CRITIC_ASYNC_LOG_PATH = os.getenv(
    "REFLECTION_CRITIC_ASYNC_LOG_PATH",
    "reflection_memory/critic_async_log.jsonl",
)

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
TOOL_FN_MAP = {
    "search_text":      lambda a: search_text(**a),
    "search_image":     lambda a: search_image(**a),
    "browser_navigate": lambda a: browser_navigate(**a),
    "browser_get_text": lambda a: browser_get_text(**a),
    "browser_click":    lambda a: browser_click(**a),
    "browser_type":     lambda a: browser_type(**a),
    "browser_parallel": lambda a: browser_parallel(**a),
}


def _remaining_seconds(started_at: float) -> float:
    if TASK_TIME_BUDGET_SEC <= 0:
        return 10**9
    return max(0.0, TASK_TIME_BUDGET_SEC - (time.time() - started_at))


def _prepare_tool_args(fn_name: str, fn_args: dict, remaining: float) -> dict:
    args = dict(fn_args or {})
    if not FAST_TOOL_MODE:
        return args

    if fn_name in {"search_text", "search_image"}:
        args["top_k"] = max(1, min(int(args.get("top_k", FAST_SEARCH_TOP_K) or FAST_SEARCH_TOP_K), FAST_SEARCH_TOP_K))
        args["max_chars"] = max(200, min(int(args.get("max_chars", FAST_SEARCH_MAX_CHARS) or FAST_SEARCH_MAX_CHARS), FAST_SEARCH_MAX_CHARS))
        args["fetch"] = bool(FAST_SEARCH_FETCH)
    elif fn_name == "browser_navigate":
        args["timeout"] = max(1, min(int(args.get("timeout", FAST_BROWSER_TIMEOUT) or FAST_BROWSER_TIMEOUT), FAST_BROWSER_TIMEOUT, int(max(1, remaining))))
        args["max_text"] = max(200, min(int(args.get("max_text", FAST_BROWSER_MAX_TEXT) or FAST_BROWSER_MAX_TEXT), FAST_BROWSER_MAX_TEXT))
        args.setdefault("wait_until", "domcontentloaded")
    elif fn_name == "browser_get_text":
        args["timeout"] = max(1, min(int(args.get("timeout", FAST_BROWSER_TIMEOUT) or FAST_BROWSER_TIMEOUT), FAST_BROWSER_TIMEOUT, int(max(1, remaining))))
        args["max_chars"] = max(200, min(int(args.get("max_chars", FAST_BROWSER_MAX_TEXT) or FAST_BROWSER_MAX_TEXT), FAST_BROWSER_MAX_TEXT))
    elif fn_name == "browser_parallel":
        args["timeout"] = max(1, min(int(args.get("timeout", FAST_BROWSER_TIMEOUT) or FAST_BROWSER_TIMEOUT), FAST_BROWSER_TIMEOUT, int(max(1, remaining))))
        args["max_chars"] = max(200, min(int(args.get("max_chars", FAST_BROWSER_MAX_TEXT) or FAST_BROWSER_MAX_TEXT), FAST_BROWSER_MAX_TEXT))
        args["max_concurrency"] = max(1, min(int(args.get("max_concurrency", 2) or 2), 2))
        args.setdefault("wait_until", "domcontentloaded")
    return args


def _prepare_task_type_tool_args(fn_name: str, fn_args: dict, task_type: str, image_url: str | None) -> dict:
    """Apply conservative task-aware defaults without changing tool schemas."""
    args = dict(fn_args or {})
    if task_type in {"2wiki_text", "benchmark_text"} and fn_name == "search_text":
        args.setdefault("top_k", 2)
        args.setdefault("fetch", True)
        args.setdefault("max_chars", 1200)
    elif task_type in {"simplevqa_multimodal", "benchmark_multimodal"}:
        if fn_name == "search_text":
            args.setdefault("top_k", 2)
            args.setdefault("fetch", True)
            args.setdefault("max_chars", 900)
        elif fn_name == "search_image" and image_url and not args.get("image_url"):
            args["image_url"] = image_url
    return args


def _is_text_task(task_type: str) -> bool:
    return task_type in {"2wiki_text", "benchmark_text"}


def _is_multimodal_task(task_type: str) -> bool:
    return task_type in {"simplevqa_multimodal", "benchmark_multimodal"}

TOOLS_SUMMARY = [
    {
        "name": item["function"]["name"],
        "description": item["function"].get("description", "")[:240],
    }
    for item in TOOLS_SCHEMA
]


def _tool_query_key(fn_name: str, fn_args: dict) -> Optional[tuple[str, str]]:
    if fn_name == "search_text":
        query = str(fn_args.get("query", "") or "").strip()
    elif fn_name == "search_image":
        query = str(fn_args.get("image_url", "") or "").strip()
    elif fn_name == "browser_navigate":
        query = str(fn_args.get("url", "") or "").strip()
    else:
        query = ""
    return (fn_name, query) if query else None


def _task_type_appendix(task_type: str, has_image: bool, has_image_url: bool) -> str:
    """Task-aware ReAct policy to keep max-step=8 from being spent on loops."""
    if task_type in {"2wiki_text", "benchmark_text"}:
        return """

## Text QA Strategy
- 先把问题拆成 2-3 个实体/关系子目标，不要盲目长查询。
- 每次 search_text 查询只解决一个子目标；优先用 top_k=2 的短查询。
- 不要连续重复同一 query；若证据不足，改写实体别名、年份、地点或关系词。
- 已有两个独立证据片段能支持答案时，立即输出 <answer>...</answer>，不要继续扩展搜索。
- 2Wiki/文本多跳题遵循：找实体 A -> 找实体 B -> 验证关系/比较条件 -> 答案。
- 不要输出 Based on / According to / I found / 证据链 等解释性前缀；最终答案必须是短 span。
"""
    if task_type in {"simplevqa_multimodal", "benchmark_multimodal"}:
        if has_image_url:
            image_rule = "- 可以使用 search_image，但每次必须带有效 image_url。"
        else:
            image_rule = "- 没有在线 image_url，禁止调用 search_image；先识别图像关键实体/场景，再用 search_text/browser 验证。"
        return f"""

## Multimodal QA Strategy
- 第一步必须先做 Visual Notes：列出可见文字、人物/物体、logo/品牌、地点线索、颜色/版式、题目要求的答案类型；不要第一步直接回答。
{image_rule}
- 根据 Visual Notes 生成 2-3 个短 search_text 查询，每个查询只验证一个候选实体或关系。
- 图像理解和网页证据冲突时，优先再查一个文本证据；没有公网 image_url 时不要调用 search_image。
- 不要过早输出 Insufficient evidence；如果图像线索不完整，也要先用最强视觉线索搜索验证。
- 证据足够时立即输出 <answer>...</answer>，只放答案本身，不要写证据链。
"""
    return ""


class _SyntheticFunction:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = json.dumps(arguments, ensure_ascii=False)


class _SyntheticToolCall:
    def __init__(self, name: str, arguments: dict):
        self.id = f"rescued_{uuid.uuid4().hex[:12]}"
        self.function = _SyntheticFunction(name, arguments)
        self.type = "function"
        self.index = 0

    def model_dump(self) -> dict:
        return {
            "id": self.id,
            "function": {"name": self.function.name, "arguments": self.function.arguments},
            "type": self.type,
            "index": self.index,
            "rescued_from_text": True,
        }


def _coerce_tool_arg(value: str) -> object:
    s = str(value or "").strip()
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    if re.fullmatch(r"-?\d+", s):
        try:
            return int(s)
        except ValueError:
            return s
    return s


def _extract_textual_tool_calls(text: str, limit: int = 2) -> list[_SyntheticToolCall]:
    """Recover XML-like tool intents that Qwen sometimes emits in reasoning text."""
    if not ENABLE_TEXTUAL_TOOL_RESCUE or not text:
        return []
    rescued: list[_SyntheticToolCall] = []

    for match in re.finditer(r"<function=([A-Za-z_][\w]*)>\s*(.*?)\s*</function>", text, flags=re.S):
        name = match.group(1)
        if name not in TOOL_FN_MAP:
            continue
        body = match.group(2)
        args: dict[str, object] = {}
        for p in re.finditer(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", body, flags=re.S):
            args[p.group(1).strip()] = _coerce_tool_arg(p.group(2))
        if args:
            rescued.append(_SyntheticToolCall(name, args))
        if len(rescued) >= limit:
            return rescued

    for match in re.finditer(r"\b(search_text|search_image|browser_navigate|browser_get_text|browser_parallel)\s*\((\{.*?\})\)", text, flags=re.S):
        name = match.group(1)
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
        if isinstance(args, dict):
            rescued.append(_SyntheticToolCall(name, args))
        if len(rescued) >= limit:
            break
    return rescued


def _synthesize_final_answer(
    client: OpenAI,
    model_name: str,
    instruction: str,
    task_type: str,
    messages: list[dict],
) -> str:
    """Produce a best-effort final answer from existing trajectory evidence.

    This is a no-tool finalizer used to avoid returning the useless
    "[HARNESS] Max steps reached" string for hard text multi-hop cases.
    It does not receive gold answers and cannot call tools.
    """
    recent = summarize_recent(messages, limit=18)
    prompt = {
        "task_type": task_type,
        "instruction": instruction,
        "recent_trajectory": recent,
        "requirement": (
            "You are an answer extractor, not an explainer. Use only evidence already present "
            "in the trajectory. Output exactly one short answer span inside <answer>...</answer>. "
            "Do not include reasoning, citations, uncertainty phrases, or explanations. "
            "If the question is yes/no, output exactly yes or no. If the answer is a person, "
            "title, date, number, nationality, or location, output only that span. "
            "Choose the best-supported candidate if evidence is incomplete. Do not request tools."
        ),
    }
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict final answer extractor. You cannot call tools. "
                        "Return only <answer>short answer</answer>. Never write 'Based on', "
                        "'likely', 'unable to determine', or any explanation."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            max_tokens=TEXT_FINAL_MAX_TOKENS,
            temperature=0.1,
            timeout=TEXT_FINAL_TIMEOUT,
            extra_body={"enable_thinking": False},
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Final synthesis failed: %s", exc)
        return ""


def _fallback_answer_from_existing_messages(messages: list[dict]) -> str:
    """Never return the harness failure string when a text task must stop.

    Prefer an answer already emitted by the model.  If the trajectory contains
    no usable content because the only search timed out, return an explicit
    low-confidence placeholder that is still a valid final-answer shape.
    """
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content") or "").strip()
        if not content or "[HARNESS]" in content:
            continue
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", content, flags=re.S | re.I)
        if match and match.group(1).strip():
            return f"<answer>{match.group(1).strip()}</answer>"
        return f"<answer>{content.splitlines()[-1].strip()}</answer>"
    return "<answer>Insufficient evidence</answer>"


def _is_uncertain_answer(content: str) -> bool:
    text = str(content or "")
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S | re.I)
    if match:
        text = match.group(1)
    low = text.lower()
    return any(
        phrase in low
        for phrase in (
            "cannot determine",
            "unable to determine",
            "insufficient evidence",
            "not enough evidence",
            "not definitively identify",
            "does not definitively",
            "could not find",
            "无法确定",
            "证据不足",
            "无法判断",
        )
    )


def _is_search_proxy_failure(tool_result: object) -> bool:
    text = str(tool_result).lower()
    return (
        "proxy-error" in text
        or "readtimeout" in text
        or "read timed out" in text
        or "connectionpool" in text
        or "search proxy request failed" in text
    )

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一个高效、严谨的任务执行 Agent，运行在配备多工具的自动化框架中。

## 行为准则
1. 每一步只做一个动作：调用一个必要工具，或在证据足够时直接给出最终答案。
2. 工具调用必须使用系统提供的 function/tool_calls 协议；禁止在普通文本或 reasoning 中写 XML/JSON 伪工具调用。
3. 若工具返回 ok=False，分析 error，最多重试 2 次同类操作；仍失败则换工具或方法。
4. 若调用search_image工具，请使用输入图像的在线链接。
5. 如果图像任务没有在线 image_url，不要调用 search_image；先基于图像内容识别关键实体/场景，再用 search_text 或 browser 验证。
6. 控制推理长度：优先短查询、短证据链和尽早回答，避免反复搜索同一问题。
7. 最终答案用 <answer>...</answer> 包裹，只放答案本身；禁止写 Based on、According to、可能是、证据链、解释或免责声明。
"""


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

    Returns:
        Dict with keys: task_id, answer, steps, trajectory_path, summary
    """
    task_id     = task.get("id") or str(uuid.uuid4())[:8]
    instruction = task["instruction"]
    image_b64   = task.get("image_b64")
    image_url   = task.get("image_url")
    task_type   = task.get("task_type") or ("simplevqa_multimodal" if (image_b64 or image_url) else "general")
    image_info  = task.get("image_info") or {}
    if image_url:
        image_info.setdefault("image_url", image_url)

    logger.info("run_task: task_id=%s", task_id)
    task_started_at = time.time()

    traj   = Trajectory(task_id, output_dir=trajectory_dir)
    client = OpenAI(base_url=llm_base_url, api_key="EMPTY", timeout=LLM_TIMEOUT)
    reflection = ReflectionManager(
        ReflectionConfig(
            enabled=ENABLE_REFLECTION,
            use_llm=REFLECTION_USE_LLM,
            use_light_llm=REFLECTION_USE_LIGHT_LLM,
            memory_path=REFLECTION_MEMORY_PATH,
            max_memory_items=REFLECTION_MAX_MEMORY,
            memory_reuse_threshold=REFLECTION_MEMORY_REUSE_THRESHOLD,
            text_memory_reuse_threshold=REFLECTION_TEXT_MEMORY_REUSE_THRESHOLD,
            critic_min_step=REFLECTION_CRITIC_MIN_STEP,
            critic_max_calls_per_task=REFLECTION_CRITIC_MAX_CALLS_PER_TASK,
            async_critic=REFLECTION_ASYNC_CRITIC,
            force_critic_failure_types=REFLECTION_FORCE_CRITIC_ON_FAILURE_TYPES,
            critic_async_log_path=REFLECTION_CRITIC_ASYNC_LOG_PATH,
            llm_model=REFLECTION_MODEL,
            base_url=REFLECTION_BASE_URL or None,
            api_key=REFLECTION_API_KEY,
            timeout=REFLECTION_TIMEOUT,
            max_prompt_chars=REFLECTION_MAX_CONTEXT_CHARS,
            allow_vision_context=REFLECTION_ALLOW_VISION_CONTEXT,
        ),
        client=None,
        model_name=REFLECTION_MODEL,
    )

    # ------------------------------------------------------------------ step 0
    # Write system turn
    system_prompt = (
        SYSTEM_PROMPT
        + _task_type_appendix(task_type, has_image=bool(image_b64 or image_url), has_image_url=bool(image_url))
        + reflection.build_system_appendix(instruction, task_type=task_type)
    )
    memory_hits_count = len(reflection.last_memory_hits)
    traj.write(Role.SYSTEM, system_prompt, step_id=0)

    # Build user message (optionally include image)
    if image_b64 and image_url:
        user_content = [
            {"type": "text",      "text": instruction + "\n输入图像的在线链接：" + image_url},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
    elif image_b64:
        user_content = [
            {"type": "text",      "text": instruction},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
    else:
        user_content = instruction

    traj.write(Role.USER, user_content, step_id=0)
    if MULTIMODAL_FORCE_VISUAL_NOTES and _is_multimodal_task(task_type):
        traj.write(
            Role.USER,
            (
                "[Multimodal grounding protocol]\n"
                "先观察输入图片并写出简短 Visual Notes，再决定是否搜索；不要第一步直接回答。"
                "Visual Notes 必须覆盖：visible_text、objects/people、logos/brands、location/style clues、"
                "answer_type、2-3 个可执行 search_text 查询。"
                "如果看不清图片，也要说明可见线索并用题目约束生成搜索查询；"
                "禁止在未至少尝试文本搜索验证前输出 Insufficient evidence。"
            ),
            step_id=0,
            extra={"multimodal_grounding_protocol": True},
        )

    # ------------------------------------------------------------------ loop
    final_answer = ""
    query_counter: dict[tuple[str, str], int] = {}
    tool_result_cache: dict[tuple[str, str], str] = {}
    total_tool_calls = 0
    text_search_calls = 0
    text_browser_calls = 0
    text_search_failures = 0
    multimodal_search_calls = 0
    multimodal_browser_calls = 0
    multimodal_search_failures = 0
    force_answer_injected = False
    uncertain_retry_injected = False

    for step in range(1, max_steps + 1):
        remaining_for_task = _remaining_seconds(task_started_at)
        if TASK_TIME_BUDGET_SEC > 0 and remaining_for_task <= 1:
            logger.warning("Task time budget reached: %.1fs", TASK_TIME_BUDGET_SEC)
            final_answer = ""
            if (TEXT_FINAL_SYNTHESIS and _is_text_task(task_type)) or (
                MULTIMODAL_FINAL_SYNTHESIS and _is_multimodal_task(task_type)
            ):
                final_answer = _synthesize_final_answer(
                    client=client,
                    model_name=model_name,
                    instruction=instruction,
                    task_type=task_type,
                    messages=traj.to_messages(),
                )
                if final_answer:
                    traj.write(
                        Role.ASSISTANT,
                        final_answer,
                        step_id=step,
                        extra={"final_synthesis": True, "finish_reason": "forced_text_time_budget"},
                    )
            if not final_answer:
                final_answer = _fallback_answer_from_existing_messages(traj.to_messages())
            break
        logger.info("--- step %d ---", step)

        force_answer_mode = False
        if (
            TEXT_BOUNDED_MODE
            and _is_text_task(task_type)
            and not force_answer_injected
            and (
                step >= min(max_steps, TEXT_FORCE_ANSWER_STEP)
                or total_tool_calls >= TEXT_MAX_TOOL_CALLS
                or text_search_failures >= TEXT_SEARCH_FAILURE_LIMIT
            )
        ):
            force_answer_injected = True
            force_answer_mode = True
            reason = (
                "search_proxy_degraded"
                if text_search_failures >= TEXT_SEARCH_FAILURE_LIMIT
                else "bounded_budget"
            )
            traj.write(
                Role.USER,
                (
                    "[Text bounded mode]\n"
                    f"触发原因：{reason}。\n"
                    "停止继续搜索和浏览。请只基于当前已有证据和模型已有知识综合最终答案。"
                    "如果证据不完整，给出最有支持的答案，不要输出 [HARNESS] 或请求更多工具。"
                    "最终只输出 <answer>...</answer>。"
                ),
                step_id=step,
                extra={
                    "text_bounded_force_answer": True,
                    "total_tool_calls": total_tool_calls,
                    "text_search_calls": text_search_calls,
                    "text_browser_calls": text_browser_calls,
                    "text_search_failures": text_search_failures,
                    "force_answer_reason": reason,
                },
            )

        if (
            MULTIMODAL_BOUNDED_MODE
            and _is_multimodal_task(task_type)
            and not force_answer_injected
            and (
                step >= min(max_steps, MULTIMODAL_FORCE_ANSWER_STEP)
                or total_tool_calls >= MULTIMODAL_MAX_TOOL_CALLS
                or multimodal_search_failures >= MULTIMODAL_SEARCH_FAILURE_LIMIT
            )
        ):
            force_answer_injected = True
            force_answer_mode = True
            reason = (
                "search_proxy_degraded"
                if multimodal_search_failures >= MULTIMODAL_SEARCH_FAILURE_LIMIT
                else "multimodal_bounded_budget"
            )
            traj.write(
                Role.USER,
                (
                    "[Multimodal bounded mode]\n"
                    f"触发原因：{reason}。\n"
                    "停止继续扩展搜索。请只基于已有图片观察、搜索摘要和页面证据综合最终答案。"
                    "如果证据不完整，选择最有支持的候选，不要输出 [HARNESS]、不要请求更多工具、"
                    "不要输出 Insufficient evidence，除非完全没有任何候选。最终只输出 <answer>...</answer>。"
                ),
                step_id=step,
                extra={
                    "multimodal_bounded_force_answer": True,
                    "total_tool_calls": total_tool_calls,
                    "multimodal_search_calls": multimodal_search_calls,
                    "multimodal_browser_calls": multimodal_browser_calls,
                    "multimodal_search_failures": multimodal_search_failures,
                    "force_answer_reason": reason,
                },
            )

        messages = traj.to_messages()
        logger.info("messages count=%d, sending to LLM ...", len(messages))

        # 构造请求参数：调试模式下不注册 tools，避免协议不匹配
        request_kwargs = dict(
            model=model_name,
            messages=messages,
            max_tokens=min(MAX_TOKENS, TEXT_FINAL_MAX_TOKENS) if force_answer_mode else MAX_TOKENS,
            temperature=AGENT_TEMPERATURE,
            extra_body={"enable_thinking": False if force_answer_mode else ENABLE_THINKING},
            timeout=max(1.0, min(LLM_TIMEOUT, remaining_for_task - 1 if remaining_for_task < 10**8 else LLM_TIMEOUT)),
        )
        if not DISABLE_TOOLS and not force_answer_mode:
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
        if DEBUG_PRINT_MESSAGES:
            print(msg)
        content = msg.content or ""
        reasoning_content = getattr(msg, "reasoning_content", None) or ""
        total_tokens = response.usage.total_tokens if response.usage else ""

        # 调试模式下强制忽略 tool_calls（虽然不传 tools 通常不会出现）
        tool_calls = None if DISABLE_TOOLS else msg.tool_calls
        rescued_tool_calls = []
        if not tool_calls:
            rescued_tool_calls = _extract_textual_tool_calls("\n".join([content, reasoning_content]))
            if rescued_tool_calls:
                logger.info("rescued %d textual tool call(s)", len(rescued_tool_calls))
                tool_calls = rescued_tool_calls

        # Write assistant turn
        tool_calls_data = (
            [tc.model_dump() for tc in tool_calls]
            if tool_calls else []
        )
        
        extra = {}
        
        if tool_calls_data:
            extra["tool_calls"] = tool_calls_data
        if rescued_tool_calls:
            extra["textual_tool_rescue"] = True
        if reasoning_content:
            extra["reasoning_content"] = reasoning_content
        if choice.finish_reason:
            extra["finish_reason"] = choice.finish_reason
        if total_tokens:
            extra["total_tokens"] = total_tokens
                        
        traj.write(
            Role.ASSISTANT,
            content,
            step_id=step,
            extra= extra if extra else None,
        )

        if content:
            logger.info("assistant: %s", content[:200])
        logger.info("finish_reason=%s, has_tool_calls=%s", choice.finish_reason, bool(tool_calls))

        assistant_failure = reflection.detect_assistant_failure(
            task_id=task_id,
            step_id=step,
            instruction=instruction,
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            recent_messages=summarize_recent(messages),
            task_type=task_type,
            tools_summary=TOOLS_SUMMARY,
            image_info=image_info,
        )
        if assistant_failure:
            record = reflection.reflect(assistant_failure)
            traj.write(
                Role.USER,
                reflection.to_feedback_message(record),
                step_id=step,
                extra={
                    "reflection_trigger": True,
                    "reflection_mode": record.critic_model,
                    "reflection_source": record.critic_model,
                    "failure_type": record.failure_type,
                    "root_cause": record.root_cause,
                    "correction_strategy": record.correction_strategy,
                    "memory_used": memory_hits_count > 0,
                    "memory_hits": memory_hits_count,
                    "memory_reused": record.critic_model == "memory_reuse",
                    "source_memory_id": record.memory_id,
                    "memory_written": record.memory_written,
                    "critic_confidence": record.confidence,
                },
            )
            continue

        # Done?
        # 文本题如果过早输出 "Cannot determine / Insufficient evidence"，且搜索预算
        # 还没用完，则不直接结束；注入一次纠偏提示，要求继续改写 query。
        if (
            TEXT_UNCERTAIN_RETRY
            and _is_text_task(task_type)
            and not tool_calls
            and content.strip()
            and not force_answer_mode
            and not uncertain_retry_injected
            and _is_uncertain_answer(content)
            and text_search_calls < TEXT_MAX_SEARCH_CALLS
            and total_tool_calls < TEXT_MAX_TOOL_CALLS
        ):
            uncertain_retry_injected = True
            traj.write(
                Role.USER,
                (
                    "[Uncertain answer blocked]\n"
                    "当前回答是不确定/证据不足类型，但搜索预算尚未用完。请不要结束任务。"
                    "先列出缺失的关键实体或关系，改写为一个更短、更具体的 search_text 查询；"
                    "如果已有候选实体，则打开最相关页面验证。最终答案必须是一个短 span。"
                ),
                step_id=step,
                extra={
                    "uncertain_answer_blocked": True,
                    "text_search_calls": text_search_calls,
                    "total_tool_calls": total_tool_calls,
                },
            )
            continue

        if (
            MULTIMODAL_UNCERTAIN_RETRY
            and _is_multimodal_task(task_type)
            and not tool_calls
            and content.strip()
            and not force_answer_mode
            and not uncertain_retry_injected
            and _is_uncertain_answer(content)
            and multimodal_search_calls < MULTIMODAL_MIN_SEARCH_CALLS
            and total_tool_calls < MULTIMODAL_MAX_TOOL_CALLS
        ):
            uncertain_retry_injected = True
            traj.write(
                Role.USER,
                (
                    "[Multimodal uncertain answer blocked]\n"
                    "当前回答是不确定/证据不足类型，但多模态题还没有完成最少文本验证。"
                    "请不要结束任务。先用图片中的可见文字、物体、logo、地点或风格线索生成一个短 search_text 查询；"
                    "如果已有候选实体，则搜索候选实体加题目关键词进行验证。最终答案必须是一个短 span。"
                ),
                step_id=step,
                extra={
                    "multimodal_uncertain_answer_blocked": True,
                    "multimodal_search_calls": multimodal_search_calls,
                    "total_tool_calls": total_tool_calls,
                },
            )
            continue

        # 标准退出条件：没有 tool_calls 时就结束（finish_reason 可能是 stop / length 等）
        if not tool_calls and choice.finish_reason and content != "":
            final_answer = content
            logger.info("Task complete at step %d", step)
            break
        
        if not tool_calls and content == "":
            continue

        # -------------------------------------------------------- tool calls
        tool_reflections = []
        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as exc:
                fn_args = {}
                logger.warning("Bad tool args JSON: %s", exc)

            logger.info("tool_call: %s(%s)", fn_name, fn_args)
            fn_args = _prepare_tool_args(fn_name, fn_args, _remaining_seconds(task_started_at))
            fn_args = _prepare_task_type_tool_args(fn_name, fn_args, task_type, image_url)
            query_key = _tool_query_key(fn_name, fn_args)
            repeat_count = 0
            if query_key:
                query_counter[query_key] = query_counter.get(query_key, 0) + 1
                repeat_count = query_counter[query_key]
            tool_cache_hit = False

            # Dispatch
            if fn_name not in TOOL_FN_MAP:
                tool_result = f"[ERROR] Unknown tool: {fn_name}"
            elif query_key and ENABLE_TOOL_CACHE and query_key in tool_result_cache:
                tool_result = tool_result_cache[query_key]
                tool_cache_hit = True
                logger.info("tool_cache_hit: %s", query_key)
            elif (
                query_key
                and BLOCK_REPEATED_TOOL_EXECUTION
                and repeat_count > MAX_REPEATED_ACTIONS
            ):
                tool_result = (
                    f"[HARNESS ERROR] Repeated action blocked after {repeat_count} attempts: "
                    f"{fn_name} {query_key[1]}. Revise query/tool or answer with existing evidence."
                )
            elif TEXT_BOUNDED_MODE and _is_text_task(task_type) and total_tool_calls >= TEXT_MAX_TOOL_CALLS:
                tool_result = (
                    f"[HARNESS NOTICE] Text tool budget reached ({total_tool_calls}/{TEXT_MAX_TOOL_CALLS}). "
                    "Stop using tools and answer from existing evidence."
                )
            elif MULTIMODAL_BOUNDED_MODE and _is_multimodal_task(task_type) and total_tool_calls >= MULTIMODAL_MAX_TOOL_CALLS:
                tool_result = (
                    f"[HARNESS NOTICE] Multimodal tool budget reached ({total_tool_calls}/{MULTIMODAL_MAX_TOOL_CALLS}). "
                    "Stop using tools and answer from existing visual/search evidence."
                )
            elif TEXT_BOUNDED_MODE and _is_text_task(task_type) and fn_name == "search_text" and text_search_calls >= TEXT_MAX_SEARCH_CALLS:
                tool_result = (
                    f"[HARNESS NOTICE] Text search budget reached ({text_search_calls}/{TEXT_MAX_SEARCH_CALLS}). "
                    "Use existing evidence or switch to final answer."
                )
            elif (
                MULTIMODAL_BOUNDED_MODE
                and _is_multimodal_task(task_type)
                and fn_name == "search_text"
                and multimodal_search_calls >= MULTIMODAL_MAX_SEARCH_CALLS
            ):
                tool_result = (
                    f"[HARNESS NOTICE] Multimodal search budget reached ({multimodal_search_calls}/{MULTIMODAL_MAX_SEARCH_CALLS}). "
                    "Use existing visual/search evidence or switch to final answer."
                )
            elif (
                TEXT_BOUNDED_MODE
                and _is_text_task(task_type)
                and fn_name == "search_text"
                and text_search_failures >= TEXT_SEARCH_FAILURE_LIMIT
            ):
                tool_result = (
                    f"[HARNESS NOTICE] Search proxy appears degraded after {text_search_failures} failed search calls. "
                    "Do not call search_text again for this task; answer from existing evidence or general knowledge."
                )
            elif (
                MULTIMODAL_BOUNDED_MODE
                and _is_multimodal_task(task_type)
                and fn_name == "search_text"
                and multimodal_search_failures >= MULTIMODAL_SEARCH_FAILURE_LIMIT
            ):
                tool_result = (
                    f"[HARNESS NOTICE] Search proxy appears degraded after {multimodal_search_failures} failed multimodal search calls. "
                    "Do not call search_text again for this task; answer from existing visual/search evidence."
                )
            elif (
                TEXT_BOUNDED_MODE
                and _is_text_task(task_type)
                and fn_name in {"browser_navigate", "browser_get_text", "browser_parallel"}
                and text_browser_calls >= TEXT_MAX_BROWSER_CALLS
            ):
                tool_result = (
                    f"[HARNESS NOTICE] Text browser budget reached ({text_browser_calls}/{TEXT_MAX_BROWSER_CALLS}). "
                    "Prefer search evidence and answer now."
                )
            elif (
                MULTIMODAL_BOUNDED_MODE
                and _is_multimodal_task(task_type)
                and fn_name in {"browser_navigate", "browser_get_text", "browser_parallel"}
                and multimodal_browser_calls >= MULTIMODAL_MAX_BROWSER_CALLS
            ):
                tool_result = (
                    f"[HARNESS NOTICE] Multimodal browser budget reached ({multimodal_browser_calls}/{MULTIMODAL_MAX_BROWSER_CALLS}). "
                    "Prefer visual/search evidence and answer now."
                )
            elif fn_name == "search_image" and not fn_args.get("image_url"):
                tool_result = "[HARNESS ERROR] search_image requires an online image_url; use visual description + search_text instead."
            elif TASK_TIME_BUDGET_SEC > 0 and _remaining_seconds(task_started_at) <= 1:
                tool_result = "[HARNESS ERROR] Task time budget reached before tool execution."
            else:
                total_tool_calls += 1
                if _is_text_task(task_type) and fn_name == "search_text":
                    text_search_calls += 1
                if _is_text_task(task_type) and fn_name in {"browser_navigate", "browser_get_text", "browser_parallel"}:
                    text_browser_calls += 1
                if _is_multimodal_task(task_type) and fn_name == "search_text":
                    multimodal_search_calls += 1
                if _is_multimodal_task(task_type) and fn_name in {"browser_navigate", "browser_get_text", "browser_parallel"}:
                    multimodal_browser_calls += 1
                try:
                    raw = TOOL_FN_MAP[fn_name](fn_args)
                    # 工具返回结构化对象时，序列化为 JSON 字符串方便 LLM 解读
                    if isinstance(raw, (dict, list)):
                        tool_result = json.dumps(raw, ensure_ascii=False)
                    else:
                        tool_result = str(raw)
                except Exception as exc:
                    tool_result = f"[ERROR] Tool '{fn_name}' raised: {type(exc).__name__}: {exc}"
                    if "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
                        logger.warning("Tool timeout (%s): %s", fn_name, exc)
                    else:
                        logger.exception("Tool error")
            if query_key and ENABLE_TOOL_CACHE and not str(tool_result).startswith("[HARNESS ERROR] Repeated action blocked"):
                tool_result_cache[query_key] = str(tool_result)
            if _is_text_task(task_type) and fn_name == "search_text":
                if _is_search_proxy_failure(tool_result):
                    text_search_failures += 1
                elif str(tool_result).strip() and "proxy-error" not in str(tool_result).lower():
                    text_search_failures = 0
            if _is_multimodal_task(task_type) and fn_name == "search_text":
                if _is_search_proxy_failure(tool_result):
                    multimodal_search_failures += 1
                elif str(tool_result).strip() and "proxy-error" not in str(tool_result).lower():
                    multimodal_search_failures = 0

            logger.info("tool_result (%s): %s", fn_name, str(tool_result)[:200])

            traj.write(
                Role.TOOL,
                tool_result,
                step_id=step,
                tool_call_id=tc.id,
                extra={
                    "fn_name": fn_name,
                    "fn_args": fn_args,
                    "tool_cache_hit": tool_cache_hit,
                    "repeat_count": repeat_count,
                    "total_tool_calls": total_tool_calls,
                    "text_search_calls": text_search_calls,
                    "text_browser_calls": text_browser_calls,
                    "multimodal_search_calls": multimodal_search_calls,
                    "multimodal_browser_calls": multimodal_browser_calls,
                },
            )

            tool_failure = reflection.detect_tool_failure(
                task_id=task_id,
                step_id=step,
                instruction=instruction,
                tool_name=fn_name,
                tool_args=fn_args,
                tool_result=tool_result,
                recent_messages=summarize_recent(traj.to_messages()),
                task_type=task_type,
                tools_summary=TOOLS_SUMMARY,
                image_info=image_info,
            )
            if tool_failure:
                tool_reflections.append(reflection.reflect(tool_failure))

            if query_key:
                # Trigger loop correction once per repeated action; repeated
                # failures after that are handled by cache/blocking instead of
                # spending more external calls on the same query.
                if repeat_count == 2:
                    repeated_event = reflection.repeated_query_failure(
                        task_id=task_id,
                        step_id=step,
                        instruction=instruction,
                        tool_name=fn_name,
                        query=query_key[1],
                        count=repeat_count,
                        recent_messages=summarize_recent(traj.to_messages()),
                        task_type=task_type,
                        tools_summary=TOOLS_SUMMARY,
                        image_info=image_info,
                    )
                    tool_reflections.append(reflection.reflect(repeated_event))

        if tool_reflections:
            first = tool_reflections[0]
            traj.write(
                Role.USER,
                "\n\n".join(reflection.to_feedback_message(r) for r in tool_reflections[:2]),
                step_id=step,
                extra={
                    "reflection_trigger": True,
                    "reflection_mode": first.critic_model,
                    "reflection_source": first.critic_model,
                    "failure_type": first.failure_type,
                    "root_cause": first.root_cause,
                    "correction_strategy": first.correction_strategy,
                    "memory_used": memory_hits_count > 0,
                    "memory_hits": memory_hits_count,
                    "memory_reused": first.critic_model == "memory_reuse",
                    "source_memory_id": first.memory_id,
                    "memory_written": any(r.memory_written for r in tool_reflections),
                    "critic_confidence": first.confidence,
                },
            )
    else:
        logger.warning("Reached max_steps=%d without finish_reason=stop", max_steps)
        final_answer = ""
        if (TEXT_FINAL_SYNTHESIS and _is_text_task(task_type)) or (
            MULTIMODAL_FINAL_SYNTHESIS and _is_multimodal_task(task_type)
        ):
            final_answer = _synthesize_final_answer(
                client=client,
                model_name=model_name,
                instruction=instruction,
                task_type=task_type,
                messages=traj.to_messages(),
            )
            if final_answer:
                traj.write(
                    Role.ASSISTANT,
                    final_answer,
                    step_id=max_steps,
                    extra={"final_synthesis": True, "finish_reason": "forced_final"},
                )
        if not final_answer:
            final_answer = _fallback_answer_from_existing_messages(traj.to_messages())
        if ENABLE_REFLECTION:
            event = reflection.max_steps_failure(
                task_id=task_id,
                step_id=max_steps,
                instruction=instruction,
                recent_messages=summarize_recent(traj.to_messages()),
                task_type=task_type,
                tools_summary=TOOLS_SUMMARY,
                image_info=image_info,
            )
            record = reflection.reflect(event)
            traj.write(
                Role.USER,
                reflection.to_feedback_message(record),
                step_id=max_steps,
                extra={
                    "reflection_trigger": True,
                    "reflection_mode": record.critic_model,
                    "reflection_source": record.critic_model,
                    "failure_type": record.failure_type,
                    "root_cause": record.root_cause,
                    "correction_strategy": record.correction_strategy,
                    "memory_used": memory_hits_count > 0,
                    "memory_hits": memory_hits_count,
                    "memory_reused": record.critic_model == "memory_reuse",
                    "source_memory_id": record.memory_id,
                    "memory_written": record.memory_written,
                    "critic_confidence": record.confidence,
                },
            )

    summary = traj.summary()
    logger.info("Trajectory summary: %s", summary)
    raw_answer = final_answer
    cleaned_answer = clean_pred_for_submit(raw_answer, instruction)

    return {
        "task_id":         task_id,
        "answer":          cleaned_answer,
        "raw_answer":      raw_answer,
        "steps":           step,
        "trajectory_path": str(traj.path),
        "summary":         summary,
    }


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
