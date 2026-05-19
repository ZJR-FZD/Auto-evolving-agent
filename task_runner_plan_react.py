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
from typing import Optional
from pathlib import Path

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
MAX_STEPS    = int(os.getenv("MAX_STEPS", "20"))
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
                    "fetch":     {"type": "boolean", "description": "是否抓取正文", "default": True},
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
                    "fetch":     {"type": "boolean", "description": "是否抓取正文", "default": True},
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

# PLACEHOLDER_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# System prompt — Plan & ReAct
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一个高效、严谨的任务执行 Agent，运行在配备多工具的自动化框架中。

## 工作流程（Plan → ReAct → Answer）

### 第一步：制定计划
收到任务后，先分析问题并制定 2-5 步的搜索计划。明确：
- 需要确认哪些关键信息
- 每步搜索的目标是什么
- 信息之间的依赖关系

### 第二步：逐步执行（ReAct）
每一步：
1. 回顾已获取的信息，评估进展
2. 决定下一步行动（调用工具）
3. 执行后反思：这个结果是否有用？是否需要调整策略？

### 第三步：输出答案
当信息足够时，用 <answer>你的最终答案</answer> 标签输出简洁的最终答案。
答案应该只包含问题要求的核心信息，不要包含解释过程。

## 行为准则
1. 每一步要么调用工具，要么输出最终答案（用 <answer> 标签包裹）。
2. 若工具返回失败、或返回内容表明目标网站拒绝访问（如反爬拦截、403、需要登录、验证码、"automated tool" 等提示），视同失败。不要重复访问同一域名，改用搜索引擎获取该页面的摘要信息。
3. 同类操作最多重试 2 次，仍失败则必须换工具或换搜索策略。
4. 若调用 search_image 工具，请使用输入图像的在线链接。
5. 搜索时优先使用英文关键词，结果更丰富。
6. 不要在一个方向上花费超过 3 步，如果没有进展就换个角度。
"""

FORCE_ANSWER_PROMPT = """[系统提示] 你已经使用了大部分可用步数。请立即根据目前已收集到的信息，输出你的最终答案。
即使信息不完整，也请给出你最有把握的答案。用 <answer>你的答案</answer> 标签包裹。"""


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

    for step in range(1, max_steps + 1):
        logger.info("--- step %d/%d ---", step, max_steps)

        # Inject force-answer prompt near the end
        if step == max_steps - 2 and not force_answer_injected:
            traj.write(Role.USER, FORCE_ANSWER_PROMPT, step_id=step)
            force_answer_injected = True
            logger.info("Injected FORCE_ANSWER_PROMPT")

        messages = traj.to_messages()
        logger.info("messages count=%d, sending to LLM ...", len(messages))

        request_kwargs = dict(
            model=model_name,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            extra_body={"enable_thinking": True},
        )
        if not DISABLE_TOOLS:
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

        # --- Execute tool calls ---
        for tc in (tool_calls or []):
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

            logger.info("tool_result (%s): %s", fn_name, str(tool_result)[:200])
            traj.write(Role.TOOL, tool_result, step_id=step, tool_call_id=tc_id,
                       extra={"fn_name": fn_name, "fn_args": fn_args})
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
