"""
Qwen Agent Harness — Plan & ReAct with Online Negative Critic
==============================================================

Pattern:
  - Main model (e.g. 9B) runs normal ReAct loop with tools.
  - A stronger critic model (e.g. 32B) evaluates trajectory online.
  - Critic can ONLY output GOOD/BAD + short reason, never action hints.
  - If BAD, harness injects a forced self-reflection prompt to main model.

Usage:
    python task_runner_plan_react_negcrit.py -i "your question" -t task_001
"""

import argparse
import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from openai import OpenAI

from roles import Role
from trajectory import Trajectory
from task_runner_plan_react import (
    TOOL_FN_MAP,
    TOOLS_SCHEMA,
    extract_answer,
    filter_garbage_results,
    parse_leaked_tool_calls,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("harness.plan_react_negcrit")


LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3.5-9B")
MAX_STEPS = int(os.getenv("MAX_STEPS", "15"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "16000"))
DISABLE_TOOLS = os.getenv("DISABLE_TOOLS", "0") == "1"

NEG_CRITIC_ENABLED = os.getenv("NEG_CRITIC_ENABLED", "1") != "0"
NEG_CRITIC_BASE_URL = os.getenv(
    "NEG_CRITIC_BASE_URL",
    "http://127.0.0.1:8001/v1",
)
NEG_CRITIC_MODEL = os.getenv("NEG_CRITIC_MODEL", "Qwen3-32B")
NEG_CRITIC_API_KEY = os.getenv("NEG_CRITIC_API_KEY", "EMPTY")
NEG_CRITIC_TIMEOUT = float(os.getenv("NEG_CRITIC_TIMEOUT", "25"))
NEG_EVAL_EVERY_STEPS = int(os.getenv("NEG_EVAL_EVERY_STEPS", "3"))
NEG_EVAL_AFTER_TOOL = os.getenv("NEG_EVAL_AFTER_TOOL", "1") != "0"
NEG_EVAL_MIN_STEP = int(os.getenv("NEG_EVAL_MIN_STEP", "2"))
NEG_EVAL_MAX_WINDOW = int(os.getenv("NEG_EVAL_MAX_WINDOW", "10"))
FORCE_ANSWER_STEP = int(os.getenv("NEG_FORCE_ANSWER_STEP", "13"))


SYSTEM_PROMPT = """你是一个精确的推理搜索 Agent。

要求：
1) 每步只做一个关键动作：调用工具，或在证据充足时直接回答。
2) 使用工具时必须是标准 function/tool_calls，不要在文本中伪造工具调用。
3) 遇到失败信号（空结果、403、captcha、噪音域名）要及时换线索，不要重复同义查询。
4) 最终答案必须使用 <answer>...</answer>。
"""


NEG_CRITIC_SYSTEM_PROMPT = """You are an online NEGATIVE evaluator for an agent trajectory.

Hard constraints:
- Output STRICT JSON only.
- Allowed schema exactly:
  {"judgment":"GOOD|BAD","reason_short":"..."}
- judgment must be either GOOD or BAD.
- reason_short must be concise and diagnostic.
- NEVER provide action suggestions, tool suggestions, search keywords, or next steps.
- NEVER use phrases like "should", "try", "you can", "建议", "应该", "可以去".
- You are only a brake signal and fault locator.
"""


def _clip(text: str, limit: int = 400) -> str:
    s = str(text or "")
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def _safe_json_load(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        val = json.loads(raw)
        if isinstance(val, dict):
            return val
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        val = json.loads(m.group(0))
        if isinstance(val, dict):
            return val
    except Exception:
        return None
    return None


def _normalize_neg_judgment(obj: dict[str, Any] | None) -> tuple[str, str]:
    if not obj:
        return "GOOD", "critic_parse_failed"
    judgment = str(obj.get("judgment", "")).strip().upper()
    reason = str(obj.get("reason_short", "")).strip()
    if judgment not in {"GOOD", "BAD"}:
        judgment = "GOOD"
        reason = "invalid_judgment_fallback"
    return judgment, reason or "no_reason"


def _recent_trajectory_window(entries: list[dict[str, Any]], limit: int) -> list[dict[str, str]]:
    rows = []
    for e in entries[-limit:]:
        role = str(e.get("role", ""))
        content = _clip(str(e.get("content", "")), 500)
        if not content:
            continue
        rows.append({"role": role, "content": content})
    return rows


def _run_negative_critic(
    critic_client: OpenAI,
    instruction: str,
    task_id: str,
    entries: list[dict[str, Any]],
) -> tuple[str, str]:
    payload = {
        "task_id": task_id,
        "instruction": _clip(instruction, 500),
        "recent_trajectory": _recent_trajectory_window(entries, NEG_EVAL_MAX_WINDOW),
        "note": "Judge only if current strategy is on-track or off-track. No actions.",
    }
    try:
        resp = critic_client.chat.completions.create(
            model=NEG_CRITIC_MODEL,
            messages=[
                {"role": "system", "content": NEG_CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=220,
            timeout=NEG_CRITIC_TIMEOUT,
        )
        content = resp.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("Negative critic call failed: %s", exc)
        return "GOOD", "critic_unavailable"
    obj = _safe_json_load(content)
    return _normalize_neg_judgment(obj)


def _forced_self_reflection_prompt(reason_short: str) -> str:
    return (
        "[FORCED_REFLECTION]\n"
        "External evaluator judged your current trajectory as BAD.\n"
        f"Reason: {reason_short}\n"
        "You MUST pause the current strategy and self-correct NOW.\n"
        "Rules:\n"
        "1) First output a short internal correction note (why current line failed).\n"
        "2) Then state a revised plan with different constraints/entity chain.\n"
        "3) Then continue with exactly one concrete next action (tool call or final answer).\n"
        "4) Do not repeat the previous near-duplicate query direction."
    )


def run_task(
    task: dict,
    max_steps: int = MAX_STEPS,
    llm_base_url: str = LLM_BASE_URL,
    model_name: str = MODEL_NAME,
    trajectory_dir: str = "trajectories",
) -> dict:
    task_id = task.get("id") or str(uuid.uuid4())[:8]
    instruction = task["instruction"]
    image_b64 = task.get("image_b64")
    image_url = task.get("image_url")

    logger.info("run_task [plan&react+negcritic]: task_id=%s", task_id)
    traj = Trajectory(task_id, output_dir=trajectory_dir)
    main_client = OpenAI(base_url=llm_base_url, api_key="EMPTY")
    critic_client = OpenAI(base_url=NEG_CRITIC_BASE_URL, api_key=NEG_CRITIC_API_KEY or "EMPTY")

    traj.write(Role.SYSTEM, SYSTEM_PROMPT, step_id=0)
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

    final_answer = ""
    force_answer_injected = False
    recent_queries: list[set] = []
    low_signal_streak = 0
    pending_forced_reflection: str | None = None

    for step in range(1, max_steps + 1):
        logger.info("--- step %d/%d ---", step, max_steps)

        if pending_forced_reflection:
            traj.write(Role.USER, _forced_self_reflection_prompt(pending_forced_reflection), step_id=step)
            pending_forced_reflection = None

        if step >= FORCE_ANSWER_STEP and not force_answer_injected:
            traj.write(
                Role.USER,
                "[SYSTEM] Step budget nearly exhausted. If evidence is enough, output <answer>...</answer> now. "
                "If evidence is weak, output <answer confidence=\"low\">...</answer>. Avoid further wandering search.",
                step_id=step,
            )
            force_answer_injected = True

        messages = traj.to_messages()
        request_kwargs = dict(
            model=model_name,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.6,
            extra_body={"enable_thinking": True},
        )
        if not DISABLE_TOOLS and not force_answer_injected:
            request_kwargs["tools"] = TOOLS_SCHEMA
            request_kwargs["tool_choice"] = "auto"
        try:
            response = main_client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            logger.error("Main LLM call failed: %s", exc, exc_info=True)
            traj.write(Role.TOOL, f"[HARNESS ERROR] Main LLM call failed: {exc}", step_id=step)
            break

        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        reasoning_content = getattr(msg, "reasoning_content", "") or ""
        tool_calls = None if DISABLE_TOOLS else msg.tool_calls
        if not tool_calls and not content and reasoning_content:
            leaked = parse_leaked_tool_calls(reasoning_content)
            if leaked:
                tool_calls = leaked
                logger.info("Recovered %d leaked tool call(s)", len(leaked))

        extra = {}
        if tool_calls:
            if hasattr(tool_calls[0], "model_dump"):
                extra["tool_calls"] = [tc.model_dump() for tc in tool_calls]
            else:
                extra["tool_calls"] = tool_calls
        if reasoning_content:
            extra["reasoning_content"] = reasoning_content
        traj.write(Role.ASSISTANT, content, step_id=step, extra=extra if extra else None)
        logger.info("finish_reason=%s, has_tool_calls=%s", choice.finish_reason, bool(tool_calls))

        if not tool_calls and content:
            final_answer = extract_answer(content)
            logger.info("Task complete at step %d", step)
            break
        if not tool_calls and not content:
            continue

        if step == max_steps and tool_calls:
            forced = '<answer confidence="low">Unable to determine from available evidence</answer>'
            traj.write(Role.ASSISTANT, forced, step_id=step)
            final_answer = extract_answer(forced)
            break

        def _exec_one_tool(tc):
            if hasattr(tc, "function"):
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
            logger.info("tool_call: %s(%s)", fn_name, fn_args)
            if fn_name not in TOOL_FN_MAP:
                return tc_id, fn_name, fn_args, f"[ERROR] Unknown tool: {fn_name}"
            try:
                raw = TOOL_FN_MAP[fn_name](fn_args)
                if isinstance(raw, (dict, list)):
                    tool_result = json.dumps(raw, ensure_ascii=False)
                else:
                    tool_result = str(raw)
            except Exception as exc:
                tool_result = f"[ERROR] Tool '{fn_name}' raised: {type(exc).__name__}: {exc}"
            return tc_id, fn_name, fn_args, tool_result

        pending = tool_calls or []
        if len(pending) > 1:
            with ThreadPoolExecutor(max_workers=min(len(pending), 4)) as pool:
                futures = [pool.submit(_exec_one_tool, tc) for tc in pending]
                results = [f.result() for f in futures]
        else:
            results = [_exec_one_tool(tc) for tc in pending]

        tool_executed_this_step = False
        for tc_id, fn_name, fn_args, tool_result in results:
            tool_executed_this_step = True
            if fn_name == "search_text":
                tool_result, has_signal = filter_garbage_results(tool_result)
                low_signal_streak = 0 if has_signal else (low_signal_streak + 1)
                q = str(fn_args.get("query", "")).lower()
                kw = {w for w in re.sub(r'["\'\(\)]', " ", q).split() if len(w) > 2}
                if kw:
                    recent_queries.append(kw)
            traj.write(
                Role.TOOL,
                tool_result,
                step_id=step,
                tool_call_id=tc_id,
                extra={"fn_name": fn_name, "fn_args": fn_args},
            )

        should_eval = (
            NEG_CRITIC_ENABLED
            and step >= NEG_EVAL_MIN_STEP
            and (
                (NEG_EVAL_AFTER_TOOL and tool_executed_this_step)
                or (NEG_EVAL_EVERY_STEPS > 0 and step % NEG_EVAL_EVERY_STEPS == 0)
            )
        )
        if should_eval:
            entries = traj.read_all()
            judgment, reason_short = _run_negative_critic(
                critic_client=critic_client,
                instruction=instruction,
                task_id=task_id,
                entries=entries,
            )
            traj.write(
                Role.USER,
                f"[NEG_CRITIC] {json.dumps({'judgment': judgment, 'reason_short': reason_short}, ensure_ascii=False)}",
                step_id=step,
                extra={"neg_critic": True, "judgment": judgment},
            )
            logger.info("NEG_CRITIC judgment=%s reason=%s", judgment, reason_short)
            if judgment == "BAD":
                pending_forced_reflection = reason_short

        if low_signal_streak >= 2:
            traj.write(
                Role.USER,
                "[SYSTEM] Recent search results are low-signal/noisy. Switch to a different clue chain now.",
                step_id=step,
            )

    else:
        logger.warning("Reached max_steps=%d", max_steps)
        entries = traj.read_all()
        for e in reversed(entries):
            if e["role"] == "assistant" and e.get("content"):
                final_answer = extract_answer(e["content"])
                break
        if not final_answer:
            final_answer = "[LOW_CONFIDENCE] Unable to determine from available evidence"

    summary = traj.summary()
    raw_answer = final_answer
    cleaned_answer = raw_answer
    return {
        "task_id": task_id,
        "answer": cleaned_answer,
        "raw_answer": raw_answer,
        "steps": step,
        "trajectory_path": str(traj.path),
        "summary": summary,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen Agent Harness (Plan & ReAct + Online Negative Critic)",
    )
    p.add_argument("--instruction", "-i", required=True, help="Task instruction text")
    p.add_argument("--task-id", "-t", default=None, help="Optional task ID")
    p.add_argument("--max-steps", "-s", type=int, default=MAX_STEPS)
    p.add_argument("--llm-url", default=LLM_BASE_URL)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--traj-dir", default="trajectories")
    p.add_argument("--image", default=None, help="Local image path")
    p.add_argument("--image-url", default=None, help="Online image URL")
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
        "image_b64": image_b64,
        "image_url": args.image_url,
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
