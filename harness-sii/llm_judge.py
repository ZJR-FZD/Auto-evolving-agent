"""LLM-based reflection, judging, and lesson extraction using an external model.

Uses Qwen3-32B (or configurable external model) for:
- Post-task reflection: structured diagnosis of failures
- LLM-as-Judge: confidence scoring without gold answers
- Lesson extraction: generating transferable insights from trajectories

The external model is NOT the base agent model — it serves as a "teacher"
that helps the agent evolve by providing deeper analysis.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from openai import OpenAI

logger = logging.getLogger("harness.llm_judge")

JUDGE_LLM_URL = os.getenv("JUDGE_LLM_URL", "http://127.0.0.1:8001/v1")
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "Qwen3-32B")
JUDGE_MAX_TOKENS = int(os.getenv("JUDGE_MAX_TOKENS", "4096"))
JUDGE_TIMEOUT = float(os.getenv("JUDGE_TIMEOUT", "60"))
JUDGE_ENABLED = os.getenv("JUDGE_ENABLED", "1") != "0"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=JUDGE_LLM_URL, api_key="EMPTY", timeout=JUDGE_TIMEOUT)
    return _client


def _call_judge(system: str, user: str, max_tokens: int = JUDGE_MAX_TOKENS) -> str:
    if not JUDGE_ENABLED:
        return ""
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=JUDGE_MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("Judge LLM call failed: %s", exc)
        return ""


def _parse_json_from_response(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data:
            return data[0] if isinstance(data[0], dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


# ---------------------------------------------------------------------------
# LLM-as-Judge: evaluate answer confidence without gold answer
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
你是一个答案质量评估专家。你需要根据问题、推理过程和最终答案，判断答案是否可能正确。
注意：你没有标准答案，只能根据推理逻辑和证据充分性来判断。

评估维度：
1. 证据充分性：推理过程中是否找到了支持答案的可靠证据？
2. 逻辑一致性：答案是否与搜索到的信息一致？
3. 答案规范性：答案是否简洁、直接回答了问题？
4. 推理效率：是否存在无效循环或重复搜索？

返回JSON格式：
{"confidence": 0.0-1.0, "reasoning": "简短评估理由", "suggestion": "如果置信度低，给出改进建议"}
"""


def judge_answer(
    question: str,
    trajectory_summary: str,
    final_answer: str,
    task_family: str = "",
) -> dict[str, Any]:
    """Evaluate answer quality without gold answer (LLM-as-Judge).

    Returns: {"confidence": float, "reasoning": str, "suggestion": str}
    """
    if not JUDGE_ENABLED or not final_answer.strip():
        return {"confidence": 0.0, "reasoning": "judge disabled or empty answer", "suggestion": ""}

    user_msg = (
        f"## 任务类型\n{task_family or 'generic'}\n\n"
        f"## 问题\n{question[:500]}\n\n"
        f"## 推理过程摘要\n{trajectory_summary[:2000]}\n\n"
        f"## 最终答案\n{final_answer[:300]}\n\n"
        "请评估这个答案的可信度，返回JSON。"
    )
    raw = _call_judge(_JUDGE_SYSTEM, user_msg, max_tokens=512)
    result = _parse_json_from_response(raw)
    return {
        "confidence": float(result.get("confidence", 0.5)),
        "reasoning": str(result.get("reasoning", "")),
        "suggestion": str(result.get("suggestion", "")),
    }


# ---------------------------------------------------------------------------
# Post-task reflection: structured diagnosis of failures
# ---------------------------------------------------------------------------

_REFLECT_SYSTEM = """\
你是一个智能体行为分析专家。分析以下任务执行轨迹，诊断失败原因并提取可迁移的经验教训。

要求：
1. 识别根本原因（不是表面现象）
2. 提取可以帮助未来类似任务的策略
3. 指出应该避免的行为模式
4. 经验必须是抽象的、可迁移的，不能包含具体答案

返回JSON格式：
{
  "root_cause": "失败的根本原因",
  "lesson": "可迁移的经验教训（不超过100字）",
  "strategy": "建议的改进策略（不超过100字）",
  "avoid": "应避免的行为（不超过80字）",
  "category": "failure_type: loop|wrong_entity|insufficient_evidence|tool_misuse|format_error",
  "confidence": 0.0-1.0
}
"""


def reflect_on_trajectory(
    question: str,
    trajectory_summary: str,
    final_answer: str,
    task_family: str = "",
    correct: bool | None = None,
) -> dict[str, Any]:
    """Analyze a task trajectory and extract structured lessons.

    Returns structured reflection with root_cause, lesson, strategy, avoid.
    """
    if not JUDGE_ENABLED:
        return {}

    outcome = "未知"
    if correct is True:
        outcome = "正确"
    elif correct is False:
        outcome = "错误"

    user_msg = (
        f"## 任务类型\n{task_family or 'generic'}\n\n"
        f"## 问题\n{question[:500]}\n\n"
        f"## 执行轨迹摘要\n{trajectory_summary[:3000]}\n\n"
        f"## 最终答案\n{final_answer[:300]}\n\n"
        f"## 结果\n{outcome}\n\n"
        "请分析并返回JSON。"
    )
    raw = _call_judge(_REFLECT_SYSTEM, user_msg, max_tokens=800)
    result = _parse_json_from_response(raw)
    if not result.get("lesson"):
        return {}
    return {
        "root_cause": str(result.get("root_cause", "")),
        "lesson": str(result.get("lesson", "")),
        "strategy": str(result.get("strategy", "")),
        "avoid": str(result.get("avoid", "")),
        "category": str(result.get("category", "unknown")),
        "confidence": max(0.3, min(1.0, float(result.get("confidence", 0.6)))),
    }


# ---------------------------------------------------------------------------
# LLM-based lesson extraction (batch mode for memory updates)
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """\
你是一个经验提取专家。从智能体的任务执行记录中提取可迁移的经验教训。

规则：
- 经验必须是抽象的，不能包含具体答案或具体实体名
- 经验必须对未来类似任务有帮助
- 避免过于笼统的建议（如"要仔细搜索"）
- 关注策略层面的洞察（如"比较题应先分别查询两个实体的属性再比较"）

返回JSON数组，每个元素：
{"lesson": "...", "strategy": "...", "avoid": "...", "category": "...", "confidence": 0.0-1.0}
最多返回2条最有价值的经验。
"""


def extract_lessons_llm(
    question: str,
    trajectory_summary: str,
    final_answer: str,
    task_family: str = "",
    correct: bool | None = None,
) -> list[dict[str, Any]]:
    """Extract transferable lessons from a task execution using LLM."""
    if not JUDGE_ENABLED:
        return []

    outcome = "正确" if correct else ("错误" if correct is False else "未知")
    user_msg = (
        f"任务类型: {task_family}\n问题: {question[:400]}\n"
        f"轨迹摘要: {trajectory_summary[:2500]}\n"
        f"最终答案: {final_answer[:200]}\n结果: {outcome}\n"
        "请提取经验教训，返回JSON数组。"
    )
    raw = _call_judge(_EXTRACT_SYSTEM, user_msg, max_tokens=800)
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data[:2] if isinstance(d, dict) and d.get("lesson")]
        if isinstance(data, dict) and data.get("lesson"):
            return [data]
    except json.JSONDecodeError:
        pass
    return []
