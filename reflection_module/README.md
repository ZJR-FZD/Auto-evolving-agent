# Reflection Module

该模块为 `harness-sii` 提供 harness-level Reflection：主任务仍由 `Qwen3.5-9B` 执行，反思模块只在失败后分析轨迹、定位原因、生成修正策略并沉淀可复用经验。LLM 路径中，`Qwen3-30B-A3B` 被用作 **open_source_reflection_critic / failure-analysis agent**，不是答题模型。

## Why Not Prompt-Only Reflection

| 问题 | 设计选择 |
| --- | --- |
| 基模本身可能会“口头反思”，但仍重复失败动作 | 在 runner 层检测真实失败信号，例如工具错误、空动作、伪工具调用、超时、`max_steps`。 |
| prompt-level self-critique 容易凭空归因 | critic 输入只包含 trajectory evidence、failure event、工具摘要和可选图像上下文。 |
| 反思经验可能积累噪声 | memory 写入前经过 validator、置信度过滤、空 lesson 过滤和相似 lesson 去重。 |
| 强模型可能直接代答 | critic prompt 和 validator 都禁止输出最终答案、`answer/gold` 字段或 “the answer is ...” 模板。 |

## Related Work and Rationale

| 思想来源 | 借鉴点 | 本项目落地 |
| --- | --- | --- |
| Reflexion | 语言化失败经验与 episodic memory | `memory_lesson` 写入 JSONL memory，后续任务按 task type 和关键词检索。 |
| Self-Refine | `feedback -> refine -> retry` 闭环 | 失败后注入简短 reflection feedback，让主 agent 先修正策略再继续。 |
| ReAct | reasoning 与 action 必须对齐 | 检测 reasoning 中伪 `<tool_call>` 但没有真实 OpenAI `tool_calls` 的失败。 |
| LATS / Agent-R / Re-ReST / RISE | 失败修正应基于真实轨迹反馈 | critic 只能看 observed trajectory evidence，不看 gold answer。 |
| MAST / failure taxonomy | 结构化错误归因 | failure type 覆盖 tool-use、execution、termination、format、visual grounding 等类型。 |
| Reflective Memory Management | 记忆要过滤、去重、检索和使用计数 | schema 增加 `memory_id`、`task_type`、`critic_model`、`confidence`、`use_count`。 |
| AutoResearchClaw 工程思想 | self-learning lessons、knowledge base、watchdog、quality gate | 轻量化为 failure detector + critic + validator + JSONL memory，不照搬其框架。 |

## Qwen3-30B-A3B Critic

`Qwen3-30B-A3B` 只作为 open_source_reflection_critic / failure-analysis agent：

| 允许 | 禁止 |
| --- | --- |
| 分析失败原因 | 直接回答原始问题 |
| 引用 trajectory evidence | 使用或推断 gold answer |
| 生成修正策略和下一步提示 | 修改 `pred` 或绕过主 agent |
| 生成可复用 lesson | 充当 2Wiki/SimpleVQA 主答题模型 |

critic 输出必须是 strict JSON，字段包括 `failure_type`、`root_cause`、`evidence`、`correction_strategy`、`next_prompt`、`next_action_type`、`memory_lesson`、`confidence` 等。输出会经过 `validate_critic_output`，失败、超时或 JSON 不合法时自动回退到 `rule_fallback`。

## Task Types

| task_type | 策略重点 |
| --- | --- |
| `2wiki_text` | 实体拆分、多跳关系链、query rewrite、证据链完整性、最终答案格式。 |
| `simplevqa_multimodal` | 图像关键实体/场景识别、`image_url` 是否可用、`search_image`/browser 验证、视觉描述与搜索证据一致性。 |
| `general` | 非特定任务的工具错误、空动作、超时和步数耗尽恢复。 |

## Reflection Skill Wrapper

`reflection_module/reflection_skill.py` 提供一个可直接接入 ReAct runner 的轻量 Skill 层。它不替代 `core.py`，而是把 failure trajectory database、memory reuse、conditional critic 和 async reflection 串起来。

| 组件 | 文件 / 类 | 作用 |
| --- | --- | --- |
| Memory-augmented skill | `ReflectionSkill` | 失败后统一入口：先查失败库，再查 memory/9B/30B/rule。 |
| 失败轨迹数据库 | `FailureTrajectoryDB` | JSONL 存储每个失败 case 的工具历史、推理链、失败类型和任务特征。 |
| 轨迹记录 schema | `FailureTrajectoryRecord` | 包含 `tool_history`、`reasoning_chain`、`final_failure`、`correction_strategy`。 |
| 条件深度反思 | `ReflectionManager` | 无高匹配经验时才调用 `Qwen3-30B-A3B`，失败则 `rule_fallback`。 |
| 异步反思 | `REFLECTION_ASYNC_CRITIC=1` | 当前任务先拿快速策略继续，30B 在后台沉淀经验给后续任务。 |

典型接入方式：

```python
from reflection_module import ReflectionSkill
from reflection_module.reflection_skill import build_failure_event_from_runner

skill = ReflectionSkill()

# task start: 注入 task-type aware policy + memory hints
system_prompt += skill.task_prompt_appendix(task_type)
system_prompt += skill.on_task_start(instruction, task_type)

# failure hook: runner 检测到真实失败后调用
event = build_failure_event_from_runner(
    task_id=task_id,
    step_id=step_id,
    instruction=instruction,
    task_type=task_type,
    failure_type="tool_timeout",
    evidence=tool_result,
    trajectory_rows=trajectory_rows,
    tool_name="search_text",
    tool_args={"query": query},
)
decision = skill.on_failure(event, trajectory_rows)
feedback = skill.feedback_message(decision)
messages.append({"role": "user", "content": feedback})
```

执行流程：

| 阶段 | 动作 | 输出 |
| --- | --- | --- |
| Failure Detection | Harness 根据工具错误、空动作、循环、max-step、格式异常触发 | `FailureEvent` |
| Failure DB Match | `FailureTrajectoryDB.query()` 匹配相似失败轨迹 | 可复用策略或空 |
| Strategy Reuse | 高匹配时直接复用 `correction_strategy` | `failure_db_reuse` |
| Deep Reflection | 低匹配时进入 memory / 9B light critic / 30B critic / rule fallback | `ReflectionRecord` |
| Memory/DB Write | 写入 JSONL memory 和 failure trajectory DB | 后续任务可检索 |
| Prompt Injection | 将 feedback 注入 ReAct loop | 主 agent 修正下一步策略 |

## Environment Variables

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ENABLE_REFLECTION` | `1` | 总开关。 |
| `REFLECTION_USE_LLM` | `0` | 设为 `1` 启用 LLM critic。 |
| `REFLECTION_MODEL` | `Qwen3-30B-A3B` | 独立开源 reflection critic 模型，不影响主模型。 |
| `REFLECTION_BASE_URL` | `http://127.0.0.1:8001/v1` | 开源 critic 的 OpenAI-compatible endpoint。 |
| `REFLECTION_BASE_URL` | 空 | 可选 OpenAI-compatible endpoint。 |
| `REFLECTION_API_KEY` | 空 | critic API key，不写死到代码。缺失时自动 fallback。 |
| `REFLECTION_TIMEOUT` | `30` | critic 调用超时秒数。 |
| `REFLECTION_MAX_CONTEXT_CHARS` | `6000` | 传给 critic 的上下文最大长度。 |
| `REFLECTION_MAX_MEMORY` | `4` | 每个新任务最多注入的 memory 条数。 |
| `REFLECTION_ALLOW_VISION_CONTEXT` | `1` | 是否允许向 critic 提供 image path/url/已有描述等上下文。 |
| `MODEL_NAME` | `Qwen3.5-9B` | 主任务模型应保持为 Qwen3.5-9B。 |
| `REFLECTION_FAILURE_DB_PATH` | `reflection_memory/failure_trajectories.jsonl` | Reflection Skill 的失败轨迹数据库。 |
| `REFLECTION_SKILL_REUSE_THRESHOLD` | `0.42` | 失败轨迹策略复用阈值。 |
| `REFLECTION_ASYNC_CRITIC` | `0` | 设为 `1` 启用后台 30B 反思。 |
| `REFLECTION_FORCE_CRITIC_ON_FAILURE_TYPES` | `budget_exhausted,answer_format_error,insufficient_evidence` | 这些 hard failures 会绕过泛化 memory，优先触发 30B critic。 |
| `REFLECTION_CRITIC_ASYNC_LOG_PATH` | `reflection_memory/critic_async_log.jsonl` | 记录 30B critic 成功调用、置信度和 memory 写入状态。 |
| `TEXT_BOUNDED_MODE` | `1` | 文本多跳题启用 bounded mode。 |
| `TEXT_FORCE_ANSWER_STEP` | `6` | 文本题中途强制生成候选答案的步数。 |
| `TEXT_SEARCH_FAILURE_LIMIT` | `2` | 搜索代理连续失败后触发文本熔断 synthesis。 |

## Offline Answer Cleaning

`answer_utils.py` 提供 gold-free 的最终答案抽取逻辑，只使用 `prediction` 和可选 `question`，不读取 gold。`reevaluate_2wiki_cleaning.py` 用于离线复评清洗收益，gold 只用于计算 EM / contains / F1 和诊断，不参与生成 cleaned prediction。

```bash
python reevaluate_2wiki_cleaning.py \
  --results results/2wiki_text_eval_with_gold_100.jsonl \
  --out results/2wiki_text_eval_with_gold_100_cleaned.jsonl \
  --report results/2wiki_text_eval_with_gold_100_cleaning_report.json
```

当前 100 条 2Wiki 离线复评示例：

| 指标 | 原始预测 | 清洗后 |
| --- | ---: | ---: |
| Exact Match | 0.40 | 0.52 |
| Contains Accuracy | 0.72 | 0.74 |
| Avg F1 | 0.5674 | 0.6613 |
| EM 回退样本 | - | 0 |

## Run Commands

2Wiki Qwen3-30B-A3B critic reflection：

```bash
export REFLECTION_API_KEY=EMPTY
python run_reflection_2wiki_qwen3_30b_a3b.py
```

输出：

```text
results/reflection_qwen3_30b_a3b_2wiki_full.jsonl
trajectories_reflection_qwen3_30b_a3b_2wiki_full/
```

SimpleVQA Qwen3-30B-A3B critic reflection：

```bash
export REFLECTION_API_KEY=EMPTY
python run_reflection_simplevqa_qwen3_30b_a3b.py
```

输出：

```text
results/reflection_qwen3_30b_a3b_simplevqa_full.jsonl
trajectories_reflection_qwen3_30b_a3b_simplevqa_full/
```

critic 服务不可用、超时或输出 JSON 不合法时会自动降级到 `rule_fallback`，不会静默伪装成 Qwen3-30B-A3B 成功。`Qwen3-14B` 可作为资源不足时的开源备选。

## Experiments

建议三组消融：

| 组别 | 主模型 | 反思模块 |
| --- | --- | --- |
| A. baseline | Qwen3.5-9B | 无 |
| B. rule reflection | Qwen3.5-9B | harness-level detector + rule fallback |
| C. Qwen3-30B-A3B critic reflection | Qwen3.5-9B | detector + open_source_reflection_critic + validator + fallback |

比较脚本：

```bash
python -m reflection_module.compare_runs \
  --baseline-results results/baseline_2wiki_full.jsonl \
  --rule-results results/reflection_2wiki_full.jsonl \
  --critic-results results/reflection_qwen3_30b_a3b_2wiki_full.jsonl \
  --out reflection_module/reflection_compare_report.json
```

指标包括 accuracy、avg_steps、tool_error_rate、max_steps_rate、malformed_tool_call_rate、empty_assistant_rate、reflection_trigger_count、critic_success_count、critic_fallback_count、memory_hit_count、recovery_after_reflection_rate。gold answer 只用于离线 accuracy 统计，不能用于触发 reflection。
