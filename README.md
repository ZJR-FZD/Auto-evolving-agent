# 自进化的任务求解智能体

本项目实现了一个基于 Harness 工程的任务求解智能体系统，面向联网多跳问答、图文问答和闭源打榜任务。系统围绕 `尝试 -> 反思 -> 策略修正 -> 再尝试` 的闭环设计，支持 ReAct 工具调用、轨迹记录、在线反思、轻量任务路由、搜索策略引导、答案兜底清洗，以及 SimpleVQA、2Wiki 和 benchmark 打榜数据的批量评测。

本仓库使用 Qwen3.5-9B 作为 Harness 基座模型，兼容 OpenAI SDK 风格接口，可接入 vLLM/SGLang 等 OpenAI-compatible 推理服务。

## 课题背景

当前大多数 LLM 应用仍停留在单轮或多轮对话层面，缺少持续行动、工具调用和失败后自我修正能力。一个更有价值的任务求解智能体应当具备以下能力：

| 能力 | 说明 |
|---|---|
| 自主规划 | 将复杂问题拆解为可执行子步骤，确定搜索、浏览、验证、作答顺序。 |
| 工具调用 | 通过联网搜索、图片搜索、浏览器导航、网页文本读取等工具补充外部证据。 |
| 过程反思 | 在低质量搜索、重复检索、候选冲突、角色混淆时触发反思，而不是无脑重试。 |
| 策略进化 | 将失败模式沉淀为通用控制策略，减少后续任务中的重复错误。 |
| 可评测闭环 | 通过轨迹、结果、耗时、工具调用次数等文件化输出分析智能体行为。 |

## 项目目标

| 目标 | 本项目对应实现 |
|---|---|
| 搭建最小 ReAct 智能体 | `task_runner.py`、`task_runner_plan_react.py`、`task_runner_plan_react_negcrit.py` |
| 支持多轮工具调用 | LLM 输出 tool call，Harness 执行工具并写回结果，循环直到最终答案或步数上限。 |
| 构建搜索和浏览器工具 | `tools/search_tool.py`、`tools/browser_tool.py`、`sandbox_client.py` |
| 添加反思模块 | negative critic、forced reflection、soft constraint ledger、route replan、blocked/search budget control |
| 添加轻量记忆/策略沉淀 | `tpgo/` 下的轨迹分析、失败标签、搜索模板、静态蒸馏策略、合规文档 |
| 评测 SimpleVQA 和 2Wiki | `run_dataset_eval.py` 输出结果 JSONL 和轨迹 JSONL |
| 公开打榜 | `run_benchmark.py`、`run_benchmark_parallel_negcrit.py`、`run_benchmark_parallel_mixed.py` |
| 输出可审计轨迹 | `trajectory.py` 将每一步 system/user/assistant/tool 写成 JSONL |

## 代码结构

```text
Auto-evolving-agent/
├── task_runner.py                         # 基础 ReAct 智能体
├── task_runner_plan_react.py              # Plan + ReAct + Reflection 智能体
├── task_runner_plan_react_negcrit.py      # ReAct + 在线负反馈 critic + TPGO 引导
├── trajectory.py                          # 轨迹记录器，逐步写 JSONL
├── roles.py                               # 消息角色定义
├── sandbox_client.py                      # 浏览器沙箱 HTTP 客户端
├── eval_metrics.py                        # 评测指标工具
├── run.sh                                 # 旧版统一入口
├── run_benchmark.py                       # benchmark 通用 runner
├── run_benchmark_parallel_negcrit.py      # negcrit benchmark 并发入口
├── run_benchmark_parallel_mixed.py        # 文本题/图片题隔离 mixed runner
├── run_dataset_eval.py                    # SimpleVQA/2Wiki 统一评测入口
├── run_simpleqa.py                        # 旧版 SimpleVQA runner
├── run_2wiki.py                           # 旧版 2Wiki runner
├── tools/
│   ├── search_tool.py                     # 文本搜索、图片搜索封装
│   └── browser_tool.py                    # 浏览器导航、文本读取、点击、输入、并发浏览
├── tpgo/
│   ├── task_router.py                     # 轻量任务类型判定
│   ├── search_templates.py                # 按题型生成搜索模板
│   ├── reflection_tags.py                 # 失败标签推断
│   ├── soft_constraints.py                # 约束账本、反思提示、重复查询控制
│   ├── answer_fallback.py                 # 低置信/空答案兜底抽取
│   ├── distilled_strategy.py              # 合规静态蒸馏策略提示
│   ├── tpgo_tools.py                      # 轨迹分析、反思记忆抽取、图生成等工具
│   ├── ablation_runner.py                 # 消融实验 runner
│   ├── README.md                          # TPGO 模块说明
│   └── DISTILLATION_README.md             # 蒸馏合规说明
├── tests/
│   └── test_task_router.py                # 任务路由单元测试
├── search-proxy/                          # 搜索代理服务
├── results/                               # 评测结果输出目录，默认不提交
└── trajectories/                          # 每题轨迹输出目录，默认不提交
```

## 智能体范式

| 范式 | 入口 | 核心机制 | 适用场景 |
|---|---|---|---|
| `basic` | `task_runner.py` | 标准 ReAct 循环 | 简单搜索任务、最小基线 |
| `plan_react` | `task_runner_plan_react.py` | 规划、候选投票、状态机、反思、强制作答 | 图文题、SimpleVQA、复杂任务 |
| `plan_react_negcrit` | `task_runner_plan_react_negcrit.py` | ReAct、negative critic、TPGO 路由、软约束、搜索预算、答案兜底 | 前 50 文本 benchmark、多跳网页检索 |
| `mixed` | `run_benchmark_parallel_mixed.py` | 前 50 文本题走 negcrit，后 50 图片题走旧 plan_react | benchmark 打榜隔离策略 |

### ReAct 基础循环

系统的核心执行方式如下：

1. 用户输入任务和可选图片。
2. LLM 分析任务并输出 tool call。
3. Harness 执行工具，如 `search_text`、`search_image`、`browser_navigate`。
4. 工具结果写入轨迹。
5. LLM 基于新证据决定继续调用工具或输出 `<answer>...</answer>`。
6. 达到最大步数、搜索预算或工具失败阈值时触发强制作答。

## 工具模块

| 工具类型 | 文件 | 已实现能力 |
|---|---|---|
| 文本搜索 | `tools/search_tool.py` | 在线文本搜索，返回 title、url、snippet、content。 |
| 图片搜索 | `tools/search_tool.py` | 基于图像 URL 或相关图像信息进行反向搜索。 |
| 浏览器访问 | `tools/browser_tool.py` | 打开页面、读取页面文本。 |
| 浏览器交互 | `tools/browser_tool.py` | 点击、输入、并发处理多个页面。 |
| 沙箱客户端 | `sandbox_client.py` | 对接浏览器服务，封装 HTTP 请求和会话。 |

工具调用遵循 OpenAI function calling 风格，轨迹中会记录 `tool_calls`、`fn_name`、`fn_args` 和工具返回内容，便于后续评估工具调用质量。

## 反思与进化模块

### 1. 在线负反馈 Critic

`task_runner_plan_react_negcrit.py` 支持一个外部 critic 对近期轨迹做 `GOOD/BAD` 判断。critic 只允许输出方向性诊断，不允许直接给答案。

关键配置：

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `NEG_CRITIC_ENABLED` | `1` | 是否启用在线 critic。合规静态蒸馏评测时应设为 `0`。 |
| `NEG_CRITIC_BASE_URL` | `http://127.0.0.1:8001/v1` | critic 模型服务地址。 |
| `NEG_CRITIC_MODEL` | `Qwen3-32B` | critic 模型名。 |
| `NEG_EVAL_EVERY_STEPS` | `3` | 每隔多少步评估一次。 |
| `NEG_EVAL_AFTER_TOOL` | `1` | 工具调用后是否立即评估。 |

### 2. Soft Constraint Ledger

`tpgo/soft_constraints.py` 提供通用反思规则：

| 机制 | 作用 |
|---|---|
| 约束账本 | 区分强约束和弱主题匹配，避免只因为主题相似就提交答案。 |
| 答案角色检查 | 防止把 donor、parent、author、source、host 等相关实体误当作目标答案。 |
| 重复查询控制 | 对 near-duplicate query 做拦截，要求换 clue chain 或增加真实 pivot。 |
| 强制作答提示 | 接近预算上限时只输出短答案，不输出过程说明。 |

### 3. Task Router

`tpgo/task_router.py` 是一个不直接答题的轻量题型判定器。它只根据 question 生成：

| 字段 | 含义 |
|---|---|
| `task_type` | 任务类型，如 `company_finance`、`person_bio_family`、`film_tv_game` 等。 |
| `search_keywords` | 初始搜索关键词。 |
| `preferred_sources` | 推荐优先来源，如 SEC、IMDb、官方报告、学术数据库等。 |
| `risk_tags` | 风险标签，如时间约束、实体重名、数字抽取等。 |
| `answer_format_hint` | 答案格式提示，如人名、年份、标题、数值。 |

支持题型：

| task_type | 说明 |
|---|---|
| `company_finance` | 公司财报、股票回购、年报、SEC 文件 |
| `person_bio_family` | 人物生平、家庭关系、事故、捐赠、受益者 |
| `film_tv_game` | 电影、电视剧、动漫、游戏、导演、角色 |
| `sports_match` | 比赛、进球、比分、任意球、阵容 |
| `book_blog_article` | 书籍、博客、文章、作者、摘要 |
| `archive_permit_history` | 考古、许可、规划、地方历史档案 |
| `science_species` | 物种、学名、分类、论文摘要 |
| `music_artist` | 音乐、歌手、专辑、厂牌 |
| `general_multihop` | 其他多跳检索任务 |

### 4. 答案兜底与格式清洗

`tpgo/answer_fallback.py` 和 `task_runner_plan_react_negcrit.py` 中的最终清洗逻辑用于处理以下失败模式：

| 失败模式 | 处理 |
|---|---|
| 空答案 | 从轨迹中抽取最可能候选。 |
| `Unable` / `Unknown` / `insufficient evidence` | 视为坏答案，触发兜底。 |
| `<answer confidence="low">...</answer>` | 输出时清理为纯答案文本。 |
| `Correction Note` / `KEY FINDINGS` / `I need to...` | 视为过程文本，触发兜底。 |
| `In` / `Our Founder` / `Read More` 等页面噪声 | 过滤掉，不作为最终提交答案。 |

### 5. TPGO 轨迹分析与记忆沉淀

`tpgo/tpgo_tools.py` 提供面向轨迹的分析能力，包括：

| 功能 | 说明 |
|---|---|
| 轨迹统计 | 统计 steps、search calls、重复搜索、低质量结果、critic BAD 频率、近似 token 等。 |
| 反思记忆抽取 | 从失败轨迹中抽取失败模式和可迁移策略。 |
| Mermaid 图生成 | 将单条轨迹转成可视化流程图，便于报告展示。 |
| TPG 初始化 | 生成 Textual Parameter Graph 配置，用于后续策略管理。 |

需要注意：为遵守评测规则，benchmark/test 运行中不将测试集经验写入可复用长期 memory 后再反复使用。当前用于提交评测的静态策略和路由模块不读取 gold answer，不把测试轨迹传给强模型做蒸馏。

## 蒸馏合规说明

项目包含一个合规静态策略蒸馏模块：

| 文件 | 说明 |
|---|---|
| `tpgo/distilled_strategy.py` | 只包含通用搜索策略提示，不含测试题、答案、轨迹或 benchmark 信息。 |
| `tpgo/DISTILLATION_README.md` | 详细说明蒸馏边界、禁止行为和审计 checklist。 |

合规原则：

| 项 | 当前实现 |
|---|---|
| 是否在 200 条 SimpleVQA/2Wiki/打榜数据上蒸馏 | 否 |
| 是否让 32B 直接看测试题或轨迹 | 否，合规运行时设置 `NEG_CRITIC_ENABLED=0` |
| 是否让强模型参与答题 | 否 |
| 是否复用测试集进化 memory | 否 |
| 是否影响图片题 | `run_benchmark_parallel_mixed.py` 后 50 图片题走旧 `plan_react`，不进 TPGO |

合规静态策略运行时建议：

```bash
TPGO_DISTILLED_STRATEGY=1
NEG_CRITIC_ENABLED=0
```

## 环境配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:8000/v1` | Qwen3.5-9B OpenAI-compatible 服务地址 |
| `MODEL_NAME` | `Qwen3.5-9B` 或 `qwen-3.5` | 基座模型名称 |
| `MAX_STEPS` | runner 内默认值 | 最大 ReAct 步数 |
| `MAX_TOKENS` | `16000` | 单次生成最大 token |
| `SEARCH_PROXY_URL` | 由环境提供 | 搜索代理服务地址 |
| `SEARCH_PROXY_TOKEN` | 可选 | 搜索代理鉴权 token |
| `SANDBOX_BASE_URL` | 由环境提供 | 浏览器沙箱服务地址 |
| `NEG_CRITIC_ENABLED` | `1` | 是否启用在线 critic |
| `TPGO_DISTILLED_STRATEGY` | `0` | 是否注入静态蒸馏策略提示 |

安装依赖：

```bash
pip install -r requirements.txt
```

模型服务需自行启动，要求兼容 OpenAI Chat Completions API。

## 运行命令

### 1. Benchmark 打榜：文本题和图片题隔离策略

推荐使用 mixed runner：

```bash
cd /inspire/qb-ilm2/project/26summer-camp-01/26210830/Auto-evolving-agent

TPGO_DISTILLED_STRATEGY=1 \
NEG_CRITIC_ENABLED=0 \
MAX_STEPS=14 \
NEG_FORCE_ANSWER_STEP=13 \
MAX_SEARCH_CALLS=10 \
MAX_BLOCKED_SEARCHES=3 \
BAD_STREAK_REPLAN=2 \
ROUTE_REPLAN_COOLDOWN=2 \
NEG_EVAL_AFTER_TOOL=0 \
NEG_EVAL_EVERY_STEPS=2 \
NEG_EVAL_MIN_STEP=2 \
python run_benchmark_parallel_mixed.py \
  --group 7 \
  --dataset /inspire/qb-ilm2/project/26summer-camp-01/26210830/datasets/benchmark.csv \
  --start 0 \
  --end 100 \
  --image-start 50 \
  -c 5
```

输出：

```text
results/benchmark/mixed_negcrit_text_planreact_image/<timestamp>/group_7.csv
results/benchmark/mixed_negcrit_text_planreact_image/<timestamp>/group_7.json
results/benchmark/mixed_negcrit_text_planreact_image/<timestamp>/group_7.zip
```

轨迹：

```text
trajectories/benchmark/mixed_negcrit_text_planreact_image/<timestamp>/plan_react_negcrit/
trajectories/benchmark/mixed_negcrit_text_planreact_image/<timestamp>/plan_react/
```

### 2. Benchmark：纯 negcrit runner

```bash
python run_benchmark_parallel_negcrit.py \
  --group 7 \
  --dataset /inspire/qb-ilm2/project/26summer-camp-01/26210830/datasets/benchmark.csv \
  --start 0 \
  --end 50 \
  -c 5
```

### 3. SimpleVQA 评测

```bash
cd /inspire/qb-ilm2/project/26summer-camp-01/26210830/Auto-evolving-agent

MAX_STEPS=14 \
NEG_CRITIC_ENABLED=0 \
TPGO_DISTILLED_STRATEGY=0 \
python run_dataset_eval.py \
  --dataset-name simplevqa \
  --dataset /inspire/qb-ilm2/project/26summer-camp-01/26210830/datasets/datasets/simpleVQA/SimpleVQA.jsonl \
  --output-dir results/datasets/simplevqa \
  --traj-dir trajectories/datasets/simplevqa \
  --runner plan_react \
  --start 0 \
  --end 200 \
  -c 5
```

输出：

```text
results/datasets/simplevqa/<timestamp>/simplevqa_results.jsonl
results/datasets/simplevqa/<timestamp>/simplevqa_trajectories.jsonl
```

### 4. 2Wiki 评测

```bash
cd /inspire/qb-ilm2/project/26summer-camp-01/26210830/Auto-evolving-agent

MAX_STEPS=14 \
NEG_FORCE_ANSWER_STEP=13 \
MAX_SEARCH_CALLS=10 \
MAX_BLOCKED_SEARCHES=3 \
NEG_CRITIC_ENABLED=0 \
TPGO_DISTILLED_STRATEGY=0 \
python run_dataset_eval.py \
  --dataset-name 2wiki \
  --dataset /inspire/qb-ilm2/project/26summer-camp-01/26210830/datasets/datasets/2wiki.jsonl \
  --output-dir results/datasets/2wiki \
  --traj-dir trajectories/datasets/2wiki \
  --runner plan_react_negcrit \
  --start 0 \
  --end 200 \
  -c 5
```

输出：

```text
results/datasets/2wiki/<timestamp>/2wiki_results.jsonl
results/datasets/2wiki/<timestamp>/2wiki_trajectories.jsonl
```

## 输出格式

### SimpleVQA / 2Wiki 结果 JSONL

每行一个样本：

```json
{"index": 0, "instruction": "...", "image": "...", "answer": "...", "pred": "..."}
```

字段说明：

| 字段 | 说明 |
|---|---|
| `index` | 数据集样本序号 |
| `instruction` | 实际输入给智能体的问题文本 |
| `image` | 图片相对路径；2Wiki 为空字符串 |
| `answer` | 数据集中提供的 ground truth |
| `pred` | 智能体预测答案，已去除 `<answer>` 和 `[LOW_CONFIDENCE]` 包装 |

### 轨迹 JSONL

每行是一个 Harness 消息事件：

```json
{"timestamp": 1778520644.2455962, "step_id": 0, "role": "system", "content": "...", "tool_call_id": null}
```

常见字段：

| 字段 | 说明 |
|---|---|
| `timestamp` | 写入时间 |
| `step_id` | ReAct 步数 |
| `role` | `system`、`user`、`assistant`、`tool` |
| `content` | 消息正文或工具返回 |
| `tool_calls` | assistant 发起的工具调用 |
| `tool_call_id` | 对应工具调用 ID |
| `fn_name` | 工具函数名 |
| `fn_args` | 工具参数 |
| `reasoning_content` | 模型推理内容，若服务返回该字段则记录 |

### Benchmark 提交文件

| 文件 | 说明 |
|---|---|
| `group_7.csv` | 与 benchmark.csv 同格式，`answer` 列为模型输出 |
| `group_7.json` | 所有题目的完整轨迹拼接 |
| `group_7.zip` | 提交压缩包 |

## 与评分标准的对应关系

### 1. 智能体搭建评分 10 分

| 要求 | 对应实现 |
|---|---|
| Harness 工程 | `run_benchmark.py`、`run_dataset_eval.py`、`trajectory.py` |
| ReAct 多轮工具调用 | `task_runner.py`、`task_runner_plan_react.py` |
| LLM tool call -> 工具执行 -> LLM 再判断 | 三个 task runner 均支持 |
| SimpleVQA / 2Wiki 评测 | `run_dataset_eval.py` |

### 2. 工具搭建评分 5 分

| 子项 | 对应实现 |
|---|---|
| 文搜文 | `search_text` |
| 图搜文 | `search_image` |
| 访问页面 | `browser_navigate` |
| 获取页面文本 | `browser_get_text` |
| 并发处理页面 | `browser_parallel` |

### 3. 反思模块评分 10 分

| 能力 | 对应实现 |
|---|---|
| 失败后自动触发 Reflection | critic BAD、low-signal、blocked search、budget 触发 |
| 定位失败原因 | `NEG_CRITIC_SYSTEM_PROMPT`、`reflection_tags.py` |
| 自动生成修正策略 | forced reflection、route replan、query strategy prompt |
| 当前任务内修正 | `pending_forced_reflection` 和系统提示注入 |
| 避免重复失败 | duplicate query block、overstuffed query feedback |

### 4. 记忆模块评分 10 分

| 能力 | 对应实现 |
|---|---|
| 短期记忆 | 轨迹上下文、recent queries、candidate votes、state attempts |
| 结构化经验 | `tpgo/reflection_tags.py`、`tpgo/tpgo_tools.py` |
| 历史轨迹分析 | `tpgo.tpgo_tools analyze` |
| 失败模式沉淀 | failure tags、Mermaid graph、TPG config |
| 合规约束 | 不把测试集运行经验写入可复用 memory 后再重复用于同一测试集 |

### 5. 进化效率评分 35 分

本项目支持从以下维度评估进化效率：

| 维度 | 可观测指标 |
|---|---|
| 准确率提升 | 对比 baseline runner 与 TPGO/negcrit runner 的 `answer` / `pred` |
| Token 优化 | `tpgo_tools.py` 近似统计轨迹 token |
| 推理轮数优化 | 轨迹中的最大 `step_id`、平均 step |
| 工具调用优化 | 搜索次数、重复搜索次数、low-signal 次数 |
| 推理时间优化 | runner 日志中每题耗时 |

项目没有在 README 中编造最终成绩；最终分数和排名应以实际运行产物和榜单结果为准。

### 6. 公开打榜评分 20 分

使用 `run_benchmark_parallel_mixed.py` 或 `run_benchmark_parallel_negcrit.py` 生成：

```text
group_7.csv
group_7.json
group_7.zip
```

最终排名由闭源 Agent Benchmark 平台评分决定。

### 7. 报告演示评分 10 分

推荐 PPT 结构：

| 部分 | 内容 |
|---|---|
| 背景与目标 | 为什么需要自进化 Agent |
| 系统架构 | LLM、Harness、Tools、Trajectory、Reflection、TPGO |
| 核心设计 | ReAct、negative critic、task router、fallback、mixed runner |
| 实验设置 | SimpleVQA、2Wiki、benchmark，模型和参数 |
| 结果分析 | 准确率、步数、工具调用、失败案例 |
| 合规说明 | 蒸馏边界、测试集不泄露、memory 不复用 |
| 未来工作 | 更稳定的记忆选择、更强的候选验证、更细粒度工具路由 |

### 8. 加分题：大模型蒸馏 0-10 分

当前项目提供的是合规静态策略蒸馏：

| 要点 | 当前状态 |
|---|---|
| 是否使用 32B 直接答题 | 否 |
| 是否使用 32B 看 200 条测试题 | 否 |
| 是否使用测试轨迹蒸馏 | 否 |
| 是否可打开静态蒸馏提示 | 是，`TPGO_DISTILLED_STRATEGY=1` |
| 合规文档 | `tpgo/DISTILLATION_README.md` |

如后续做真正的强模型数据蒸馏，必须使用独立开发集或合成任务，不能使用 200 条 SimpleVQA、2Wiki 或打榜数据。

## TPGO 工具命令

分析已有轨迹：

```bash
python -m tpgo.tpgo_tools analyze \
  --traj-dir trajectories/benchmark/plan_react_negcrit/<timestamp> \
  --out-dir tpgo/outputs/<run_name>
```

生成轨迹图：

```bash
python -m tpgo.tpgo_tools graph \
  --traj trajectories/benchmark/plan_react_negcrit/<timestamp>/bench_001.jsonl \
  --out tpgo/outputs/bench_001.mmd
```

初始化 TPG 配置：

```bash
python -m tpgo.tpgo_tools init-tpg --out tpgo/current_tpg.json
```

消融实验：

```bash
python -m tpgo.ablation_runner \
  --dataset /inspire/qb-ilm2/project/26summer-camp-01/26210830/datasets/benchmark.csv \
  --end 3 \
  --timeout-seconds 480 \
  --modes basic plan_react plan_react_negcrit
```

## 测试与检查

语法检查：

```bash
python -m compileall run_dataset_eval.py run_benchmark_parallel_mixed.py task_runner_plan_react.py task_runner_plan_react_negcrit.py tpgo tests
```

任务路由测试：

```bash
PYTHONPATH=. python tests/test_task_router.py
```

若环境安装了 pytest：

```bash
python -m pytest tests/test_task_router.py
```

结果格式检查示例：

```bash
LATEST=$(ls -dt results/benchmark/mixed_negcrit_text_planreact_image/* | head -1)
export LATEST

python - <<'PY'
import csv, os, re, sys
csv.field_size_limit(sys.maxsize)
p = os.path.join(os.environ["LATEST"], "group_7.csv")
rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
bad = []
empty = []
for i, row in enumerate(rows):
    ans = (row.get("answer") or "").strip()
    if not ans:
        empty.append(i)
    if re.search(
        r"LOW_CONFIDENCE|<answer|Correction Note|Revised Plan|Constraint Ledger|"
        r"Unable|insufficient|I need to|not definitively|cannot|KEY FINDINGS|"
        r"Based on my search|Internal Correction",
        ans,
        re.I,
    ):
        bad.append((i, ans[:180].replace("\n", " ")))
print("result_dir:", os.environ["LATEST"])
print("rows:", len(rows))
print("empty:", len(empty), empty[:20])
print("bad_format_or_prose:", len(bad))
for item in bad[:30]:
    print(item)
PY
```

## 已知挑战与应对

| 挑战 | 现象 | 当前应对 |
|---|---|---|
| Agent 死循环 | 重复搜索、重复访问、长期无答案 | duplicate query block、search budget、blocked budget |
| 搜索结果低质量 | YouTube/Reddit/403/CAPTCHA/空结果 | `filter_garbage_results`、low-signal streak、route replan |
| 候选实体混淆 | donor/recipient、author/title、team/player 混淆 | answer role check、soft constraint ledger |
| 反思污染答案 | 输出 `Correction Note`、`KEY FINDINGS` | final answer normalization、fallback extraction |
| 图片题受文本路由影响 | 后 50 图片题轨迹异常变长 | mixed runner 中图片题走旧 `plan_react` |
| benchmark 不稳定 | prompt 和模型设置影响结果 | 固定命令参数、输出完整轨迹、支持消融对比 |

## 依赖数据与模型

| 类型 | 内容 |
|---|---|
| 基座模型 | Qwen3.5-9B |
| 可选外部模型 | Qwen3-32B，仅用于 critic/反思辅助；合规静态蒸馏评测时关闭 |
| 验证集 | SimpleVQA、2WikiMultihopQA |
| 打榜集 | 闭源 Agent Benchmark |
| 工具 | 网页搜索、图像搜索、浏览器沙箱 |

## 提交建议

GitHub 仓库建议提交源码、README、测试和配置，不提交以下运行产物：

```text
results/
trajectories/
logs/
__pycache__/
*.pyc
*.log
```

`.gitignore` 已包含上述规则。评测结果和轨迹文件用于平台提交或报告分析，不建议进入源码仓库。

## 项目总结

本项目实现了一个可运行、可审计、可扩展的自进化任务求解智能体原型。系统从基础 ReAct 出发，逐步加入 Plan、Reflection、negative critic、TPGO 路由、软约束、答案兜底和 mixed runner，实现了面向文本多跳问答与图文问答的完整评测闭环。项目重点不是单次回答，而是让智能体在失败、低质量搜索、重复尝试和角色混淆中具备可控的自我修正能力，并通过轨迹文件和评测脚本量化这些机制的效果。
