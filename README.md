# Harness-SII: Qwen Agent 自动化评测框架

基于 Qwen-3.5 大模型的 Agent 自动化执行框架，通过 Sglang OpenAI 兼容 API 驱动 LLM 进行多轮工具调用（tool-calling），完成联网搜索、浏览器操作等复杂任务。

## 代码结构

```
harness-sii/
├── run.sh                         # 统一入口脚本（切换数据集/范式/范围）
├── run_benchmark.py               # Benchmark 打榜 runner
├── run_simpleqa.py                # SimpleVQA 评测 runner
├── run_2wiki.py                   # 2Wiki 评测 runner
├── task_runner.py                 # 基础范式：标准 ReAct 循环
├── task_runner_plan_react.py      # 进阶范式：Plan & ReAct + Reflection
├── trajectory.py                  # 轨迹记录器（JSONL 格式）
├── sandbox_client.py              # 浏览器服务 HTTP 客户端
├── roles.py                       # 消息角色枚举
├── eval_metrics.py                # 评测指标计算
├── tools/
│   ├── search_tool.py             # 联网搜索（文字 + 反向图搜）
│   └── browser_tool.py            # 浏览器操作（导航/点击/输入）
├── search-proxy/                  # 搜索代理服务（部署在有外网机器）
│   └── app/
├── results/                       # 结果输出目录
│   ├── basic/                     #   基础范式结果
│   └── plan_react/                #   进阶范式结果
└── trajectories/                  # 轨迹输出目录
    ├── benchmark/{mode}/{ts}/     #   按范式+时间戳隔离
    ├── simpleqa/{mode}/{ts}/
    └── 2wiki/{mode}/{ts}/
```

## 数据集介绍

| 数据集 | 文件 | 数量 | 类型 | 说明 |
|--------|------|------|------|------|
| Benchmark | `datasets/benchmark.csv` | 100 题 | 打榜 | 无 ground truth，需提交评分 |
| SimpleVQA | `datasets/simpleVQA/SimpleVQA.jsonl` | 99 题 | 评测 | 图文问答，有 ground truth |
| 2Wiki | `datasets/2wiki.jsonl` | 100 题 | 评测 | 纯文本多跳问答，有 ground truth |

### Benchmark（打榜数据集）

CSV 列：`problem`、`image`（base64，纯文本题为空）、`answer`（待填）

- index 0-49：纯文本题，复杂多跳推理，需多次联网搜索
- index 50-99：图片题，先识别图中内容再搜索相关事实

### SimpleVQA（评测集）

JSONL 字段：`question`、`answer`、`image`（相对路径）、`image_url`

给定一张图片和一个问题，需要结合图片内容联网搜索回答。

### 2Wiki（评测集）

JSONL 字段：`question`、`answer`、`type`（comparison/compositional）、`context`

纯文本多跳问答，需要跨多个实体推理。

## 两种范式

| 范式 | 模块 | 特点 | 适用场景 |
|------|------|------|----------|
| `basic` | `task_runner.py` | 标准 ReAct 循环，LLM 直接调用工具 | 简单题、快速迭代 |
| `plan_react` | `task_runner_plan_react.py` | Plan-then-Solve + ReAct + Reflection | 复杂多跳题、需要规划的任务 |

**basic 范式**：LLM → tool_call → 结果 → LLM → ... → 最终回答

**plan_react 范式**：
1. Planning：模型先分解问题为子步骤
2. ReAct：按计划逐步执行工具调用
3. Reflection：每步执行后评估进展，动态调整
4. Forced answer：接近步数上限时强制总结

## 环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_BASE_URL` | `http://127.0.0.1:8000/v1` | Sglang LLM 服务地址 |
| `MODEL_NAME` | `qwen-3.5` | 模型标识 |
| `MAX_STEPS` | `20`（basic）/ `25`（plan_react） | 最大循环步数 |
| `MAX_TOKENS` | `16000` | 单次生成最大 token |
| `SEARCH_PROXY_URL` | — | 搜索代理地址（必填） |
| `SEARCH_PROXY_TOKEN` | — | 搜索代理认证 token（可选） |
| `SANDBOX_BASE_URL` | — | 浏览器服务地址（图片题需要） |
| `SERPER_API_KEY` | — | Serper API Key（直连模式需要） |
| `JINA_API_KEY` | — | Jina API Key（直连模式可选） |

## 运行命令

### 统一入口 run.sh

```bash
./run.sh [dataset] [mode] [start] [end]
```

参数说明：
- `dataset`：`benchmark` | `simpleqa` | `2wiki` | `all`（默认 benchmark）
- `mode`：`basic` | `plan_react`（默认 basic）
- `start`/`end`：题目范围（可选，不填跑全量）

示例：

```bash
# 打榜全量，基础范式
./run.sh benchmark basic

# 打榜全量，进阶范式
./run.sh benchmark plan_react

# SimpleVQA 前 5 题测试
./run.sh simpleqa plan_react 0 5

# 2Wiki 第 10-20 题
./run.sh 2wiki basic 10 20

# 全部数据集一起跑
./run.sh all plan_react
```

可通过 `GROUP_ID` 环境变量覆盖默认组号（默认 7）：

```bash
GROUP_ID=3 ./run.sh benchmark plan_react
```

### 单独运行 Python 脚本

```bash
# 带 --mode 参数切换范式
python run_benchmark.py --group 7 --mode plan_react --start 0 --end 5
python run_simpleqa.py --group 7 --mode basic
python run_2wiki.py --group 7 --mode plan_react --start 10 --end 20
```

### 断点续跑

所有脚本支持断点续跑。中断后重新运行同样的命令，已完成的题目自动跳过。
进度文件在 `results/{mode}/group_7_*_progress.jsonl`，如需重跑某个数据集，删掉对应 progress 文件即可。

## 输出文件说明

### 目录结构

```
results/
├── basic/
│   ├── group_7_benchmark_progress.jsonl          # 断点续跑（无时间戳）
│   ├── group_7_benchmark_20260519_143022.jsonl   # 结果文件
│   ├── group_7_benchmark_traj_20260519_143022.jsonl  # 汇总轨迹
│   └── group_7_20260519_143022.zip               # 提交压缩包
└── plan_react/
    └── ...（同上）

trajectories/
├── benchmark/
│   ├── basic/20260519_143022/      # 每题单独轨迹
│   │   ├── bench_000.jsonl
│   │   ├── bench_001.jsonl
│   │   └── ...
│   └── plan_react/20260519_150511/
├── simpleqa/
│   └── {mode}/{timestamp}/
└── 2wiki/
    └── {mode}/{timestamp}/
```

### 结果文件格式

每行一个 JSON：

```json
{"index": 0, "instruction": "问题", "image": "", "answer": "ground truth", "pred": "模型预测"}
```

- 评测集（simpleqa/2wiki）：`answer` = ground truth，`pred` = 模型预测
- 打榜集（benchmark）：`answer` = 空，`pred` = 模型输出

### 轨迹文件格式

每行一个 trajectory step：

```json
{
  "timestamp": 1716000000.0,
  "step_id": 1,
  "role": "assistant",
  "content": "我需要搜索...",
  "tool_calls": [{"function": {"name": "search_text", "arguments": "..."}}],
  "reasoning_content": "思考过程..."
}
```

`trajectories/` 下是每题独立的轨迹文件，`results/` 下的 `*_traj_*.jsonl` 是所有题目轨迹的汇总拼接。

### 提交文件

#### 评测集（自测用）

| 文件 | 说明 |
|------|------|
| `group_7_simpleqa_{ts}.jsonl` | 99 条，含 answer 和 pred |
| `group_7_simpleqa_traj_{ts}.jsonl` | 轨迹汇总 |
| `group_7_2wiki_{ts}.jsonl` | 100 条，含 answer 和 pred |
| `group_7_2wiki_traj_{ts}.jsonl` | 轨迹汇总 |

#### 打榜数据集（正式提交）

| 文件 | 说明 |
|------|------|
| `group_7.json` | Agent 推理轨迹 |
| `group_7.csv` | answer 列为模型输出 |
| `group_7.zip` | 包含上述两个文件 |

## 架构图

```
┌─────────────────────────────────────────────────────────┐
│                    GPU Host (无外网)                      │
│                                                         │
│  run.sh / run_*.py                                      │
│       │                                                 │
│       ▼                                                 │
│  task_runner.py / task_runner_plan_react.py              │
│       │                                                 │
│       ├── tools/search_tool.py ──→ search-proxy (转发)  │
│       └── tools/browser_tool.py ──→ browser-service     │
│       │                                                 │
│       ▼                                                 │
│  trajectory.py  ← 记录每一步交互到 JSONL                 │
└──────────────────────────┬──────────────────────────────┘
                           │ SSH 端口转发
┌──────────────────────────▼──────────────────────────────┐
│                  CPU Host (有外网)                        │
│                                                         │
│  search-proxy (FastAPI)                                 │
│       ├── Serper API (Google 搜索 / Google Lens)        │
│       └── Jina Reader (网页正文抽取)                     │
└─────────────────────────────────────────────────────────┘
```
