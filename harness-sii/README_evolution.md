# 自进化 Agent 使用说明

本目录在原始 ReAct Harness 之上增加了轻量的反思与长期记忆闭环：

| 模块 | 文件 | 作用 |
| --- | --- | --- |
| 反思监控 | `reflection.py` | 检测空回复、重复工具调用、工具错误、最大轮数等失败信号，并在运行中注入简短修正提示 |
| 长期记忆 | `agent_memory.py` | JSONL 结构化存储经验，按任务族、标签、置信度、时间和词面相关性召回 |
| Agent 接入 | `task_runner.py` | 开局注入记忆，运行中做反思与工具结果截断，结束后写入 stats/signals/memory |
| 对比分析 | `analyze_runs.py` | 汇总准确率、轮数、token、工具调用、反思次数、记忆召回/写入等指标 |

## 运行模式

| 模式 | 参数 | 说明 |
| --- | --- | --- |
| 原始基线 | `--agent-mode raw` | 禁用 memory 和 reflection，用于无进化 baseline |
| 进化模式 | `--agent-mode evolved` | 启用 memory 和 reflection，默认使用当前 run 的 `<run_dir>/memory` |
| 离线训练记忆 | `--memory-update-mode gold --allow-gold-feedback` | 只建议在允许暴露 label 的开源训练/开放评测数据上使用 |
| 打榜模式 | `run_benchmark.py --agent-mode evolved --memory-update-mode heuristic` | 不暴露 gold，只根据轨迹失败信号做 test-time memory update |

## 推荐命令

SimpleVQA 原始基线：

```bash
python harness-sii/eval_simplevqa.py \
  --dataset datasets/simpleVQA/SimpleVQA.jsonl \
  --data-root datasets/simpleVQA \
  --run-name simplevqa_raw \
  --agent-mode raw \
  --result-format full \
  --overwrite
```

SimpleVQA 进化模式：

```bash
python harness-sii/eval_simplevqa.py \
  --dataset datasets/simpleVQA/SimpleVQA.jsonl \
  --data-root datasets/simpleVQA \
  --run-name simplevqa_evolved \
  --agent-mode evolved \
  --memory-update-mode heuristic \
  --result-format full \
  --overwrite
```

2Wiki 原始基线：

```bash
python harness-sii/run_2wiki.py \
  --dataset datasets/2wiki.jsonl \
  --run-name 2wiki_raw \
  --agent-mode raw \
  --result-format full \
  --overwrite
```

2Wiki 进化模式：

```bash
python harness-sii/run_2wiki.py \
  --dataset datasets/2wiki.jsonl \
  --run-name 2wiki_evolved \
  --agent-mode evolved \
  --memory-update-mode heuristic \
  --result-format full \
  --overwrite
```

用开源数据离线沉淀带 gold 反馈的记忆：

```bash
python harness-sii/run_2wiki.py \
  --dataset datasets/2wiki.jsonl \
  --run-name 2wiki_gold_memory_train \
  --agent-mode evolved \
  --memory-dir harness-sii/memory_train \
  --memory-update-mode gold \
  --allow-gold-feedback \
  --result-format full \
  --overwrite
```

打榜数据运行。注意：这里没有 `--allow-gold-feedback`，也不支持 gold mode：

```bash
python harness-sii/run_benchmark.py \
  --dataset datasets/benchmark.csv \
  --run-name benchmark_evolved \
  --agent-mode evolved \
  --memory-dir harness-sii/memory_train \
  --memory-update-mode heuristic \
  --result-format full \
  --overwrite
```

## 指标对比

```bash
python harness-sii/analyze_runs.py \
  harness-sii/runs/simplevqa_raw \
  harness-sii/runs/simplevqa_evolved
```

输出包含：

| 指标 | 含义 |
| --- | --- |
| `exact / contains` | 精确匹配与包含 gold 的比例 |
| `avg_steps` | 平均 ReAct 轮数 |
| `avg_tokens` | 平均 token 消耗 |
| `avg_tool` | 平均工具调用数 |
| `tool_err / repeat` | 工具错误与重复工具调用次数 |
| `reflect` | 运行中反思提示次数 |
| `mem_in / mem_out` | 记忆召回与写入次数 |

## 注意事项

- `--agent-mode raw` 是对照基线，必须用于“无进化”结果。
- `--allow-gold-feedback` 只用于允许 label 暴露的数据，不要用于闭源打榜。
- 如果要严格遵守“测试集整体只跑一次”，不要在同一个测试集 run 完成后带着该 run 的 memory 回头重跑同一批样本。
- 并发运行会让 test-time memory update 的顺序性变弱；若要观察逐样本进化效果，建议 `--concurrency 1`。

