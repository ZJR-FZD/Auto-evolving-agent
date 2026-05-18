# Harness-SII: Qwen Agent 自动化评测框架

## 项目概述

本项目是一个基于 Qwen-3.5 大模型的 Agent 自动化执行框架，通过 Sglang OpenAI 兼容 API 驱动 LLM 进行多轮工具调用（tool-calling），完成联网搜索、浏览器操作等复杂任务，并将完整交互轨迹记录为 JSONL 文件。

整体架构采用 **GPU 主机（无外网）+ CPU 主机（有外网）** 的分离部署模式，通过 SSH/VSCode 端口转发打通网络。

## 比赛评测（快速开始）

### 跑 Benchmark（比赛提交）

```bash
cd /inspire/qb-ilm2/project/26summer-camp-01/26210094/harness-sii

# 跑全部 100 题（X 替换为你的组号）
python run_benchmark.py --group X

# 先跑前 3 题测试
python run_benchmark.py --group X --start 0 --end 3

# 指定输出目录
python run_benchmark.py --group X -o my_results
```

**结果文件位置**（默认在 `results/` 目录下）：

| 文件 | 说明 |
|------|------|
| `results/group_X.csv` | 答案文件（problem + image + answer 列） |
| `results/group_X.json` | 推理轨迹文件（每题的完整 Agent 交互记录） |
| `results/group_X.zip` | 提交用压缩包（包含上面两个文件） |
| `results/group_X_progress.jsonl` | 进度文件（支持断点续跑） |
| `trajectories/benchmark/bench_XXX.jsonl` | 每题的原始轨迹 |

断点续跑：中断后重新运行同样的命令，已完成的题目会自动跳过。

### 跑 SimpleVQA

```bash
python run_simpleqa.py --start 0 --end 99
```

结果在 `results/simpleqa_results.jsonl`，轨迹在 `trajectories/simpleqa/`。

## 架构图

```
┌─────────────────────────────────────────────────────────┐
│                    GPU Host (无外网)                      │
│                                                         │
│  task_runner.py  ──→  Sglang (Qwen-3.5 LLM)            │
│       │                                                 │
│       ├── tools/search_tool.py ──→ search-proxy (转发)  │
│       └── tools/browser_tool.py ──→ browser-service     │
│                                                         │
│  trajectory.py  ← 记录每一步交互到 JSONL                 │
└──────────────────────────┬──────────────────────────────┘
                           │ SSH 端口转发
┌──────────────────────────▼──────────────────────────────┐
│                  CPU Host (有外网)                        │
│                                                         │
│  search-proxy (FastAPI :8090)                           │
│       ├── Serper API (Google 搜索 / Google Lens)        │
│       └── Jina Reader (网页正文抽取)                     │
└─────────────────────────────────────────────────────────┘
```

## 目录结构与文件说明

```
harness-sii/
├── task_runner.py          # 主编排器：驱动 Agent 循环（LLM ↔ 工具调用）
├── sandbox_client.py       # 浏览器服务 HTTP 客户端（单例模式）
├── trajectory.py           # 轨迹记录器：将每轮交互写入 JSONL
├── roles.py                # 消息角色枚举（system/user/assistant/tool）
├── requirements.txt        # Python 依赖
├── tools/                  # 工具实现目录
│   ├── search_tool.py      # 联网搜索工具（文字搜索 + 反向图搜）
│   └── browser_tool.py     # 浏览器操作工具（导航/点击/输入/并发）
├── search-proxy/           # 搜索代理服务（部署在有外网的 CPU 主机）
│   ├── run.sh              # 一键启动脚本
│   ├── requirements.txt    # 代理服务依赖
│   └── app/
│       ├── main.py         # FastAPI 入口
│       ├── routes.py       # HTTP 路由（/search/text, /search/image 等）
│       ├── schemas.py      # Pydantic 请求/响应模型
│       ├── upstream.py     # 外部 API 调用（Serper / Jina / 0x0）
│       └── config.py       # 配置项（环境变量）
├── trajectories/           # 轨迹输出目录（JSONL 文件）
├── README_browser.md       # 浏览器工具详细文档
├── README_search.md        # 搜索工具详细文档
└── README_wiki.md          # Wiki 工具文档
```

## 核心模块详解

### 1. `task_runner.py` — 主编排器

Agent 循环的核心入口，职责：
- 构建 system prompt + user 指令，初始化对话
- 循环调用 Qwen-3.5（通过 Sglang OpenAI 兼容接口）
- 解析 LLM 返回的 `tool_calls`，分发到对应工具函数执行
- 将工具结果写回对话历史，继续下一轮
- 循环终止条件：LLM 不再调用工具（`finish_reason=stop`）或达到 `MAX_STEPS`

关键配置（环境变量）：
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_BASE_URL` | `http://127.0.0.1:8000/v1` | Sglang 服务地址 |
| `MODEL_NAME` | `qwen-3.5` | 模型标识 |
| `MAX_STEPS` | `20` | 最大循环步数 |
| `MAX_TOKENS` | `16000` | 单次生成最大 token |
| `DISABLE_TOOLS` | `0` | 设为 1 可关闭工具注册，纯文本调试 |

CLI 用法：
```bash
python -m task_runner \
    --instruction "请帮我查询上海创智学院谢源老师的相关信息" \
    --task-id my_task_010

# 带图像输入
python -m task_runner \
    --instruction "请分析图像内容并搜索相关信息" \
    --image ./path/to/image.jpg \
    --image-url "https://..." \
    --task-id my_task_011
```

### 2. `tools/search_tool.py` — 联网搜索工具

提供两个工具函数：
- **`search_text(query, top_k, fetch, max_chars)`**：Google 文字搜索 + Jina 正文抽取
- **`search_image(image_url, top_k, fetch, max_chars)`**：Google Lens 反向图搜

支持两种运行模式（自动切换）：
- **Proxy 模式**（推荐）：GPU 主机无外网时，通过 `SEARCH_PROXY_URL` 转发请求到 CPU 主机的 search-proxy 服务
- **Direct 模式**：本地直连 Serper/Jina API（需要外网 + API Key）

### 3. `tools/browser_tool.py` — 浏览器操作工具

通过 `sandbox_client.py` 驱动远程 Chromium 浏览器，提供：
- **`browser_navigate(url)`**：打开页面，返回文本预览
- **`browser_get_text()`**：获取当前页面完整可见文本
- **`browser_click(selector)`**：CSS 选择器点击元素
- **`browser_type(selector, text)`**：向输入框键入文本
- **`browser_parallel(urls)`**：并发打开多个 URL（多标签页）

### 4. `sandbox_client.py` — 浏览器服务客户端

单例模式的 HTTP 客户端，封装了 browser-service 的所有 API：
- 会话管理（create / reset / close）
- 浏览器操作（navigate / get_text / click / type / scroll / screenshot / eval_js）
- 标签页管理（new_tab / close_tab / list_tabs）

通过 `SANDBOX_BASE_URL` 环境变量指定 browser-service 地址。

### 5. `trajectory.py` — 轨迹记录器

将 Agent 每一步交互以 JSONL 格式持久化，每行包含：
```json
{
  "timestamp": 1716000000.0,
  "step_id": 1,
  "role": "assistant",
  "content": "...",
  "tool_call_id": null,
  "tool_calls": [...],
  "reasoning_content": "..."
}
```

核心方法：
- `write()` — 追加一条记录
- `to_messages()` — 转换为 OpenAI messages 格式（用于下一轮 LLM 调用）
- `summary()` — 返回统计信息

### 6. `search-proxy/` — 搜索代理服务

独立的 FastAPI 微服务，部署在有外网的 CPU 主机上，为无外网的 GPU 主机提供搜索能力。

路由：
| 端点 | 方法 | 功能 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/search/text` | POST | 文字搜索（Serper + Jina） |
| `/search/image` | POST | 反向图搜（Serper Lens + Jina） |
| `/fetch` | POST | 单独抓取网页正文（Jina） |
| `/upload_image` | POST | 上传本地图片到公网（0x0.st） |

启动方式：
```bash
cd search-proxy
export SERPER_API_KEY=xxx
export JINA_API_KEY=xxx        # 可选
export PROXY_API_TOKEN=xxx     # 可选，建议设置
./run.sh
```

## 快速开始

1. **启动 LLM 服务**：在 GPU 主机上用 Sglang 部署 Qwen-3.5
2. **启动 search-proxy**：在 CPU 主机上运行 `search-proxy/run.sh`
3. **启动 browser-service**：确保浏览器服务在运行
4. **端口转发**：通过 SSH/VSCode 将 CPU 主机的 8090 端口转发到 GPU 主机
5. **运行任务**：
```bash
export SEARCH_PROXY_URL=http://127.0.0.1:8090
python -m task_runner -i "你的任务指令" -t task_001
```

## 依赖

```
openai>=1.0.0       # OpenAI 兼容客户端（调用 Sglang）
requests>=2.28.0    # HTTP 请求
rank_bm25>=0.2.2    # BM25 排序（备用）
```
