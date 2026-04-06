# SecondBrain 项目规划图

## 系统整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React + Vite)                  │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌────────────┐  │
│  │ 聊天问答  │  │ 知识盲区诊断  │  │ Quiz界面  │  │ 文档上传    │  │
│  └────┬─────┘  └──────┬───────┘  └────┬─────┘  └─────┬──────┘  │
└───────┼───────────────┼───────────────┼───────────────┼─────────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Orchestrator (意图路由 + 结果汇总)              │   │
│  │         ┌──────────────┼──────────────┐                   │   │
│  │         ▼              ▼              ▼                   │   │
│  │  ┌─────────────┐ ┌──────────┐ ┌──────────────┐           │   │
│  │  │ QA SubAgent │ │  Quiz    │ │ Gap Analysis │           │   │
│  │  │ (RAG问答)   │ │ SubAgent │ │  SubAgent    │           │   │
│  │  └─────────────┘ └──────────┘ └──────────────┘           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │     Ingestion Pipeline (多文件并行解析 → 串行写入)         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Services: BGE-m3 Embedding | BGE-reranker-v2 | Qwen3-Plus LLM │
│  Storage:  Milvus (向量) | SQLite (mastery/对话/概念)            │
└─────────────────────────────────────────────────────────────────┘
```

## 技术栈

- **Embedding**: BGE-m3, 1024维, 中英文多语言, dense + sparse 双编码
- **Reranker**: BGE-reranker-v2-m3, cross-encoder, fp16
- **LLM**: Qwen3.6-Plus, chat + stream_chat + tool-calling
- **向量库**: Milvus Standalone (HNSW + SPARSE_INVERTED_INDEX, RRF 融合)
- **后端**: FastAPI + LangGraph + SQLite (async via aiosqlite)
- **前端**: React + Vite + TypeScript + Tailwind CSS v4
- **PDF 解析**: `magic-pdf`（import 名 `magic_pdf`），与 PyPI 包 `mineru` 2.x 无模块级兼容；后端依赖以 `requirements.txt` 为准
- **部署**: Docker Compose (etcd + minio + milvus + backend)

## 文档索引

按需查阅，不必全部阅读：

| 文档 | 内容 | 何时需要 |
|------|------|----------|
| [docs/ARCHITECTURE_AND_DATAFLOW.md](docs/ARCHITECTURE_AND_DATAFLOW.md) | **架构 + 存储 + 数据流 + LangGraph 工作流（详述）** | 新人上手、联调、写设计说明 |
| [docs/agents.md](docs/agents.md) | Agent 分层架构 (Orchestrator + 3 SubAgent) | 修改 Agent 调度逻辑 |
| [docs/langgraph.md](docs/langgraph.md) | LangGraph 节点、流程图、条件路由 | 修改图结构或节点 |
| [docs/tools.md](docs/tools.md) | Tool 定义、Agent×Tool 分配、Gap 规则 | 增删改 Tool |
| [docs/data-storage.md](docs/data-storage.md) | 数据流、检索链路、SQLite 表结构、Mastery 公式 | 修改存储或检索逻辑 |
| [docs/memory.md](docs/memory.md) | 短期/长期记忆机制、consolidation 流程 | 修改记忆相关代码 |
| [docs/evaluation.md](docs/evaluation.md) | 6 维评估指标体系 | Step 8 评估阶段 |
| [docs/directory.md](docs/directory.md) | 项目目录结构 | 需要了解文件位置 |

## 实现流程

```
Step 1: Services → Step 2: Ingestion → Step 3: Tools → Step 4: SubAgents
→ Step 5: Orchestrator → Step 5.5: Memory → Step 6: API → Step 7: Frontend
→ Step 8: 评估 & 优化 (当前)
```

### 当前进度

- [x] Step 1-7: 全部完成
- [ ] Step 8: 评估 & 优化 (进行中：已完成评测脚本修正与 `smoke / standard / full` 分档，待产出稳定结果与优化建议)

### Step 8 当前实现补充

- 评测脚本已支持三档运行口径，均复用同一份数据集文件：
  - `smoke`：开发时快速回归
  - `standard`：日常主评测口径
  - `full`：阶段性全量验收
- `evaluation/profiles.py` 现使用**确定性分层采样**而非简单截取文件头部：
  - QA 子集按 `single-hop / multi-hop` 分桶后，再按来源文档（multi-hop 为来源组合）做稳定 round-robin 选样，避免结果受数据集排序影响
  - Intent 子集按 `qa / quiz / gap_analysis` 标签均衡抽样，保证 `smoke / standard` 档位都覆盖三类路由
- `eval_rag.py` 与 `eval_agent.py` 均通过 `--profile {smoke|standard|full}` 选择档位，并在结果 `meta` 中记录本次 profile；每轮各只落盘一份 JSON（`results/rag/rag_eval_*.json`、`results/agent/agent_eval_*.json`），文件名与 `meta.started_at` / `meta.finished_at` 为 **洛杉矶时间**（`America/Los_Angeles`），`meta.output_file` 为绝对路径
- 评测日志按任务拆分目录落盘：RAG 写入 `results/logs_rag/eval_rag_*.log`，Agent 写入 `results/agentlog/eval_agent_*.log`；两类结果 JSON 的 `meta.run_log_file` 均记录对应日志绝对路径，便于回查
- Judge LLM 默认使用 Moonshot `kimi-k2.5`（由 `settings.judge_api_key / settings.judge_base_url / settings.judge_model` 配置，默认 `https://api.moonshot.ai/v1`）：
  - Agent judge 走 Kimi 非 thinking 路径，请求显式带 `thinking={"type":"disabled"}`
  - RAGAS 侧仍按官方文档使用 `AsyncOpenAI` + `llm_factory` + `aevaluate()`，但会显式覆盖 Kimi K2.5 的固定参数约束
  - Kimi judge 口径默认使用 `temperature=0.6`、`top_p=0.95`、`max_tokens=32768`（RAGAS 路径），避免被 `ragas` 默认值覆盖后触发 Kimi 参数错误或过早截断
- 评测工程质量（已完成重构）：
  - `evaluation/eval_utils.py` 提取共享工具（logging 配置、P50/P95 计算、时区常量），消除 `eval_rag.py` / `eval_agent.py` 间的重复代码
  - RAGAS 导入全部延迟到函数内部，模块级别不依赖 ragas，避免未安装 ragas 时整个 evaluation 包不可导入
  - RAGAS metrics 使用公开 API（`Faithfulness` / `AnswerRelevancy` / `LLMContextPrecisionWithReference` / `LLMContextRecall`），不依赖下划线前缀私有类
  - `evaluation/ragas_runner.py` 将 RAGAS 官方接入流与项目评测主流程解耦，`eval_rag.py` 仅保留数据集采样、RAG 执行、IR 指标与结果落盘逻辑
  - `eval_agent.py` — `classify_intent` 异常保护 + orchestrator 120s 超时 + latency 仅统计成功 case（`latency_p50/p95_ms` = success-only；`latency_p50/p95_ms_all` = 含失败）
  - `requirements.txt` 已提升到 `ragas>=0.4.3`，与当前官方文档和 async eval API 对齐
