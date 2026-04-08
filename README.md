# SecondBrain

基于个人笔记的 RAG + Quiz + 知识盲区诊断系统。上传你的学习文档，让 AI 帮你问答、出题、找出你的知识盲区。

## 系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│  Frontend (React 19 + Vite + TypeScript + Tailwind v4)               │
│    ChatPanel (Markdown/LaTeX) │ QuizModal │ FileUpload               │
└───────────────────────────┬──────────────────────────────────────────┘
                            │ HTTP (JSON / SSE)
┌───────────────────────────▼──────────────────────────────────────────┐
│  FastAPI Backend                                                      │
│    /api/chat (+ /stream + /history)                                   │
│    /api/quiz/answer                                                   │
│    /api/ingest/upload · /documents · DELETE                           │
├───────────────────────────────────────────────────────────────────────┤
│  LangGraph Orchestrator (Plan → Execute → Replan)                     │
│    ┌─────────────┐   ┌──────────────┐   ┌──────────────┐            │
│    │ QA SubAgent  │   │ Quiz SubAgent│   │ Gap SubAgent │            │
│    │ ReAct +      │   │ RAG 出题     │   │ 规则分类 +    │            │
│    │ Reflection   │   │ 2 道 MCQ     │   │ LLM 报告     │            │
│    └──────┬───────┘   └──────┬───────┘   └──────┬───────┘            │
│           └──────────────────┼──────────────────┘                    │
│                              ▼                                        │
│    Tools: rag_search │ read_mastery │ update_mastery │ search_concepts│
├───────────────────────────────────────────────────────────────────────┤
│  Services                                                             │
│    BGE-m3 Embedding (dense + sparse) │ Qwen3-Plus LLM (tool-calling)│
├───────────────────────────────────────────────────────────────────────┤
│  Storage                                                              │
│    Milvus (文档向量 + 长期对话记忆) │ SQLite (掌握度/概念/短期对话)    │
└───────────────────────────────────────────────────────────────────────┘
```

## 核心功能

| 功能 | 说明 |
|------|------|
| **智能问答 (QA)** | 基于文档的 RAG 问答，支持对比/因果/并列类复杂问题；ReAct + Reflection 架构，自动检索验证 |
| **出题测验 (Quiz)** | 根据文档内容和用户薄弱点生成 MCQ 选择题，答题后实时更新掌握度（时间衰减模型） |
| **知识盲区诊断 (Gap)** | 对比知识库概念与用户掌握度，按"未接触/薄弱/需巩固/已掌握"四级分类生成诊断报告 |
| **文档管理** | 支持 PDF / Word / Markdown 上传，自动解析 → 分块 → 概念抽取 → 向量化写入 |
| **对话记忆** | 短期 SQLite + 长期 Milvus 双轨记忆，自动整合与上下文注入 |

## 技术栈

| 层次 | 技术 |
|------|------|
| 前端 | React 19 + Vite + TypeScript + Tailwind CSS v4 |
| 后端 | FastAPI + LangGraph + Python 3.11 |
| Embedding | BGE-m3（1024 维，中英文多语言，dense + sparse 双编码 + Heading Augmentation） |
| LLM | Qwen3-Plus（chat / stream / tool-calling） |
| 向量库 | Milvus Standalone（HNSW + SPARSE_INVERTED_INDEX，RRF 混合检索） |
| 数据库 | SQLite（aiosqlite + SQLAlchemy Async，WAL 模式） |
| PDF 解析 | magic-pdf（MinerU） |
| 部署 | Docker Compose（Milvus + etcd + MinIO + Backend） |

## 技术亮点

### RAG Pipeline

- **Hybrid Retrieval**：Dense + Sparse 混合检索，RRF（Reciprocal Rank Fusion）融合
- **Heading Augmentation**：Embedding 编码时拼接文档标题路径到 chunk 文本前，解决"关键词只在标题中"的检索失败问题
- **数据驱动的 Reranker 决策**：通过控制变量 A/B 实验发现 Cross-Encoder Reranker 在高质量 embedding 下为负增益，关闭后检索精度和延迟均显著改善

### 多 Agent 编排

- **Plan & Execute with Replan**：LangGraph 状态图驱动，支持 DAG 依赖调度、并发执行（Semaphore 限流）、失败隔离
- **单步优化**：单步 plan 自动跳过 replan 节点，减少无效 LLM 调用
- **架构做减法**：QA SubAgent 从四阶段 pipeline 重构为 ReAct + Reflection 两阶段，LLM 调用从 5-10 次降至 2-4 次

## 目录结构

```
SecondBrain/
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI 入口 + lifespan
│   │   ├── config.py              # 配置（读取 .env）
│   │   ├── api/routes/            # chat / quiz / ingest / gap 路由
│   │   ├── agent/
│   │   │   ├── graph.py           # LangGraph 状态图编译与运行
│   │   │   ├── state.py           # OverallState TypedDict
│   │   │   ├── nodes/             # orchestrator (plan/execute/replan) + memory_manager
│   │   │   ├── subagents/         # qa_agent / quiz_agent / gap_agent
│   │   │   └── tools/             # rag_search / read_mastery / update_mastery / search_concepts
│   │   ├── services/              # embedding / reranker / llm_client / milvus_client / doc_processor
│   │   └── db/                    # SQLite 模型 & CRUD
│   ├── evaluation/                # RAGAS & Agent 评估框架
│   ├── tests/                     # 单元 & 集成测试
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/            # ChatPanel / QuizModal / FileUpload / Layout
│   │   ├── services/api.ts        # API 客户端
│   │   └── App.tsx
│   └── package.json
├── docker-compose.yml             # Milvus + etcd + MinIO + Backend
└── README.md
```

## 快速启动

### 前置条件

- Docker & Docker Compose
- Node.js 18+（仅本地开发前端时需要）
- 下载 BGE 模型到 `backend/models/`（见[模型准备](#模型准备)）

### 1. 配置环境变量

复制并填写 backend 的环境变量：

```bash
cp backend/.env.example backend/.env
```

编辑 `backend/.env`，至少填写以下字段：

```env
# 通义千问 API Key（https://dashscope.aliyuncs.com）
LLM_API_KEY=sk-xxxx
```

### 2. 模型准备

将以下模型下载到 `backend/models/` 目录：

```
backend/models/
├── bge-m3/               # Embedding 模型（BAAI/bge-m3）
└── bge-reranker-v2-m3/   # Reranker 模型（BAAI/bge-reranker-v2-m3）
```

可通过 Hugging Face 下载：

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-m3', local_dir='backend/models/bge-m3')
snapshot_download('BAAI/bge-reranker-v2-m3', local_dir='backend/models/bge-reranker-v2-m3')
"
```

### 3. 启动后端服务（Docker Compose）

```bash
# 启动 Milvus（etcd + minio + milvus）+ FastAPI Backend
docker compose up -d

# 查看日志
docker compose logs -f backend
```

服务启动后：
- **Backend API**：http://localhost:18080
- **API 文档**：http://localhost:18080/docs
- **Milvus**：localhost:19530

### 4. 启动前端（本地开发）

```bash
cd frontend
npm install
npm run dev
```

前端开发服务器启动后访问：http://localhost:5173

> 前端已配置 Vite dev proxy，`/api` 请求默认自动转发到 `localhost:18080`。
> 如需改为本地直跑后端，可在 `frontend/.env.local` 设置 `VITE_PROXY_TARGET=http://localhost:8000`。

### 5. 生产部署（前端构建）

```bash
cd frontend
npm run build
# 构建产物位于 frontend/dist/，可用 nginx 等静态服务器托管
```

---

## 本地开发（不使用 Docker）

### 启动 Milvus

```bash
# 仅启动基础设施（不启动 backend 容器）
docker compose up -d etcd minio milvus
```

### 启动 Backend

```bash
cd backend
pip install -r requirements.txt

# 设置 Milvus 连接地址（本地开发时 host 为 localhost）
export MILVUS_HOST=localhost

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 使用说明

1. **上传文档**：在「文档管理」面板上传 PDF、Word 或 Markdown 文件，系统自动解析并写入向量库。
2. **智能问答**：在聊天框输入问题，系统自动检索相关知识并回答。
3. **出题测验**：输入「帮我出题」或「quiz」触发 Quiz 模式，系统出 2 道选择题并实时反馈掌握度变化。
4. **知识盲区诊断**：输入「分析我的知识盲区」，系统结合你的历史掌握度数据生成诊断报告。
