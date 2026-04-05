# SecondBrain：架构、数据流与工作流（详述）

> 本文档与代码库当前实现对齐，用于整体把握系统。精简入口见仓库根目录 [`ARCHITECTURE.md`](../ARCHITECTURE.md)。模块级细节仍可查阅 `docs/agents.md`、`docs/langgraph.md`、`docs/tools.md`、`docs/data-storage.md`、`docs/memory.md` 等。

---

## 1. 产品定位与能力边界

SecondBrain 是基于**个人上传文档**的 **RAG 问答 + 知识盲区诊断（Gap）+ 测验（Quiz）** 系统。核心特点：

- 文档经解析、分块、向量化后写入 **Milvus**，供检索与 RAG。
- 概念经 **LLM 抽取** 后写入 **SQLite**（`knowledge_concepts` + `concept_sources`），供 Gap、概念检索与文档维度过滤。
- 用户掌握度 **SQLite**（`user_mastery`），Quiz 判分后更新；读取时带**时间衰减**得到有效掌握度。
- 对话 **短期 SQLite + 长期 Milvus** 双轨记忆，在 Orchestrator 规划前注入上下文。

---

## 2. 逻辑分层架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Frontend (React + Vite + TypeScript + Tailwind v4)                      │
│  ChatPanel / QuizModal / FileUpload · 调用 /api/chat、/api/quiz、/api/ingest │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ HTTP (JSON / SSE)
┌───────────────────────────────────▼─────────────────────────────────────┐
│ FastAPI (`app/main.py`)                                                 │
│  /api/chat · /api/chat/stream · /api/chat/history                       │
│  /api/quiz/answer                                                       │
│  /api/ingest/upload · /api/ingest/documents · DELETE .../documents/{fn} │
│  /api/gap/analyze — 占位路由（实际 Gap 经 Orchestrator → Gap SubAgent）   │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────┐
│ LangGraph (`app/agent/graph.py`)                                        │
│  memory → plan → execute_step ⇄ replan → passthrough | synthesize → END │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│ QA SubAgent   │         │ Gap SubAgent    │         │ Quiz SubAgent   │
│ rag_search 等  │         │ read_mastery 等  │         │ 单次 2 题 MCQ    │
└───────────────┘         └─────────────────┘         └─────────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Tools (`app/agent/tools/`) · Services (`app/services/`) · DB CRUD       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 技术栈（实现级）

| 组件 | 说明 |
|------|------|
| **Embedding** | BGE-m3，1024 维，dense + sparse（`app/services/embedding.py`） |
| **Reranker** | BGE-reranker-v2-m3，fp16（`app/services/reranker.py`） |
| **主 LLM** | 默认 Qwen3.6-Plus（`config.py`：`llm_model`），chat / stream_chat / tool-calling |
| **评估 Judge LLM** | 配置项 Kimi K2.5（`judge_*`），用于规划中的 Step 8 评估，与主链路解耦 |
| **Milvus** | Standalone；主库集合默认名 `secondbrain_docs`；记忆集合 `conversation_memory` |
| **SQLite** | aiosqlite + SQLAlchemy Async（`app/db/`） |
| **PDF** | `magic_pdf`（`magic-pdf` 生态），与 PyPI `mineru` 2.x 非模块级兼容；依赖以 `requirements.txt` 为准 |
| **Word / Markdown** | python-docx、直读 Markdown（`doc_processor.py`） |
| **部署** | Docker Compose（etcd、minio、milvus、backend 等，见仓库 compose 配置） |

---

## 4. 存储与子系统

### 4.1 Milvus：`secondbrain_docs`（主知识库）

由 `MilvusService`（`app/services/milvus_client.py`）管理。单条 child chunk 逻辑字段：

- `child_id`（PK）、`parent_id`、`source_doc`（上传文件名）、`heading_path`、`chunk_index`
- `text`（供 rerank 打分）
- `embedding`（FLOAT_VECTOR, HNSW + IP）
- `sparse_embedding`（SPARSE_INVERTED_INDEX + IP）

检索侧对 query 做 **dense + sparse 混合检索**，**RRF（`RRFRanker`）** 融合，再可选 **cross-encoder rerank**（见 `rag_search` 工具）。

### 4.2 Milvus：`conversation_memory`（长期对话记忆）

由 `MemoryMilvusService` 管理。字段包括：`memory_id`、`user_id`、`summary`、`topics`（JSON 字符串）、`session_id`、`created_at`、`embedding`、`sparse_embedding`。按 `user_id` 过滤 + hybrid search，分数阈值见 `config.py` 的 `memory_similarity_threshold`。

### 4.3 SQLite 表（`app/db/models.py`）

| 表 | 用途 |
|----|------|
| **user_mastery** | 每用户每 topic 的 `mastery_score`、衰减系数 `decay_rate`、`total_attempts`、`correct_count`、`last_reviewed`；`(user_id, topic)` 唯一 |
| **knowledge_concepts** | 概念词条，`topic` 全局唯一 |
| **concept_sources** | 概念 ↔ 文档 ↔ **parent_chunk_id** 多对多；支持同一概念出现在多文档、多父块 |
| **conversation_history** | 短期对话：`role`（user/assistant）、`content`、`round_number`、按用户查询最近 N 轮 |
| **quiz_sessions** | Quiz 会话元数据（历史设计保留；当前 Orchestrator 内 Quiz 为单次出题，判分走独立 API） |

概念查询与文档列表：

- `list_documents`：按 `concept_sources.source_doc` 聚合 **distinct concept_id** 数量。
- `search_concepts`：`scope` = 按文档名过滤；`keyword` = topic ILIKE。
- 重入库：`delete_concepts_by_source` 删该文档下所有 source 行，并清理**无映射**的孤儿 `knowledge_concepts`。

---

## 5. 数据流一：文档入库（Ingestion）

**入口**：`POST /api/ingest/upload`（`app/api/routes/ingest.py`）。

### 5.1 阶段划分

1. **校验与落盘**：扩展名（pdf/docx/doc/md）、大小上限、文件名清洗（防路径穿越与危险字符）。
2. **并行解析**：每个文件 `asyncio.to_thread(_parse_file)`，CPU 密集不阻塞事件循环。
3. **串行嵌入与写入**（每个文件一轮）：避免多请求并发撑爆显存/内存；同一文件内对所有 **child** 批量 `embedding_service.encode_documents`。

### 5.2 解析与分块（`doc_processor.py`）

- **两刀切分**：先按最高级标题切成 **ParentChunk**，再在父块内按子标题 + 字符上限（默认子块约 750 字、overlap）切成 **ChildChunk**。
- Child 是 **Milvus 检索与嵌入** 单元；Parent 承载更广上下文（概念抽取等）。

### 5.3 写入顺序（单文件）

1. `milvus_service.delete_by_source(filename)` — 覆盖式重传。
2. `delete_concepts_by_source(session, filename)` — 清该文档的概念关联与孤儿概念。
3. `milvus_service.insert(...)` — 写入所有 child 的 dense/sparse 向量与元数据。
4. **`_extract_concepts_for_doc`** — 对每个 **parent** 调用 `concept_extractor.extract_concepts`：
   - 传入父块全文与当前库中已有 topic 列表（`get_all_concept_topics`），提示 LLM **复用已有命名**。
   - 结果经 `crud.upsert_concepts(topics, source_doc, parent_chunk_id)`：**get-or-create** `KnowledgeConcept`，并插入 `ConceptSource`（同一 concept + parent_chunk 去重）。

### 5.4 文档管理 API

- `GET /api/ingest/documents`：文档列表 + 概念数。
- `DELETE /api/ingest/documents/{filename}`：Milvus 按 source 删 + SQLite 概念清理。

---

## 6. 数据流二：RAG 与 Tools

Tools 位于 `app/agent/tools/`（由 SubAgents 通过 LLM tool-calling 或代码路径调用）：

| Tool | 作用 |
|------|------|
| **rag_search** | hybrid search + RRF + rerank；返回排序 chunk 与来源信息；Milvus 空库时上层可有 fallback 策略 |
| **query_decompose** | 复杂问句拆成 2–4 子查询；失败则回退原 query |
| **read_mastery** | 读 `user_mastery`，计算时间衰减后的 **effective** 掌握度 |
| **update_mastery** | Quiz 判分等：二元 `is_correct`，在 effective 基础上 **+0.1 / -0.15**（首次 topic 默认分见 `default_mastery_score`） |
| **search_concepts** | 封装 `crud.search_concepts`，供 Gap / 选题等 |

---

## 7. 数据流三：对话与 Orchestrator 工作流

### 7.1 API 入口

- **`POST /api/chat/`**：`run_orchestrator` → 完整图 `ainvoke` → 返回 `answer` + `step_results`。
- **`POST /api/chat/stream`**：`run_orchestrator_stream`，SSE：`status`（memory/plan/execute/replan/synthesize/output）、`token`、`final`。
- **`GET /api/chat/history`**：SQLite 最近 N 轮对话，供前端恢复 UI。

请求体常见字段：`user_query`、`user_id`、`session_id`（见 `schemas.py`）。

### 7.2 OverallState（`app/agent/state.py`）

Orchestrator 共享状态要点：

- `user_query`、`user_id`、`messages`
- `memory_context`：由 memory 节点注入的格式化字符串
- `plan`：`[{step, intent, params, depends_on}, ...]`
- `step_results`：`[{step, intent, result}, ...]`
- `current_step_index`、`replan_action`（continue / revise / finish）、`replan_count`
- `final_answer`

意图仅三类：**`qa`**、**`gap_analysis`**、**`quiz`**。规划 prompt 要求：**若含 quiz，必须为最后一步**，且可设 `depends_on` 以串联前序步骤。

### 7.3 LangGraph 拓扑（与代码一致）

```text
START → memory → plan → route_after_plan
  ├── no_steps → empty_response → END
  └── has_steps → execute_step → replan →
        ├── route_after_replan == execute → execute_step（循环）
        └── finish → route_to_output
              ├── single → passthrough → END
              └── multi → synthesize → END
```

- **memory**：`load_memory_context` — SQLite 近期轮次 + Milvus 长期摘要（hybrid + 阈值）。
- **plan**：`plan_node` — LLM 输出 JSON 计划；注入 `memory_context`。
- **execute_step**：每次执行 `plan[current_step_index]` 一步；若有依赖，用前序 `step_results` **LLM 改写 params**；分发 `run_qa_agent` / `run_gap_agent` / `run_quiz_agent`。
- **replan**：`replan_node` — continue / revise / finish；revise 可替换剩余步骤；有次数上限防死循环。
- **passthrough**：单步结果直接作为最终回答。
- **synthesize**：多步结果合并（非 stream 图内节点；stream 路径用 `build_synthesize_messages` + `llm_client.stream_chat`）。

### 7.4 请求结束：短期记忆与整合

- `run_orchestrator` / stream 结束时 **`save_conversation`**：
  - `get_next_round_number` → `add_conversation_round`（同一 `round_number` 写 user 与 assistant 两行）。
  - 若该用户轮次数 **超过** `memory_short_term_limit`：取最旧 `memory_consolidate_count` 轮，**LLM 压成摘要 + topics**，嵌入后写入 **Milvus conversation_memory**，再从 SQLite **删除**对应轮次。

### 7.5 SubAgent 行为摘要

- **QA**（`qa_agent.py`）：Planner → 逐步 Execute（ReAct + tools）→ Synthesize → QC（有限次重试）。
- **Gap**（`gap_agent.py`）：tools 读掌握度与概念，LLM 生成盲区报告；支持 `scope` 文档过滤。
- **Quiz**（`quiz_agent.py`）：在 Orchestrator 中为 **单次生成 2 道选择题**（非旧版 10 题长会话）；**不出分、不写 mastery**。

### 7.6 Quiz 判分独立链路

- **出题**：仅通过 **`/api/chat`**（Orchestrator → Quiz SubAgent），响应中携带题目结构供前端展示。
- **判分**：**`POST /api/quiz/answer`** — 选项字母规则比对 → `update_mastery` → 返回是否正确与 mastery 变化。**不经过 LangGraph。**

---

## 8. 前端工作流（简述）

- **ChatPanel**：统一发送聊天请求；可消费 stream 事件展示阶段与打字效果；加载 `/api/chat/history`。
- **QuizModal**：展示题目；提交答案时调 **`/api/quiz/answer`**。
- **FileUpload**：`/api/ingest/upload`、文档列表与删除。

---

## 9. 应用生命周期（后端）

`app/main.py` 的 `lifespan`：

1. `init_db()` — SQLite 建表/迁移逻辑由 `database` 模块负责。
2. `milvus_service.connect()` + `init_collection()` — 主文档集合（含稀疏字段检测与必要时重建）。
3. `memory_milvus_service.init_collection()` — 长期记忆集合。

---

## 10. 当前工程阶段（与 TASKS.md 对齐）

- **Step 1–7**：功能链路已打通（含记忆、统一 Chat、入库、前端）。
- **Step 8**：评估与优化 — 规划中包含 **RAGAS**（检索/生成指标）、**任务驱动 Agent 评估**、**Kimi K2.5 作 Judge**、E2E 延迟与数据集建设等；**执行状态以 `TASKS.md` 勾选为准**。

---

## 11. 关键文件索引

| 区域 | 路径 |
|------|------|
| LangGraph 编译与 run | `backend/app/agent/graph.py` |
| 规划 / 执行 / 重规划 | `backend/app/agent/nodes/orchestrator.py` |
| 记忆加载 / 保存 / 整合 | `backend/app/agent/nodes/memory_manager.py` |
| 状态 TypedDict | `backend/app/agent/state.py` |
| SubAgents | `backend/app/agent/subagents/*.py` |
| Tools | `backend/app/agent/tools/*.py` |
| Chat / Stream / History | `backend/app/api/routes/chat.py` |
| Ingest | `backend/app/api/routes/ingest.py` |
| Quiz 判分 | `backend/app/api/routes/quiz.py` |
| Milvus | `backend/app/services/milvus_client.py` |
| 文档解析 | `backend/app/services/doc_processor.py` |
| 概念 LLM 抽取 | `backend/app/services/concept_extractor.py` |
| 配置 | `backend/app/config.py` |

---

*文档生成依据：仓库内 `ARCHITECTURE.md`、`TASKS.md` 与 `backend/app` 下当前源码。若实现变更，请同步更新本文档。*
