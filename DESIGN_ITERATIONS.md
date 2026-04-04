# 设计迭代记录

## 2026-04-03: Agent Tool 定义与 Gap Analysis 设计决策

### 讨论背景

确定各 SubAgent 需要哪些 tools，以及 Gap Analysis 的 Search Concepts 数据来源和 Cross Compare 实现方式。

### 初始想法

- 用户最初只想到 RAG 作为 tool

### 讨论过程

1. **Tool 候选梳理**：列出了 rag_search, read_mastery, update_mastery, search_concepts, query_decompose 五个 tool
2. **排除项**：Reranker（内嵌到 rag_search）、LLM Generate/Judge（agent 自身能力）、Sufficiency Check/Complexity Judge（prompt 引导）
3. **Gap Analysis Search Concepts 数据来源**：
   - 方案 A: Ingestion 时从 Heading 提取 topic，写入 knowledge_concepts 表 ✅ 采用
   - 方案 B: 运行时 LLM 动态聚合 ✗ 不稳定、耗时
   - 方案 C: 复用 RAG 检索 ✗ 需求不同（枚举 vs 召回）
4. **Cross Compare 实现方式**：采用纯规则（mastery 阈值判定），不需要 LLM
5. **QA SubAgent 风格**：确认为 ReAct 风格，query_decompose 作为 tool 由 agent 自行决定调用

### 最终方案

**5 个 Tools**:
| Tool | 使用方 |
|------|--------|
| `rag_search` | QA SubAgent |
| `query_decompose` | QA SubAgent |
| `read_mastery` | Quiz SubAgent, Gap Analysis SubAgent |
| `update_mastery` | Quiz SubAgent |
| `search_concepts` | Gap Analysis SubAgent |

**Gap Analysis Cross Compare 规则**:
- knowledge_concepts 有 + user_mastery 无记录 → 未接触盲区（高优先级）
- effective_mastery < 0.3 → 薄弱盲区（高优先级）
- effective_mastery 0.3~0.5 → 需巩固（中优先级）
- effective_mastery >= 0.5 → 掌握中/已掌握（低优先级）

**新增 SQLite 表**: `knowledge_concepts`（topic, source_doc, heading_path, chunk_count）

### 文档变更

- `ARCHITECTURE.md`: 新增第 4 章 Agent Tool 定义与分配、knowledge_concepts 表、更新 Ingestion Pipeline 描述、更新目录结构（nodes/ → tools/ + nodes/）

## 2026-04-03: 确定实现流程（自底向上）

### 最终方案

8 步实现流程：Services → Ingestion → Tools → SubAgents → Orchestrator → API → Frontend → 评估优化。替换原有的 Phase 1-5 分阶段规划，改为更细粒度的依赖驱动顺序。

### 文档变更

- `ARCHITECTURE.md`: 原第 9 章「开发阶段规划」替换为「实现流程（自底向上）」，含当前进度 checklist

## 2026-04-03: Services 层实现

### 决策记录

1. **Embedding**: bge-base-zh-v1.5 → `BAAI/bge-m3` (1024-dim, 中英文多语言)
2. **Reranker**: `BAAI/bge-reranker-v2-m3` cross-encoder, FlagEmbedding 加载, fp16
3. **LLM**: Qwen3-Plus via DashScope OpenAI-compatible API, 统一模型不做多模型切换
4. **流式策略**: agent 内部 tool-calling 用 `chat()` 非流式, 用户响应用 `stream_chat()` 流式

### 文档变更

- `ARCHITECTURE.md`: 模型引用 bge-base-zh → bge-m3, Step 1 标记完成
- `config.py`: embedding_model_path 更新为 bge-m3

## 2026-04-03: doc_processor 设计与实现

### 决策记录

1. **文档解析**: MinerU (Python API, PymuDocDataset) 解析 PDF → Markdown; python-docx 解析 Word → Markdown; .md 直读
2. **分块策略**: 两刀切 — 第一刀按文档中最大 heading 级别切 parent, 第二刀 parent 内部按子 heading + 750 chars 上限切 child
3. **概念抽取**: 从 heading_path 提取, 通用词 (概述/介绍等) 自动拼接父 heading 避免重复
4. **取消 Notion**: 不做 Notion 文档支持, 只支持 PDF/Word/Markdown
5. **依赖变更**: PyPDF2 → mineru[core] (MinerU 内含 PDF 解析能力)

### 文档变更

- `TASKS.md`: doc_processor 标记完成, 取消 notion_loader
- `requirements.txt`: PyPDF2 → mineru[core]

## 2026-04-03: Milvus Schema 设计决策

- **单 Collection 方案**：只存 child chunks，不单独存 parent full_text
- **去掉 parent 回溯**：parent chunk 可能过长导致上下文爆炸，当前阶段直接用命中的 child chunks 送 LLM
- **保留相邻扩展能力**：child 上保留 parent_id + chunk_index 字段，后续可通过取相邻 chunk 做 A/B 测试
- **索引**：HNSW + IP（BGE-m3 向量已归一化），source_doc 建 INVERTED 标量索引
- **text 字段存 Milvus**：reranker 需要原文做 cross-encoder 打分，避免二次 IO

## 2026-04-04: rag_search 混合检索实现

### 决策
- embedding.py 从 SentenceTransformer 切换到 FlagEmbedding.BGEM3FlagModel，同一模型解锁 dense + sparse 双输出
- Milvus collection 新增 sparse_embedding (SPARSE_FLOAT_VECTOR) 字段 + SPARSE_INVERTED_INDEX 索引
- hybrid_search 使用 Milvus 原生 AnnSearchRequest + RRFRanker(k=60) 融合双路召回
- rag_search tool: encode(dense+sparse) → hybrid_search(top_k×3 candidates) → rerank(top_k) → 返回 RankedChunk
- requirements.txt 移除 sentence-transformers，完全使用 FlagEmbedding

### 变更文件
- `backend/app/services/embedding.py` — BGEM3FlagModel, 新增 EmbeddingOutput dataclass, encode 返回 dense+sparse
- `backend/app/services/milvus_client.py` — 新增 sparse_embedding 字段, hybrid_search 方法, insert 增加 sparse_embeddings 参数
- `backend/app/agent/tools/rag_search.py` — 新建, 混合检索+rerank tool
- `backend/tests/test_rag_search.py` — 新建, mock 测试
