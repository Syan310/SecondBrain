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

## 2026-04-04: Step 3 Tools 层完成

完成剩余 4 个 Tools 实现:

1. **query_decompose.py** — 调用 LLM 将复杂问题拆解为 2-4 个独立子查询 (JSON 数组), temperature=0.3, 解析失败 fallback 原始 query
2. **read_mastery.py** — 读取 UserMastery + 计算 effective_mastery = score × exp(-λ×Δt), 返回 MasteryRecord dataclass
3. **update_mastery.py** — 基于 effective_mastery 应用 +0.1(答对)/-0.15(答错), 首次默认 0.3, 调用 crud.upsert_mastery 持久化
4. **search_concepts.py** — 查询 knowledge_concepts 表, 支持 scope(source_doc精确匹配) + keyword(topic模糊匹配)

同时扩展了 crud.py, 新增:
- `get_mastery()` — 按 user_id + optional topics 查询
- `upsert_mastery()` — 创建或更新 mastery 记录
- `search_concepts()` — scope/keyword 过滤查询

设计决策:
- update_mastery 的 new_score 由 tool 层计算后传给 crud 层, crud 层不做业务逻辑
- read_mastery 的 `_compute_effective_mastery` 抽为内部函数, update_mastery 复用

## 2026-04-04: Step 4 SubAgents 实现

### 设计决策

1. **Agent 模式而非固定 Pipeline**: 三个 SubAgent 都采用 ReAct 循环（LLM 自行决定 tool 调用），而非 LangGraph 固定节点 DAG。原因是需要灵活处理边界情况。

2. **Tool 绑定方式**: 不使用 LangGraph 的 create_react_agent，而是手写 ReAct 循环直接调用 llm_client.chat(tools=...)。原因:
   - 我们的 LLM 是 Qwen3-Plus (OpenAI-compatible)，不是 langchain ChatModel
   - DB 工具需要注入 AsyncSession，用工厂函数 `_build_tool_executor(session)` 闭包绑定
   - QA agent 的 rag_search 是同步的（本地模型），其余是 async

3. **Quiz 多轮设计**: `run_quiz_agent()` 处理单轮交互，caller 通过 `message_history` 参数维护跨轮上下文。每次调用 = 一次用户输入 → agent 处理（可能多次 tool call）→ 一次文本回复。

4. **Quiz tool 分配更新**: 原设计只有 read_mastery + update_mastery，新增 rag_search（检索出题素材）+ search_concepts（无指定 topic 时获取全部 topic 列表）。

5. **State 定义**: 各 SubAgent 独立 State (QAState/QuizState/GapState) + 顶层 OverallState。当前 State 主要用于类型文档化，实际 agent 通过 message_history 传递上下文。

### 实现文件
- `backend/app/agent/state.py` — 4 个 TypedDict State
- `backend/app/agent/subagents/qa_agent.py` — run_qa_agent(), ReAct loop, max 15 iterations
- `backend/app/agent/subagents/quiz_agent.py` — run_quiz_agent(), multi-turn, max 30 iterations
- `backend/app/agent/subagents/gap_agent.py` — run_gap_agent(), max 10 iterations

## 2026-04-04: Quiz 判分从二元改为三档质量评估

### 变更
- 原设计: is_correct (bool) → 答对 +0.1 / 答错 -0.15
- 新设计: quality ("low"|"medium"|"high") → low -0.15 / medium +0.05 / high +0.1
- medium + high 计入 correct_count，low 不计入

### 影响文件
- `crud.py` — upsert_mastery 参数 is_correct → quality
- `update_mastery.py` — QUALITY_DELTA dict，三档 delta 逻辑
- `quiz_agent.py` — tool schema (is_correct → quality enum)，system prompt 判分标准更新为三档
- `ARCHITECTURE.md` §6 更新规则

## 2026-04-04: QA SubAgent 重构为 Plan & Execute + ReAct 混合架构

### 变更
原设计: 纯 ReAct 循环，所有逻辑（decompose/RAG/QC）隐含在 system prompt 中由 LLM 自行推理
新设计: 三阶段显式分离

1. **Planner** (LLM, 无 tools): 分析问题 → 输出结构化 JSON plan `{needs_decompose, plan: [{step, query, needs_rag, purpose}], synthesis_needed}`
2. **Executor** (ReAct loop, 有 tools): 按 plan 逐步执行, 每步独立 ReAct 循环 (max 8 iterations/step)
3. **QC** (LLM, 无 tools): 评估 {passed, reason, suggestions}, 不通过带 feedback 回到 Planner 重规划 (max 2 retries)

### 设计理由
- Plan 显式化: plan 写入 state, 可观测可 log
- Planner/Executor/QC 角色分离, 各自 system prompt 更聚焦
- 宏观 Plan & Execute, 微观 ReAct, 项目同时体现两种 Agent 范式
- Synthesizer 单独抽出, 多步时综合子答案

### 实现细节
- Planner JSON 解析失败 → fallback 单步 plan
- QC JSON 解析失败 → default pass
- Executor 每步 max 8 iterations, 整体 max 4 步 × 8 = 32 次 LLM 调用上限
- 4 个独立 prompt: PLANNER / EXECUTOR / SYNTHESIZER / QC

## 2026-04-04 Step 5 Orchestrator 设计迭代

### 决策记录

1. **Orchestrator 采用 Plan & Execute 架构**：LLM few-shot 识别意图并生成结构化执行计划（steps + depends_on），支持多意图编排（并行/串行）。

2. **Quiz SubAgent 不经过 Orchestrator**：Quiz 是多轮交互，生命周期与单次问答不同，由 API 层直接调用。Orchestrator 识别到 quiz 意图时仅放入 `suggested_followup` 返回建议。

3. **只编排 QA + Gap**：讨论了 Summarize、Explain、Flashcard 等候选 SubAgent，最终认为：
   - Summarize/Explain 与 QA 意图边界模糊，更适合作为 QA 内部的生成策略
   - Flashcard 也是独立功能，不适合混合编排
   - QA + Gap 的组合最自然（诊断→讲解、问答+掌握度）

4. **串行步骤的参数改写**：当 step 有 depends_on 时，execute_node 将前序结果 + 原始 plan 交给 LLM 动态改写当前 step 的 params。

5. **Synthesizer 只在多结果时激活**：单 SubAgent 结果直接 passthrough，不浪费 LLM 调用。

6. **Python 3.9 兼容性**：为所有使用 `X | None` 语法的文件添加 `from __future__ import annotations`，State 中使用 `Optional[dict]` 以兼容 LangGraph 的 `get_type_hints()`。

## 2026-04-04: Quiz 模块重构 — 从独立 session 改为 Orchestrator step

### 决策
- Quiz 从独立的 10 题多轮 session 改为 Orchestrator plan 的普通 step
- 每次只出 2 道选择题，题目基于 RAG 检索内容生成
- Quiz 永远是 plan 的最后一步，可 depends_on 前序 QA/Gap 步骤

### 出题策略
- 有前序 QA step → 围绕 QA 讲解的主题出题
- 无特定主题 → 根据 mastery 弱点自动选题（effective_mastery 最低的 topics）
- RAG 检索相关内容作为出题依据，不是纯 LLM 出题

### 判分流程
- 判分不走 Orchestrator，走独立 API endpoint (`POST /quiz/answer`)
- 前端知道当前在答题状态，答案直接发到 quiz answer API
- 判分后调用 update_mastery（三档：high +0.1 / medium +0.05 / low -0.15）

### 动机
- 原来的 10 题循环感觉割裂，像"做卷子"而非"和 tutor 聊天"
- 新方案让 quiz 成为对话流的自然收尾（讲完就考，诊断完就练）
- 想继续复习 → 用户说"继续考我" → 新一轮 plan

### 移除
- `suggested_followup` 字段从 OverallState 中移除
- `quiz_passthrough` 节点从 graph 中移除
- Quiz SubAgent 的 10 题 session 循环、message_history、_check_quiz_complete 等逻辑全部移除

## 2026-04-04: Orchestrator 增加 Replan 循环

### 背景
原架构中 execute_node 一口气执行所有 plan steps，缺少标准 Plan & Execute 的 replan 环节。作为简历项目需要展示完整的 adaptive planning loop。

### 设计决策
- `execute_node` → `execute_step`: 改为每次只执行 1 个 step
- 新增 `replan_node`: 每个 step 执行后 LLM 审视结果，决定 continue/revise/finish
- `revise` 时 `revised_steps` 替换 `plan[current_step_index:]` 剩余部分，保留已完成步骤
- `max_replan_count=3` 防止无限循环
- 不做 `idx >= len(plan)` 的自动 finish，始终让 LLM 判断（确保 revise 可以追加新步骤）
- 支持 qa/gap_analysis/quiz 三种意图的 replan

### Graph 结构变化
```
旧: plan → execute_node(批量) → passthrough/synthesize → END
新: plan → execute_step → replan → route(execute/single/multi) → execute_step|passthrough|synthesize → END
```
