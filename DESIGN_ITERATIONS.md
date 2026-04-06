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

## 2026-04-04: Agent 记忆机制设计与实现

### 决策

为 Agent 添加短期记忆 + 长期记忆机制：
- **短期记忆**: SQLite `conversation_history` 表，存最近 20 轮对话原文 (user + assistant 最终回答)，全量注入 Orchestrator prompt
- **长期记忆**: Milvus `conversation_memory` Collection，存 LLM 结构化摘要 (100-200字自然语言 + topics)，按需 hybrid search Top-3

### 整合策略

短期超过 20 轮时，取最早 10 轮 → LLM 摘要压缩 → 向量化存入 Milvus → 删除已整合旧轮次。Queue 式 FIFO。

### 作用范围

仅在 Orchestrator 层统一管理（memory_node 读取 → plan_node 注入），SubAgent 透传。Quiz 不使用（已有 mastery）。

### 关键设计选择

1. 长期记忆存 LLM 摘要而非原始对话 — 信噪比高，向量检索效果好
2. 按需检索而非全量注入 — 长期记忆持续积累，全量注入会爆 context + 噪音干扰
3. 新建独立 Milvus Collection — 不与 child_chunks 混用，schema 和检索逻辑完全不同
4. 短期用 SQLite 而非内存 — 服务重启后不丢失，学习场景用户可能隔天继续

### 涉及文件

config.py, models.py, crud.py, milvus_client.py, state.py, memory_manager.py, orchestrator.py, graph.py, main.py

## 2026-04-05: Step 6 API 层 + 二元判分改造

### 决策
1. **取消 SSE 流式**：方案A 下 orchestrator 一次性跑完再返回，SSE 无实际收益。改为普通 POST → JSON 响应，后续 Step 8 优化阶段再考虑流式。
2. **三档 quality 改为二元 is_correct**：MCQ 场景下字母比对足够，无需 LLM 判分。全链路改造：update_mastery.py (quality→is_correct) + crud.py upsert_mastery (quality→is_correct)。Delta 沿用原始规格：答对+0.1 / 答错-0.15。
3. **chat.py 统一入口**：QA / Gap / Quiz 全走 orchestrator，quiz_answer 独立端点只做判分。

### 实现
- `schemas.py` — ChatRequest/Response, QuizAnswerRequest/Response, MasteryUpdate
- `chat.py` — POST /api/chat/ → run_orchestrator → JSON
- `quiz.py` — POST /api/quiz/answer → 字母比对 → update_mastery → JSON
- `update_mastery.py` — QUALITY_DELTA dict 改为 CORRECT_DELTA/INCORRECT_DELTA 常量
- `crud.py` — upsert_mastery 参数 quality:str → is_correct:bool

## 2026-04-05: Step 7 Frontend

### 决策
1. **Tailwind CSS v4** + @tailwindcss/vite 插件，零配置文件
2. **左侧 sidebar + 右侧内容区** 布局，4 面板：聊天/Quiz/盲区诊断/文档上传
3. **Quiz 自动切换**：ChatPanel 检测 step_results 中 intent=quiz → 解析 JSON → 传递 QuizData → 自动切到 QuizPanel
4. **Vite proxy**：开发环境 /api 请求代理到 localhost:8000，避免 CORS
5. **无 SSE**：与 Step 6 一致，普通 POST → JSON

### 实现
- Layout.tsx — sidebar 导航组件，高亮当前面板，quiz badge
- ChatPanel.tsx — 聊天气泡 UI，发送/接收/loading 状态，quiz 意图检测
- QuizPanel.tsx — MCQ 选项卡片，提交判分，正误反馈+掌握度变化，完成总结页
- GapAnalysis.tsx — scope 输入框 + 诊断按钮 + 报告展示
- FileUpload.tsx — 拖拽/点击上传 + 解析结果展示
- api.ts — sendChat / submitQuizAnswer / uploadFile 三个 typed 函数
- types.ts — 前后端共享类型定义

## 2026-04-05: 多文件上传 + 文档管理 + 聊天持久化 + 无文档 fallback

### 决策

1. **多文件上传**: 前端支持多选/多拖拽, 后端 `POST /ingest/upload` 接收 `List[UploadFile]`, 并行解析(asyncio.gather + to_thread), 串行 embedding+写入(避免 OOM)
2. **文档管理**: 新增 `GET /ingest/documents`(从 knowledge_concepts 按 source_doc 聚合) + `DELETE /ingest/documents/{filename}`(Milvus chunks + SQLite concepts 同时删除), 前端 FileUpload 页面持久化显示已上传文档列表, 每个文档可删除
3. **聊天历史持久化**: 新增 `GET /chat/history`(复用已有 get_recent_conversations CRUD), 前端 ChatPanel mount 时加载历史消息, 刷新后不丢失
4. **无文档 fallback**: QA Agent 入口检查 `milvus_service.count()`, 为 0 时跳过整个 Plan/Execute/QC 流程, 直接纯 LLM chat 回答 + 提示"知识库为空, 以下为通用回答"
5. **不做 session 分离**: 用户只在一个对话框里对话, 短期记忆按 user_id 全量管理, 不按 session 隔离

### 取消

- Session 分离方案: 最初计划对 CRUD 全部加 session_id 过滤, 后决定不做, 保持单对话框模式

## 2026-04-05: 文档渐进式披露重构

### 背景
ARCHITECTURE.md 膨胀到 668 行，每次新对话冷启动需读取全文消耗大量 token。

### 决策
采用三层渐进式披露架构：
1. **CLAUDE.md** — 项目快照（10行以内）+ 文档索引表，新对话只读这个 + TASKS.md
2. **ARCHITECTURE.md** — 精简为系统架构图 + 技术栈 + 文档索引，~60行
3. **docs/*.md** — 7 个模块文档，每个含"概要"（1-2句）+"详细设计"，按需加载

### 拆分结果
| 文件 | 内容 |
|------|------|
| docs/agents.md | Agent 分层架构 |
| docs/langgraph.md | LangGraph 图结构设计 |
| docs/tools.md | Tool 定义与分配 |
| docs/data-storage.md | 数据流、存储、SQLite 表结构、Mastery 公式 |
| docs/memory.md | 短期/长期记忆机制 |
| docs/evaluation.md | 评估指标体系 |
| docs/directory.md | 项目目录结构 |

### 效果
- 冷启动 token 消耗: ~668行 → ~60行 (ARCHITECTURE.md) + ~100行 (TASKS.md)
- 按需加载: 修改特定模块时才读对应 docs/*.md

## 2026-04-05

PDF 入库依赖与代码对齐：`requirements.txt` 由 `mineru[core]` 改为 `magic-pdf[full]`，因 `doc_processor._parse_pdf` 使用 `magic_pdf` 命名空间，而 `mineru` 2.x wheel 不包含该模块。

## 2026-04-05: Step 8 评估方案重设计

**变更**: 重构评估指标体系，从 6 维 20+ 指标精简为 RAG + Agent 两条线 13 个核心指标。

**决策**:
1. RAG 评估采用 RAGAS 框架（业界标准），核心 4 指标：Context Precision/Recall + Faithfulness + Answer Relevancy
2. Agent 评估采用任务驱动方式：Intent Acc + Tool Selection Acc + Task Success Rate
3. Judge 模型选用 Kimi-K2.5（避免 Qwen 自评偏差）
4. 砍掉不可离线测量的指标：Mastery Convergence, Decay Effectiveness, Adaptive Accuracy, Session Completion Rate
5. 评估数据集半自动生成（LLM 从 chunks 生成候选 → 人工审核）

**原因**: 原方案指标过多且部分需纵向用户数据无法离线评估，RAGAS 是 RAG 评估事实标准，可直接 pip install 使用。

## 2026-04-05 评测分档决策
- 目标：在不拆分现有评测数据集文件的前提下，缩短开发期评测耗时。
- 决策：评测脚本统一支持 smoke / standard / full 三档。
- 口径：RAG smoke=4单跳+2多跳，standard=10单跳+5多跳，full=全量30；Agent intent smoke=6，standard=15，full=30；Agent QA smoke=3单跳+1多跳，standard=10单跳+5多跳，full=全量。
- 原因：保留同一数据集与同一评测逻辑，仅调整抽样规模，便于开发回归与阶段性验收共存。

## 2026-04-05 评测结果输出单文件

RAG/Agent 评测由「子目录时间戳 JSON + results 根目录 latest」双写改为仅在 `results/rag/`、`results/agent/` 各写一份；报告 meta 使用 `output_file` 替代 `output_archive_file` / `output_latest_file`。

## 2026-04-05 评测时间改为洛杉矶

`eval_rag` / `eval_agent` 的报告文件名与 `meta.started_at` / `meta.finished_at` 改为 `America/Los_Angeles`；`meta` 增加 `timezone`，移除 `*_utc` 键名。

## 2026-04-06 Eval Sampling + GPT-5.4 Judge
- 决定将 Step 8 的 profile 子集选择从“按文件顺序截前 N 条”改为“确定性分层 round-robin 采样”。
- QA 评测按 single-hop / multi-hop 分桶后，再按来源文档或来源组合做稳定轮转，避免结果被数据集排序污染。
- Intent 评测按 qa / quiz / gap_analysis 三类均衡抽样，确保 smoke 与 standard 档位都覆盖完整路由面。
- Judge 评测链路统一收口到 OpenAI GPT-5.4，并按 reasoning 模型参数契约构造请求；Agent judge 直接使用 max_completion_tokens，RAGAS judge 复用其 OpenAI adapter 的 GPT-5 参数映射。

## 2026-04-06 评测脚本全面审查与修复

### 背景
对 `eval_rag.py` 和 `eval_agent.py` 进行全方位代码审查，发现 3 类严重问题 + 4 类重要问题 + 5 类代码质量问题。

### 严重问题修复
1. **RAGAS 依赖未声明** → `requirements.txt` 添加 `ragas>=0.2.5`
2. **RAGAS 模块级导入阻止模块加载** → 所有 RAGAS 导入（`BaseRagasEmbeddings`、metrics、`llm_factory`）全部延迟到 `run_ragas_evaluation()` / `_build_bge_ragas_embeddings()` 函数内部；`BGEm3RagasEmbeddings` 类也移入工厂函数
3. **RAGAS 私有 API 使用** → `_Faithfulness` 等下划线前缀类替换为公开 API：`Faithfulness`, `AnswerRelevancy`, `LLMContextPrecisionWithReference`, `LLMContextRecall`（确认 `llm_factory` 是 RAGAS 官方推荐方式，`LangchainLLMWrapper` 反而已被废弃）

### 健壮性修复
1. **classify_intent 无异常处理** → try/except 包裹 plan_node，失败时返回 error 字段并继续评测
2. **Orchestrator 无超时** → `asyncio.wait_for(run_orchestrator(...), timeout=120)` + 超时计入 orchestrator_error
3. **Latency 含错误 case** → 拆分 `ok_latencies` / `all_latencies`，P50/P95 默认仅用成功 case；报告同时输出 `_all` 版本供参考

### 代码质量修复
1. **重复代码** → 提取 `evaluation/eval_utils.py`（`configure_run_logging` + `safe_p50` + `safe_p95` + `EVAL_TZ`）
2. **P95 计算不一致** → 统一使用 `safe_p95`（含 `min(..., len-1)` 安全边界）
3. **隐式状态传递** → `eval_rag.py` 中 `item["_rag_result"]` 修改 dataset dict 的做法改为独立 `rag_results_map` dict
4. **PEP8** → 修复 `eval_agent.py` 常量与函数定义间缺少空行

## 2026-04-06 — Judge model switched to gpt-5.4-mini with reasoning disabled

- Decision: evaluation judge default changed from `gpt-5.4` to `gpt-5.4-mini`.
- Rationale: judge workloads are rubric-following and structured-output oriented; prefer lower cost and fewer parameter-compatibility issues over additional reasoning depth.
- Request shape: use OpenAI Chat Completions with `reasoning={"effort": "none"}`.
- Robustness rule: direct judge calls should avoid explicit max token caps unless a concrete truncation issue appears in eval logs.
- RAGAS compatibility: keep a local OpenAI compat proxy so third-party `max_tokens` defaults are rewritten to `max_completion_tokens` for GPT-5 models.
- Correction: the decision above means `gpt-5.4` -> `gpt-5.4-mini`.
- Implementation note: in the RAGAS path, the OpenAI compatibility proxy must wrap the Instructor client returned by `llm_factory`, not the raw OpenAI client passed into `llm_factory`; otherwise Instructor patching can yield a broken client object.

## 2026-04-06 — eval_rag returned to the official ragas integration path

- Decision: stop layering project-specific OpenAI compatibility proxies into the RAGAS path.
- New structure: keep project eval orchestration in `evaluation/eval_rag.py`, and move RAGAS-specific wiring into `evaluation/ragas_runner.py`.
- Official integration path: `AsyncOpenAI` + `ragas.llms.llm_factory(...)` + `ragas.aevaluate(...)`.
- Compatibility stance: align runtime dependency with the docs-backed API by raising the floor to `ragas>=0.4.3`.
- Goal: preserve current eval outputs (`ragas_metric_status`, per-sample rows, multi-hop lift, IR metrics) while making the judge setup simpler and easier to debug.
- Correction: the new structure keeps project eval orchestration in `evaluation/eval_rag.py` and moves RAGAS-specific wiring into `evaluation/ragas_runner.py`.

## 2026-04-06 — Judge default changed to gpt-4.1 for compatibility

- Decision: switch the default evaluation judge from `gpt-5.4-mini` to `gpt-4.1`.
- Reason: `ragas 0.4.3` does not correctly detect dotted GPT-5 model names like `gpt-5.4-mini`, which leaves its internal `max_tokens` mapping inactive and causes OpenAI 400 errors.
- Request policy: non-GPT-5 judges use standard Chat Completions params; GPT-5-only fields such as `reasoning` should not be sent on the `gpt-4.1` path.
- Goal: prioritize judge reliability over reasoning-specific behavior in the evaluation stack.
- Correction: the switch is `gpt-5.4-mini` -> `gpt-4.1`.

## 2026-04-06 — evaluation judge moved to Kimi K2.5 non-thinking mode

- Decision: switch the shared evaluation judge to `kimi-k2.5` and stop using OpenAI models for the current eval stack.
- Agent judge policy: send `thinking={"type":"disabled"}` and otherwise keep the request on the plain Chat Completions path.
- RAGAS judge policy: keep the official `AsyncOpenAI` + `llm_factory(...)` + `aevaluate(...)` flow, but override `ragas` defaults with Kimi-compatible kwargs: `thinking={"type":"disabled"}`, `temperature=0.6`, `top_p=0.95`, `max_tokens=32768`.
- Rationale: Kimi accepts `max_tokens` on the non-thinking path, while the recent OpenAI path kept failing on model/provider parameter contract mismatches.
- Correction: the shared evaluation judge was switched to `kimi-k2.5`.

---

## 2026-04-06: 基准测试完成 & 问题识别

### 背景
完成 Judge 模型从 ChatGPT 到 Kimi K2.5 的全面迁移后，首次跑通 RAG + Agent smoke profile 评估，建立基准数据集。

### Judge 模型迁移要点
- OpenAI SDK 不支持 `thinking` 顶层参数 → 改用 `extra_body` 传入
- RAGAS `InstructorModelArgs` 默认 `temperature=0.01` 被 kimi-k2.5 拒绝 → 显式传 `temperature=0.6, top_p=0.95`
- RAGAS 默认 `max_tokens=1024` 导致结构化 JSON 截断 → 提升到 `8192`
- `judge_base_url` 从 `api.moonshot.ai` 更正为官方 `api.moonshot.cn`

### 基准结果
- RAG: 检索质量优秀 (Recall@K=1.0, MRR=1.0)，但延迟严重超标 (P95=38s vs 目标<500ms)，faithfulness 偏低 (0.748)
- Agent: Intent 100% 正确，但 QA E2E 仅 25% pass (3/4 超时)

### 识别的核心问题 (P0-P3)
- P0: Reranker CPU 推理 18-29s/query，占 RAG 延迟 60-75%
- P1: Agent 120s 超时率 75%，根因是 P0 + QA Agent 多步分解叠加
- P2: Multi-hop faithfulness=0.634，多跳幻觉率高
- P3: Reranker lift=0，smoke profile 样本太少无法验证

### 决策
以当前结果作为基准，清理历史测试数据，进入优化阶段。
