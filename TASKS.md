# 任务清单

## Step 0: 基础设施

- [x] 搭建 FastAPI 项目骨架
- [x] Milvus Standalone 部署 (docker-compose)
- [x] SQLite 数据库初始化 (user_mastery + quiz_sessions 建表)

## Step 1: Services 层

- [x] `embedding.py` — bge-m3, 1024维, 中英文多语言, 懒加载单例
- [x] `reranker.py` — bge-reranker-v2-m3, cross-encoder, fp16
- [x] `llm_client.py` — Qwen3.6-Plus, chat() 非流式 + stream_chat() 流式, 支持 tool-calling
- [x] `config.py` 更新 — embedding 模型路径改为 bge-m3
- [x] `doc_processor.py` — MinerU(PDF) + python-docx(Word) + 直读(Markdown), 两刀切分块, 概念抽取

## Step 2: Ingestion Pipeline

- [x] `milvus_client.py` — child_chunks Collection(HNSW+IP+SPARSE_INVERTED_INDEX), insert/search/hybrid_search/delete_by_source 封装
- [x] 向量化写入 Milvus — 调用 embedding_service 编码 child chunks, 写入 Milvus
- [x] 概念写入 knowledge_concepts 表 — Ingestion 时提取 topic, 写入 SQLite (crud.py upsert_concepts)
- [x] `ingest.py` API 路由 — 串联解析→向量化→写入 Milvus + knowledge_concepts
- [ ] ~~`notion_loader.py`~~ — 已取消, 不做 Notion 文档

## Step 3: Tools 层

- [x] `rag_search.py` — dense+sparse 混合检索(RRF融合) + Rerank, 输入 query/top_k, 输出 ranked chunks + source info
- [x] `query_decompose.py` — LLM 拆解复杂问题为 2-4 个子查询列表, JSON 数组输出, 解析失败 fallback 原始 query
- [x] `read_mastery.py` — 读取用户掌握度, 含时间衰减 effective = score × exp(-λ×Δt), 返回 MasteryRecord 列表
- [x] `update_mastery.py` — 二元判分 is_correct: 答对+0.1/答错-0.15 (基于 effective_mastery), 更新 score + last_reviewed, 首次默认 0.3
- [x] `search_concepts.py` — 查询 knowledge_concepts 表, 支持 scope(source_doc) 和 keyword(topic LIKE) 过滤

## Step 4: SubAgents

- [x] `state.py` — OverallState + QAState + QuizState + GapState, 各 SubAgent 独立 State
- [x] `qa_agent.py` — Plan & Execute + ReAct 混合架构: Planner(结构化JSON plan) → Executor(ReAct loop per step) → Synthesize → QC(max 2 retries, 不通过带反馈重规划)
- [x] `quiz_agent.py` — Agent 驱动 10 题 Quiz, tool-calling (rag_search + read_mastery + update_mastery + search_concepts), 多轮对话 message_history 传递, 启发式判断 session 结束
- [x] `gap_agent.py` — Agent 驱动诊断, tool-calling (read_mastery + search_concepts), LLM 执行 cross compare 规则 + 生成报告, 支持 scope 过滤

## Step 5: Orchestrator

- [x] `state.py` 扩展 — OverallState 增加 plan, step_results, current_step_index, replan_action, replan_count 字段
- [x] `orchestrator.py` — Plan & Execute with Replan 架构: plan_node (LLM few-shot 意图识别+计划生成, 支持 qa/gap_analysis/quiz 三种意图) + execute_step (单步执行) + replan_node (LLM 审视结果, continue/revise/finish) + synthesize_node (多结果汇总) + passthrough_node (单结果直传)
- [x] `graph.py` — LangGraph StateGraph: plan → route → execute_step → replan → route(execute/single/multi) → execute_step|passthrough|synthesize → END
- [x] `quiz_agent.py` 重构 — 从 10 题多轮 session 改为单次出 2 道选择题（RAG 检索出题）, 作为 orchestrator plan 的普通 step
- [x] 测试 — 39 个单元测试 + 集成测试全部通过 (含 replan continue/revise/finish、quiz dispatch、graph flow)

## Step 5.5: Agent 记忆机制

- [x] `config.py` — 记忆相关配置 (短期容量/整合数量/长期 Top-K/相似度阈值)
- [x] `models.py` — ConversationHistory 模型 (短期对话记录表)
- [x] `crud.py` — 对话历史 CRUD (add_round/get_recent/get_oldest/delete/count)
- [x] `milvus_client.py` — MemoryMilvusService (conversation_memory Collection, hybrid search)
- [x] `state.py` — OverallState 新增 memory_context 字段
- [x] `memory_manager.py` — 记忆管理核心 (load_memory_context + save_conversation + consolidate)
- [x] `memory_manager.py` 回归修复 — 长期记忆检索/写入改为使用 `EmbeddingOutput.dense/sparse`，修正实现与测试接口不一致问题
- [x] `orchestrator.py` — plan_node 注入记忆上下文
- [x] `graph.py` — memory_node 加入 LangGraph + run_orchestrator 保存对话
- [x] `main.py` — 启动时初始化 memory_milvus_service
- [x] 测试 — 15 CRUD 单元测试 + 13 memory_manager 测试 (含 consolidation) + 33 orchestrator 测试 (含 Replan loop), 61 总测试全过
- [ ] 重新验证 memory 相关测试 — 当前环境缺少 `aiosqlite` 且 `pip` 因 SSL 不可用无法补装，待修复环境后重跑

## Step 6: API 层

- [x] `chat.py` — 统一对话 API, POST → orchestrator → JSON 响应 (QA/Gap/Quiz 全走此入口)
- [x] `quiz.py` — Quiz 判分 API (不走 orchestrator): 接收用户答案 → 规则比对字母 → update_mastery → 返回结果
- [x] `schemas.py` — Pydantic request/response models (ChatRequest/Response, QuizAnswerRequest/Response)
- [x] 二元判分改造 — update_mastery + crud 从三档 quality(high/medium/low) 改为 is_correct: bool, 答对+0.1/答错-0.15

## ~~Step 9: 对话记忆模块~~ (已由 Step 5.5 + Step 6 覆盖)

- [x] API 层维护 message history — run_orchestrator 自动 save_conversation, memory_node 自动加载
- [x] 上下文引用 — 记忆上下文已注入 plan_node, LLM 天然支持指代理解

## Step 7: Frontend

- [x] React + Vite + TypeScript 项目搭建, Tailwind CSS v4
- [x] `Layout.tsx` — 左侧 sidebar (聊天问答 + 文档上传) + 右侧内容区
- [x] `ChatPanel.tsx` — 统一对话界面 (QA/Quiz/Gap 三合一), 首屏能力卡片, quiz 意图自动弹窗
- [x] `QuizModal.tsx` — Quiz MCQ 弹窗: 选项卡片 + 判分 + mastery 变化 + 完成总结
- [x] `FileUpload.tsx` — 文档拖拽/点击上传, 解析结果展示
- [x] `api.ts` + `types.ts` — API 客户端 + TypeScript 类型定义
- [x] `App.tsx` — 两面板切换 (chat / upload)

## Step 8: 评估 & 优化

- [ ] 构建评估数据集 (QA pairs, 复杂度标注)
- [ ] 检索评估 — Recall@K, MRR, Reranker Lift
- [ ] 生成评估 — Faithfulness, Answer Relevance, Citation Accuracy
- [ ] 多跳评估 — Decomposition Quality, Synthesis Coherence, Multi-hop Lift
- [ ] Quiz 评估 — Answer Eval Accuracy, Mastery Convergence, Topic Coverage
- [ ] Gap Analysis 评估 — Gap Detection Precision/Recall
- [ ] 端到端评估 — E2E Latency, Intent Classification Acc
- [ ] Prompt 调优
- [ ] Docker 化部署
- [ ] README + 项目文档 (含评估结果)
