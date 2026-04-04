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
- [x] `update_mastery.py` — 答对+0.1/答错-0.15 (基于 effective_mastery), 更新 score + last_reviewed, 首次默认 0.3
- [x] `search_concepts.py` — 查询 knowledge_concepts 表, 支持 scope(source_doc) 和 keyword(topic LIKE) 过滤

## Step 4: SubAgents

- [x] `state.py` — OverallState + QAState + QuizState + GapState, 各 SubAgent 独立 State
- [x] `qa_agent.py` — Plan & Execute + ReAct 混合架构: Planner(结构化JSON plan) → Executor(ReAct loop per step) → Synthesize → QC(max 2 retries, 不通过带反馈重规划)
- [x] `quiz_agent.py` — Agent 驱动 10 题 Quiz, tool-calling (rag_search + read_mastery + update_mastery + search_concepts), 多轮对话 message_history 传递, 启发式判断 session 结束
- [x] `gap_agent.py` — Agent 驱动诊断, tool-calling (read_mastery + search_concepts), LLM 执行 cross compare 规则 + 生成报告, 支持 scope 过滤

## Step 5: Orchestrator

- [x] `state.py` 扩展 — OverallState 增加 plan, step_results, suggested_followup 字段
- [x] `orchestrator.py` — Plan & Execute 架构: plan_node (LLM few-shot 意图识别+计划生成) + execute_node (按依赖顺序调用 SubAgent, 串行时 LLM 改写 params) + synthesize_node (多结果汇总) + passthrough_node (单结果直传)
- [x] `graph.py` — LangGraph StateGraph: plan → route(has_steps/no_steps) → execute → route(single/multi) → passthrough|synthesize → END; quiz-only → quiz_passthrough → END
- [x] 测试 — 24 个单元测试 + 集成测试全部通过 (plan_node, execute_node, synthesize_node, routing, graph flow)

## Step 6: API 层

- [ ] `chat.py` — QA 问答 API, SSE 流式输出
- [ ] `quiz.py` — Quiz API, 开始/作答/结束 session
- [ ] `gap.py` — 知识盲区诊断 API
- [ ] API Schema 定义 — Pydantic request/response models

## Step 7: Frontend

- [ ] React + Vite 项目搭建
- [ ] `ChatPanel.tsx` — QA 问答界面, SSE 流式接收
- [ ] `QuizPanel.tsx` — Quiz 复习界面, 多轮交互
- [ ] `GapAnalysis.tsx` — 知识盲区诊断界面
- [ ] `FileUpload.tsx` — 文档上传功能

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
