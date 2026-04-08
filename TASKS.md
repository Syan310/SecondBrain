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
- [x] `doc_processor.py` — magic-pdf(`magic_pdf` API, PDF) + python-docx(Word) + 直读(Markdown), 两刀切分块, 概念抽取（依赖见 `requirements.txt`：`magic-pdf[full]`，非 PyPI `mineru` 包）

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
- [x] `qa_agent.py` — ReAct (max 2轮) + Reflection 架构: ReAct loop(rag_search tool-calling) → Reflection(质量审查, pass/fail) → fail 时带反馈再跑 1 轮 ReAct; 查询拆解由 Orchestrator plan_node 负责
- [x] `quiz_agent.py` — Agent 驱动 10 题 Quiz, tool-calling (rag_search + read_mastery + update_mastery + search_concepts), 多轮对话 message_history 传递, 启发式判断 session 结束
- [x] `gap_agent.py` — Agent 驱动诊断, tool-calling (read_mastery + search_concepts), LLM 执行 cross compare 规则 + 生成报告, 支持 scope 过滤

## Step 5: Orchestrator

- [x] `state.py` 扩展 — OverallState 增加 plan, step_results, current_step_index, replan_action, replan_count 字段; step_results 结构含 status/error 字段
- [x] `orchestrator.py` — Plan & Execute with Replan 架构: plan_node (原子任务拆解: 三步流程识别意图→检查可拆性→确定依赖, 每个step满足单一检索意图+单一概念聚焦+独立可回答, 支持 qa/gap_analysis/quiz 三种意图) + execute_batch (批量并发执行, Semaphore(2) 全局限流) + replan_node (LLM 审视结果, continue/revise/finish) + synthesize_node (多结果汇总) + passthrough_node (单结果直传)
- [x] `orchestrator.py` — 健壮性四项加固: ① execute_batch 全局 Semaphore(2) 跨请求并发限流; ② rewrite_params 输出按 intent schema 校验, 不合法回退原始 params; ③ 单步失败隔离 (status=ok/failed + error 字段), 不影响同批其他步; ④ plan 执行前结构校验 (_validate_plan: 编号唯一/依赖合法/Kahn 检环/params schema), replan revised_steps 同等校验 (_is_plan_structure_valid + params)
- [x] `graph.py` — LangGraph StateGraph: plan → route → execute_batch → replan → route(execute/single/multi) → execute_batch|passthrough|synthesize → END; streaming 路径支持 step 失败状态展示
- [x] `quiz_agent.py` 重构 — 从 10 题多轮 session 改为单次出 2 道选择题（RAG 检索出题）, 作为 orchestrator plan 的普通 step
- [x] 测试 — 65 个单元测试 + 集成测试全部通过 (含 execute_batch 并发/失败隔离、plan 结构校验、replan cycle/params 校验、streaming flow)

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
- [x] 重新验证 memory 相关测试 — 已修复测试隔离问题 (sys.modules mock 泄漏 + _COLLECTION 常量), 81/81 测试全过

## Step 6: API 层

- [x] `chat.py` — 统一对话 API, POST → orchestrator → JSON 响应 (QA/Gap/Quiz 全走此入口)
- [x] `chat.py` — GET /chat/history 历史对话加载 API
- [x] `quiz.py` — Quiz 判分 API (不走 orchestrator): 接收用户答案 → 规则比对字母 → update_mastery → 返回结果
- [x] `schemas.py` — Pydantic request/response models (ChatRequest/Response, QuizAnswerRequest/Response, ChatHistoryResponse, DocumentListResponse, MultiUploadResponse)
- [x] `ingest.py` — 多文件上传 (并行解析, 串行 embedding+写入) + GET /documents 文档列表 + DELETE /documents/{filename}
- [x] `crud.py` — 新增 list_documents() 按 source_doc 聚合概念数量
- [x] 二元判分改造 — update_mastery + crud 从三档 quality(high/medium/low) 改为 is_correct: bool, 答对+0.1/答错-0.15
- [x] QA Agent 无文档 fallback — Milvus 为空时跳过 RAG, 纯 LLM 通用问答 + 提示"知识库为空"

## ~~Step 9: 对话记忆模块~~ (已由 Step 5.5 + Step 6 覆盖)

- [x] API 层维护 message history — run_orchestrator 自动 save_conversation, memory_node 自动加载
- [x] 上下文引用 — 记忆上下文已注入 plan_node, LLM 天然支持指代理解

## Step 7: Frontend

- [x] React + Vite + TypeScript 项目搭建, Tailwind CSS v4
- [x] `Layout.tsx` — 左侧 sidebar (聊天问答 + 文档上传) + 右侧内容区
- [x] `ChatPanel.tsx` — 统一对话界面 (QA/Quiz/Gap 三合一), 首屏能力卡片, quiz 意图自动弹窗, 初始化加载历史消息, assistant 支持 Markdown/LaTeX 渲染
- [x] `QuizModal.tsx` — Quiz MCQ 弹窗: 选项卡片 + 判分 + mastery 变化 + 完成总结
- [x] `FileUpload.tsx` — 多文件拖拽/点击上传, 已上传文档列表持久化显示, 文档删除
- [x] `api.ts` + `types.ts` — API 客户端 (sendChat/getChatHistory/uploadFiles/getDocuments/deleteDocument) + TypeScript 类型定义
- [x] `App.tsx` — 两面板切换 (chat / upload)

## Step 8: 评估 & 优化

评估方法：RAG 用 RAGAS 框架（业界标准），Agent 用任务驱动评估。Judge 模型默认走 Moonshot/Kimi（通过 `judge_api_key/judge_base_url/judge_model` 配置，当前默认 `kimi-k2.5` 非 thinking 模式），避免 Qwen 自评偏差。

### 8.1 评估数据集生成（半自动）
- [x] 数据集生成脚本 `evaluation/generate_dataset.py` — 从 Milvus chunks 用 LLM 生成 QA 候选
- [x] QA 评估集 `evaluation/datasets/qa_dataset.json` — 30 条（20 单跳 + 10 多跳）
- [x] Intent 评估集 `evaluation/datasets/intent_dataset.json` — 30 条（qa/quiz/gap 各 10）
- [x] 人工审核 & 定稿数据集
- [ ] Quiz 评估集 — 10 组生成题目（用 LLM-as-Judge 评分）— 待补充
- [ ] Gap 评估集 — 5-10 条预设 mastery 状态场景 — 待补充

### 8.2 RAG 评估（RAGAS 框架）— `evaluation/eval_rag.py`

运行方式：`cd backend && MILVUS_HOST=localhost python -m evaluation.eval_rag --profile {smoke|standard|full}`
结果输出：`evaluation/results/rag/rag_eval_YYYYMMDD_HHMMSS.json`（每次运行一份；时间戳为洛杉矶本地）
运行日志：`evaluation/results/logs_rag/eval_rag_YYYYMMDD_HHMMSS.log`（每次运行一份；`meta.run_log_file` 记录绝对路径）

已实现指标：
- [x] RAGAS 检索层 — Context Precision, Context Recall
- [x] RAGAS 生成层 — Faithfulness, Answer Relevancy
- [x] 经典 IR 指标 — Recall@K, MRR, Reranker Lift
- [x] Multi-hop Lift — 单跳 vs 多跳 Faithfulness 差值
- [x] RAGAS 评测稳健性修复 — Kimi judge 显式参数适配（`temperature=1`, `top_p=0.95`）
- [x] RAGAS 逐指标执行 + 错误显式记录 — 输出 `ragas_metric_status`，避免整批静默 `NaN`
- [x] 评测分档 — `smoke=4单跳+2多跳`、`standard=10单跳+5多跳`、`full=全量30`
- [x] 子集采样去偏 — QA 档位由”取前 N 条”改为按来源文档/来源组合的稳定 round-robin 分层采样
- [x] Kimi K2.5 Judge 兼容 — RAGAS judge 默认切换到 `kimi-k2.5` 非 thinking 路径，并按官方文档使用 `AsyncOpenAI` + `llm_factory` + `aevaluate()`；同时显式覆盖 Kimi 固定参数约束
- [x] RAGAS 公开 API 迁移 — metrics 改用 `Faithfulness` / `AnswerRelevancy` 等公开类名；`BaseRagasEmbeddings` 从 `ragas.embeddings` 导入；所有 RAGAS 导入延迟到函数内部
- [x] 消除隐式状态 — RAG 中间结果改用独立 `rag_results_map` dict，不再向 dataset item 注入 `_rag_result`
- [x] RAGAS 结构重构 — 新增 `evaluation/ragas_runner.py` 承接官方评测接入流，`eval_rag.py` 仅保留项目评测逻辑和结果编排
- [x] Phase 1 (Classic IR) 并行化 — 逐条 RAG pipeline + reranker lift 改为 `asyncio.gather` + `Semaphore(concurrency)` 并发执行；`compute_reranker_lift` 用 `asyncio.to_thread` 包装避免阻塞事件循环

### 8.3 Agent 评估（任务驱动）— `evaluation/eval_agent.py`

运行方式：`cd backend && MILVUS_HOST=localhost python -m evaluation.eval_agent --profile {smoke|standard|full}`
结果输出：`evaluation/results/agent/agent_eval_YYYYMMDD_HHMMSS.json`（每次运行一份；时间戳为洛杉矶本地）
运行日志：`evaluation/results/agentlog/eval_agent_YYYYMMDD_HHMMSS.log`（每次运行一份；`meta.run_log_file` 记录绝对路径）

已实现指标：
- [x] Intent Classification Acc — 直接评估真实 `plan_node` 单意图规划结果
- [x] QA E2E Task Success — 分层 QA 子集 + 独立会话 + LLM-as-Judge（当前默认 `kimi-k2.5` 非 thinking，pass/partial/fail/error）
- [x] Intent 子集去偏 — `smoke / standard / full` 改为按 `qa / quiz / gap_analysis` 均衡抽样，而非取数据集前缀
- [x] Kimi K2.5 Judge 兼容 — Agent judge 请求统一按 `kimi-k2.5` 非 thinking 口径构造，直连 Chat Completions 并显式带 `thinking={“type”:”disabled”}`
- [x] E2E Latency P50/P95 — 每条 QA 的端到端耗时
- [x] Judge Error 分离统计 — `judge_error_rate` / `pass_rate_evaluated`
- [x] 评测分档 — Intent: `smoke=6` / `standard=15` / `full=30`；QA: `smoke=3单跳+1多跳` / `standard=10单跳+5多跳` / `full=全量`
- [x] classify_intent 异常保护 — plan_node 失败时记录 error 并继续，不中断整个 Intent 评测
- [x] Orchestrator 120s 超时 — `asyncio.wait_for` 防止单条 QA 卡死整个评测
- [x] Latency 成功/全量分离统计 — `latency_p50/p95_ms` 仅含成功 case；`latency_p50/p95_ms_all` 含全部
- [x] Intent ‖ QA E2E 并行 — `main()` 中两个 phase 用 `asyncio.gather` 同时启动
- [x] QA E2E case 级并行 — `_run_one_qa()` + `asyncio.gather` + `Semaphore(concurrency)`，日志加 `[qa N/M]` 前缀
- [x] Intent case 级并行 — `_classify_one()` + `asyncio.gather` + `Semaphore(concurrency)`，日志加 `[intent N/M]` 前缀

新增指标：
- [x] Replan Efficiency（附加在 QA E2E 上）— `run_orchestrator` 暴露 `plan`/`replan_count`/`replan_action`，QA E2E 每条 case 收集 replan 统计，汇总 `replan_avg` / `replan_rate` / `replan_action_distribution`
- [x] Quiz Question Quality — 10 组 topic 场景 (`quiz_eval_dataset.json`)，调 `run_quiz_agent` 直接生成题目，规则校验结构 + LLM-as-Judge 二元 pass/fail → `quiz_valid_rate`
- [x] Gap Detection Precision — 6 组 mastery 预设场景 (`gap_eval_dataset.json`)，SQLite 写入预设 mastery → 调 `run_gap_agent` → LLM 提取报告中识别的 gap → 集合覆盖比较 → `coverage_rate` + `false_positive_rate`
- [x] ~~四路并行~~ — 已改为顺序执行（见下方稳定性修复）

稳定性修复：
- [x] 模型加载线程锁 — `embedding.py` / `reranker.py` 的 `_load_model()` 加 `threading.Lock` 双重检查，防止并发首次加载；新增 `warmup()` 方法
- [x] 评测 phase 顺序执行 — `eval_agent` 四个 phase 从 `asyncio.gather` 改为 intent → QA → quiz → gap 顺序执行，避免跨 phase GPU/SQLite 抢占（各 phase 内部仍保留 case 级并发）
- [x] SQLite WAL + busy_timeout — `database.py` 通过 SQLAlchemy connect 事件设置 `PRAGMA journal_mode=WAL` 和 `busy_timeout=5000`
- [x] 评测预检 (health check) — `eval_utils.py` 新增 `warmup_and_health_check()`：单线程预热 embedding/reranker、验证 Milvus 非空、检查 knowledge_concepts > 0；失败直接标记 run invalid 并中止
- [x] Partial save — `eval_agent` / `eval_rag` 每完成一个 phase 即写入 JSON（原子 rename），中途异常已完成 phase 结果不丢失；report 含 `meta.status` / `meta.completed_phases` / `meta.health_issues`

已分析后决定不实现：
- ~~Tool Selection Acc~~ — subagent dispatch 是确定性映射（plan intent → subagent），和 Intent Acc 重复，砍掉

### 8.4 辅助模块
- [x] `evaluation/judge_client.py` — Kimi K2.5 Judge 客户端 (OpenAI-compatible)
- [x] `evaluation/eval_utils.py` — 共享工具（logging 配置、P50/P95、时区常量）
- [x] `evaluation/__init__.py`
- [x] `requirements.txt` — 提升 `ragas` 依赖下限到 `0.4.3`，与当前官方文档/API 对齐
- [x] `evaluation/eval_all.py` — RAG + Agent 并行运行入口，子进程同时启动，输出加 `[RAG]`/`[AGENT]` 前缀
- [x] `evaluation/profiles.py` — `EvalProfile` 新增 `concurrency` 字段（smoke=3, standard=5, full=8）

### 8.4.1 评估运行方式汇总

```bash
# 单独运行 RAG 评估（Phase 1 IR 指标内部已并行）
cd backend && MILVUS_HOST=localhost python -m evaluation.eval_rag --profile {smoke|standard|full}

# 单独运行 Agent 评估（Intent ‖ QA E2E 并行，case 级并发）
cd backend && MILVUS_HOST=localhost python -m evaluation.eval_agent --profile {smoke|standard|full}

# RAG + Agent 全部并行（推荐）
cd backend && MILVUS_HOST=localhost python -m evaluation.eval_all --profile {smoke|standard|full}
```

并发度由 profile 控制：smoke=3, standard=5, full=8。API 限制（Kimi 50并发/200RPM, Qwen 30K RPM）远高于实际并发数，瓶颈在单条 orchestrator 内部 LLM 调用链。

### 8.5 测试结果与优化迭代

详见 [`EVAL_ITERATIONS.md`](EVAL_ITERATIONS.md) — 按迭代轮次记录测试结果、问题分析、改进方案。

- [x] 迭代 0 (Baseline): Smoke + Full 基准测试完成
- [x] 迭代 1: QA SubAgent 架构重构 + Prompt grounding 强化 → Faithfulness 0.819→0.934 (+11.5%)
- [x] 迭代 2: Embedding Heading Augmentation — embedding 时拼接 heading_path 提升检索精度; 已重新 ingest 并完成 full profile，对比见 `EVAL_ITERATIONS.md`

### 8.6 收尾
- [x] 迭代 2 测试结果对比迭代 1
- [x] 迭代 3: Reranker Off 公平 A/B — 已完成 full profile；关闭 reranker 时保持最终传给 LLM 的 chunk 数仍为 `top_k`，结果见 `EVAL_ITERATIONS.md`
- [x] 默认策略调整：`enable_reranker` 默认值改为关闭；保留环境变量开关以便后续手动启用对照
- [ ] Prompt 调优（基于评估结果）
- [ ] Docker 化部署
- [ ] README + 项目文档（含评估结果）
