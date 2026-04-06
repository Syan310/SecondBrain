# 测试记录

## Step 5: Orchestrator + Replan Loop (2026-04-04)

### 测试文件
`tests/test_orchestrator.py` — 33 个测试，全部通过

### 测试覆盖

| 模块 | 测试数 | 覆盖内容 |
|------|--------|---------|
| `_parse_json` | 4 | 纯 JSON / markdown fence / 无效输入 |
| `plan_node` | 7 | 单 QA / 单 Gap / 多意图串行 / quiz 最后一步 / 解析失败 fallback / 非法意图过滤 / memory_context 注入验证 |
| `execute_step` | 4 | 单 QA / 单 Gap / 依赖步骤参数改写 / 超出 plan 范围 noop |
| `replan_node` | 5 | continue / finish / revise(替换剩余计划) / max replan 强制 finish / 解析失败默认 continue |
| `synthesize_node` | 1 | 多结果合并 |
| `passthrough_node` | 2 | 单结果直传 / 空结果 |
| Routing | 7 | route_after_plan(2) / route_after_replan(3) / route_to_output(2) |
| Graph 集成 | 3 | 单 QA 全流程(含 replan) / quiz-only 全流程 / 多意图(Gap→QA 含 rewrite+replan)全流程 |

### 遇到的问题

1. **Python 3.9 兼容性**：本地系统 Python 3.9 不支持 `X | None` 语法。解决：为所有相关文件添加 `from __future__ import annotations`，State 中 `Optional[dict]` 兼容 LangGraph `get_type_hints()`。

2. **重依赖导入链**：orchestrator → subagents → tools → services (FlagEmbedding/pymilvus/torch)。解决：测试中 mock 第三方重依赖模块。

3. **pytest-asyncio 版本**：需要安装 `pytest-asyncio` 并使用 `@pytest.mark.asyncio` 标记异步测试。

4. **架构升级 (Replan Loop)**：从旧版 execute_node(一次执行全部) 升级为 execute_step(逐步执行) + replan_node(LLM 审查) 循环。测试需要模拟多轮 LLM 调用 (plan → replan → rewrite → replan → synthesize)。

## Step 5.5: Agent 记忆机制 (2026-04-04)

### 测试文件
- `tests/test_memory_crud.py` — 15 个测试，全部通过
- `tests/test_memory_manager.py` — 13 个测试，全部通过

### 测试覆盖

| 模块 | 测试数 | 覆盖内容 |
|------|--------|---------|
| CRUD: add/count | 3 | 添加单轮/多轮、未知用户 count=0 |
| CRUD: get_next_round | 2 | 首轮=1、已有3轮后=4 |
| CRUD: get_recent | 5 | 全量获取/limit截断/升序/用户隔离/内容保留 |
| CRUD: get_oldest | 2 | 取最旧N轮/不足时取全部 |
| CRUD: delete | 3 | 删除指定轮次/空列表/删除后剩余验证 |
| load_memory_context | 6 | 空历史/含短期/含长期/相似度过滤/两者兼有/长期失败降级 |
| save_conversation | 2 | 写入短期/轮次递增 |
| consolidation | 5 | 超限触发/未超限不触发/LLM失败保留数据/Milvus失败保留数据/正确删除最旧轮次 |

### 遇到的问题

1. **SQLAlchemy `Real` 类型**：本地 SQLAlchemy 2.0.49 不导出 `Real`，改用 `Float`。

2. **跨测试文件 sys.modules 污染**：`test_orchestrator.py` 在模块级替换 `app.db.*` 等为 MagicMock，导致后续测试文件导入到 mock 对象。
   - 解决：`test_memory_manager.py` 使用 session-scoped fixture `_restore_real_modules` 在测试执行时恢复真实模块 + `importlib.reload()`。
   - 改用 `monkeypatch` fixture 替代 `unittest.mock.patch`，直接设置模块属性，避免 `sys.modules` 查找问题。
   - `test_orchestrator.py` 的 graph 集成测试增加 `patch("app.agent.graph.load_memory_context")` 和 `save_conversation` mock。

3. **greenlet 依赖**：SQLAlchemy async 引擎需要 `greenlet` 包，本地未预装。`pip3 install greenlet` 解决。

## Memory Manager 修复回归 (2026-04-04)

### 变更
- 修复 `memory_manager.py` 中对 `EmbeddingService.encode()` 返回值的错误访问。
- 从字典式 `encoded["dense"] / encoded["sparse"]` 改为真实 `EmbeddingOutput` 属性访问 `encoded.dense / encoded.sparse`。
- 同步修正 `tests/test_memory_manager.py`，测试假数据改为真实 `EmbeddingOutput` 对象，避免接口失配被 MagicMock 掩盖。

### 执行的测试
- `pytest -q tests/test_memory_manager.py`

### 结果
- 未能完成用例执行，环境报错：`ModuleNotFoundError: No module named 'aiosqlite'`
- 已确认 `backend/requirements.txt` 声明了 `aiosqlite>=0.20.0`，但当前 Python 环境缺少可用 SSL 模块，`pip` 无法从 PyPI 下载依赖，因此无法在本轮完成端到端验证。

### 遇到的问题

1. **测试环境缺少 `aiosqlite`**：memory_manager 测试依赖 async SQLite 引擎，当前环境未安装该包。
2. **`pip` 无法补装依赖**：执行 `python -m pip install 'aiosqlite>=0.20.0'` 时，由于 Python 缺少 SSL 支持，无法连接 `https://pypi.org` 下载包。

## Stash 合并冲突处理：`memory_manager` dense 向量 (2026-04-04)

### 决策
- `MemoryMilvusService.hybrid_search` / `insert` 的 `query_vector` / `embedding` 类型为 `list[float]`（与 `rag_search` 中 `dense.tolist()` 一致）。
- 生产代码：`dense = encoded.dense[0].tolist()`。
- 测试：`_fake_encode_result()` 返回真实 `EmbeddingOutput`，`dense` 为 `np.ndarray`，以便 `.tolist()` 与生产路径一致。

### 执行的测试
- `PYTHONPATH=. python3 -m pytest tests/test_memory_manager.py -q`

### 结果
- 13 passed。

## 全面可用性测试 (2026-04-05)

### 单元测试 (Docker 内执行, 81/81 passed)

| 测试文件 | 测试数 | 状态 |
|---------|-------|------|
| test_memory_crud.py | 15 | ALL PASSED |
| test_memory_manager.py | 13 | ALL PASSED |
| test_milvus_client.py | 8 | ALL PASSED |
| test_orchestrator.py | 33 (含 graph integration) | ALL PASSED |
| test_rag_search.py | 5 | ALL PASSED |

### 修复的测试问题

1. **sys.modules mock 泄漏** — `test_memory_manager.py` 和 `test_rag_search.py` 在模块级别无条件注入 `sys.modules["pymilvus"] = MagicMock()`，导致后续 `test_milvus_client.py` 拿到 mock 而非真实 pymilvus。修复：改为 `try: __import__(mod)` 先尝试导入，仅在 `ImportError` 时注入 mock。

2. **_COLLECTION 模块常量未被测试 fixture 覆盖** — `milvus_client.py` 的 `_COLLECTION = settings.milvus_collection_name` 在 import 时捕获，fixture 修改 `settings` 不影响已捕获的常量，导致测试操作生产 collection。修复：fixture 中同时 patch `mc._COLLECTION = test_collection`。

### API 端点测试 (全部 200 OK)

| 端点 | 方法 | 状态 |
|------|------|------|
| /health | GET | 200 OK |
| /api/chat/ (QA) | POST | 200 OK — RAG 检索+生成正常 |
| /api/chat/ (Quiz) | POST | 200 OK — 返回 quiz JSON, 含 2 道选择题 |
| /api/chat/ (Gap) | POST | 200 OK — 返回知识盲区诊断报告 |
| /api/chat/history | GET | 200 OK — 返回历史消息 |
| /api/quiz/answer | POST | 200 OK — 判分+mastery 更新正常 |
| /api/ingest/documents | GET | 200 OK — 4 篇文档, 概念数正确 |

### 前端测试

- TypeScript 编译: 零错误
- Vite 生产构建: 成功 (106ms)
- 聊天面板: 历史加载正常, 新消息发送+接收正常
- 文档上传面板: 文档列表加载正常 (4 篇, 含概念数和删除按钮)
- 面板切换: 正常
- 能力卡片点击: 触发对话正常
- Vite proxy → Backend: 所有 API 调用通过代理正常转发

## Step 8: `eval_agent` LLM-as-Judge (2026-04-05)

### 变更
- Moonshot `kimi-k2.5` 判分接口拒绝 `temperature=0`，返回 `400 invalid temperature: only 1 is allowed for this model`。
- `evaluation/judge_client.py` 默认 `temperature=1.0`；`eval_agent.judge_task_success` 不再传入 `0.0`。
- Phase 2 单条判题若抛错（如 429 `engine_overloaded`），记为 `verdict: error`、`judge_call_failed: true`，写入 `results` 明细并继续后续条目，整轮评估不再中断。
- 每次运行生成 `evaluation/results/agentlog/eval_agent_YYYYMMDD_HHMMSS.log`（根 logger，含第三方 INFO）；日志与 JSON 文件名中的时间戳均为 **洛杉矶**（`America/Los_Angeles`）。JSON 报告 `meta.run_log_file` / `meta.output_file` / `meta.started_at` / `meta.finished_at`（ISO，带 LA 偏移）便于对照排查。

### 验证
- 需在本地具备 Milvus、嵌入/重排模型与 API 密钥后运行：`MILVUS_HOST=localhost python -m evaluation.eval_agent`。

## Step 8: 评测脚本口径修复 (2026-04-06)

### 变更
- `evaluation/eval_rag.py`
  - RAGAS judge 显式适配 Moonshot `kimi-k2.5`：`temperature=1`、`top_p=0.95`
  - RAGAS 改为逐指标执行，输出 `ragas_metric_status`，单指标失败不再让整批结果静默变成 `NaN`
  - Multi-hop lift 改为仅基于有效 `faithfulness` 值计算，并按 `question_type` 区分单跳/多跳
  - 结果仅单写：`evaluation/results/rag/rag_eval_YYYYMMDD_HHMMSS.json`
  - 运行日志改为按次落盘：`evaluation/results/logs_rag/eval_rag_YYYYMMDD_HHMMSS.log`，并在 `meta.run_log_file` 记录绝对路径
- `evaluation/eval_agent.py`
  - Intent 评测改为直接调用真实 `plan_node`
  - QA E2E 样本改为独立 `user_id/session_id`，避免记忆串题
  - QA E2E 子集改为 single-hop / multi-hop 分层抽样，不再直接取前 15 条
  - judge 错误与真实 `pass/partial/fail` 分开统计，输出 `pass_rate_all` / `pass_rate_evaluated` / `judge_error_rate`
  - 结果仅单写：`evaluation/results/agent/agent_eval_YYYYMMDD_HHMMSS.json`
  - 运行日志目录改为：`evaluation/results/agentlog/eval_agent_YYYYMMDD_HHMMSS.log`

### 执行的检查
- `PYTHONPATH=/Users/sy/Desktop/SecondBrian/backend /opt/anaconda3/bin/python -m py_compile evaluation/eval_rag.py evaluation/eval_agent.py`
- IDE lints: `evaluation/eval_rag.py`, `evaluation/eval_agent.py`

### 结果
- `py_compile` 通过
- 两个评测脚本无 IDE linter 报错
- 本轮**未运行**完整 `eval_rag` / `eval_agent`，因此尚未生成新的评测结果

## Step 8: 评测日志目录整理 (2026-04-06)

### 变更
- `evaluation/eval_rag.py` 新增按次文件日志，输出到 `evaluation/results/logs_rag/eval_rag_YYYYMMDD_HHMMSS.log`，并在结果 JSON 的 `meta.run_log_file` 记录绝对路径。
- `evaluation/eval_agent.py` 的日志目录由 `evaluation/results/logs/` 调整为 `evaluation/results/agentlog/`。
- 已有 agent 结果 JSON 中的 `meta.run_log_file` 同步改为 `agentlog` 路径，避免历史报告与实际日志目录不一致。

### 执行的检查
- `cd backend && /opt/anaconda3/bin/python -m py_compile /Users/sy/Desktop/SecondBrian/backend/evaluation/eval_rag.py /Users/sy/Desktop/SecondBrian/backend/evaluation/eval_agent.py`
- `cd backend && PYTHONPATH=/Users/sy/Desktop/SecondBrian/backend /opt/anaconda3/bin/python -m evaluation.eval_rag --help`
- `cd backend && PYTHONPATH=/Users/sy/Desktop/SecondBrian/backend /opt/anaconda3/bin/python -m evaluation.eval_agent --help`

### 结果
- `py_compile` 通过
- 两个评测脚本 CLI 帮助输出正常
- 本轮未跑完整评测，因此还没有新生成的 `logs_rag` 实例日志文件

### 遇到的问题
1. `ragas` 通过 `llm_factory()` 调 Kimi 时默认采样参数与 Moonshot 约束不兼容，会导致 `400 invalid temperature` / `400 invalid top_p`，此前在 `raise_exceptions=False` 下表现为整列空值。
2. `eval_agent.py` 原实现复用了同一 `user_id/session_id`，会让记忆跨题污染 E2E 成功率。

## Step 8: 评测分档 (2026-04-05)

### 变更
- 新增 `evaluation/profiles.py`，统一维护 `smoke / standard / full` 三档评测预设。
- `evaluation/eval_agent.py`
  - 新增 `--profile {smoke,standard,full}` 参数，默认 `standard`
  - Intent 评测支持分档：`smoke=6`、`standard=15`、`full=全量30`
  - QA E2E 支持分档：`smoke=3单跳+1多跳`、`standard=10单跳+5多跳`、`full=全量`
  - 结果 `meta` 中新增 `profile`，Intent 结果中新增 `subset`
- `evaluation/eval_rag.py`
  - 新增 `--profile {smoke,standard,full}` 参数，默认 `standard`
  - RAG 评测支持分档：`smoke=4单跳+2多跳`、`standard=10单跳+5多跳`、`full=全量30`
  - 输出新增 `subset` 元信息，便于回看当次口径
- 新增 `tests/test_eval_profiles.py`，验证三档抽样数量与默认 profile 归一化逻辑

### 执行的测试
- `python -m py_compile backend/evaluation/eval_agent.py backend/evaluation/eval_rag.py backend/evaluation/profiles.py backend/tests/test_eval_profiles.py`
- `PYTHONPATH=backend pytest -q backend/tests/test_eval_profiles.py`

### 结果
- `py_compile` 通过
- `4 passed in 0.01s`

### 遇到的问题
1. `eval_rag.py` 在接入 CLI profile 参数时引入了一处重复 `main()` 定义，首次 `py_compile` 报 `IndentationError`；已删除残留定义并重新验证通过。

## Step 8: 首轮评测结果 & 问题修复 (2026-04-06)

### 首轮评测结果 (Baseline — 修复前)

运行 profile: `smoke`，两组评测同期启动。

#### RAG 评测

| 指标类别 | 指标 | 值 | 备注 |
|---------|------|---|------|
| **IR 检索** | Recall@K | 1.0 (6/6) | 全部命中 |
| | MRR (raw) | 1.0 | |
| | MRR (reranked) | 1.0 | |
| | Reranker Lift | 0.0 | raw 已完美，无提升空间 |
| | Latency P50 | 17.7s | |
| | Latency P95 | 18.2s | |
| **RAGAS** | Answer Relevancy | 0.736 (均值) | 唯一成功的指标 |
| | Faithfulness | **ERROR** | `max_tokens` 截断 |
| | Context Precision | **ERROR** | `max_tokens` 截断 |
| | Context Recall | **ERROR** | `max_tokens` 截断 |

Answer Relevancy 逐题明细：

| 样本 | 类型 | Relevancy | 备注 |
|------|------|-----------|------|
| Q1: Transformer 位置编码 | single-hop | 0.974 | |
| Q2: Multi-Head Attention 计算量 | single-hop | 0.974 | |
| Q3: Pre-Norm vs Post-Norm | single-hop | 0.812 | |
| Q4: Decoder Mask 位置 | single-hop | 0.782 | |
| Q5: 递归切分 + IVF nprobe | multi-hop | **0.000** | 模型回答"无法回答" |
| Q6: Agent Tool Definition 变更 | multi-hop | 0.872 | |

#### Agent 评测

| 指标类别 | 指标 | 值 | 备注 |
|---------|------|---|------|
| **Intent Classification** | Accuracy | 1.0 (6/6) | 全部正确分类为 qa |
| **QA E2E** | Pass Rate | 0% | 无法评估（judge 全部失败） |
| | Judge Error Rate | **100%** (4/4) | 所有判分失败 |
| | Orchestrator Error | 0 | 流水线本身正常 |
| | Latency P50 | **398s** (~6.6 min) | |
| | Latency P95 | **1425s** (~23.7 min) | |

### 问题分析与修复

#### P0-1: RAGAS 3 个指标因 `max_tokens` 截断全部失败

**根因**: RAGAS 内部通过 `llm_factory` 调用 Kimi-K2.5，默认 `InstructorModelArgs.max_tokens=1024`。Kimi 的 reasoning 内容较长，1024 token 不够，回复被截断导致 JSON 结构不完整，RAGAS 解析失败抛出 `InstructorRetryException`。

**修复**: `eval_rag.py` 中 `llm_factory()` 调用添加 `max_tokens=4096` 参数，覆盖默认值。

```python
# eval_rag.py L216-221
judge_llm = llm_factory(
    settings.judge_model,
    client=client,
    temperature=KIMI_RAGAS_TEMPERATURE,
    top_p=KIMI_RAGAS_TOP_P,
    max_tokens=4096,  # ← 新增，解决截断问题
)
```

#### P0-2: Agent Judge `max_tokens=256` 导致 100% 判分失败

**根因**: `eval_agent.py` 中 `judge_task_success()` 调用 Kimi 时 `max_tokens=256`，Kimi 回复被截断为不完整 JSON（如 `{"verdict": "`），解析失败。实际 4 个 answer 内容质量都很高，问题完全在评估管线。

**修复**: `eval_agent.py` 中 `max_tokens` 从 256 提高到 1024。

```python
# eval_agent.py L196
max_tokens=1024,  # 原值 256
```

#### P1: Multi-hop Q5 Answer Relevancy = 0

**根因**: `eval_rag.py` 的 `QA_SYSTEM_PROMPT` 指示"如果文档中没有相关信息，请说明无法回答"。Q5 是跨文档推理题（递归切分+重叠窗口 ↔ IVF nprobe），检索到了两个相关片段但没有显式关联，模型遂拒绝回答。

**修复**: 修改 prompt，允许跨片段推理，仅在完全无关时拒绝。

```python
# eval_rag.py QA_SYSTEM_PROMPT
# 旧: "如果文档中没有相关信息，请说明无法回答。"
# 新: "你可以综合多个片段的信息进行推理和关联分析，不必局限于单个片段中的显式表述。
#      如果文档片段与问题完全无关，请说明无法回答。"
```

#### P2: Agent E2E 延迟过高 (P50 ~7min)

**根因**: 一次 QA E2E 串行经过 memory → plan → qa_agent(plan → ReAct → synthesize → QC) → replan → output，约 6-8 次 LLM 调用。生成内容过长（答案 800-1500 字，含大量 markdown 表格和延伸讨论）是主要耗时来源。

**修复**: 在 prompt 中添加篇幅约束 + 配套下调 `max_tokens`。

| 位置 | 篇幅约束 | max_tokens 变更 |
|------|---------|----------------|
| `qa_agent` Executor prompt | ≤150 字：只保留核心结论和关键依据 | 不变 (2048) |
| `qa_agent` Synthesizer prompt | ≤300 字：保留核心结论、关键机制、必要公式 | 4096 → 2048 |
| `orchestrator` Synthesizer prompt | ≤400 字：保留核心结论和关键信息 | 4096 → 2048 |
| `qa_agent` fallback (空知识库) | — | 4096 → 2048 |
| `graph.py` stream synthesize | — | 4096 → 2048 |

**预期效果**: 答案长度从 800-1500 字降至 300-400 字，生成 token 数减少 60-70%，E2E 延迟预期从 P50 ~7min 降至 ~2-3min。

### 待验证

以上四项修复均需重新运行 `smoke` profile 评测以验证：
- [ ] RAGAS faithfulness / context_precision / context_recall 正常出分
- [ ] Agent judge 正常返回 pass/partial/fail 判定
- [ ] Q5 multi-hop 不再返回"无法回答"，answer_relevancy > 0
- [ ] Agent E2E latency P50 显著下降
