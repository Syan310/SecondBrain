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
