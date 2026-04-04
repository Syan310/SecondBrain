# 测试记录

## Step 5: Orchestrator (2026-04-04)

### 测试文件
`tests/test_orchestrator.py` — 24 个测试，全部通过

### 测试覆盖

| 模块 | 测试数 | 覆盖内容 |
|------|--------|---------|
| `_parse_json` | 4 | 纯 JSON / markdown fence / 无效输入 |
| `plan_node` | 6 | 单 QA / 单 Gap / 多意图串行 / quiz-only / 解析失败 fallback / 非法意图过滤 |
| `execute_node` | 4 | 单 QA / 单 Gap / 串行 Gap→QA(含参数改写) / 空 plan |
| `synthesize_node` | 1 | 多结果合并 |
| `passthrough_node` | 2 | 单结果直传 / 空结果 |
| Routing | 4 | route_after_plan / route_after_execute 各2种路径 |
| Graph 集成 | 3 | 单 QA 全流程 / quiz-only 全流程 / 多意图(Gap→QA)全流程 |

### 遇到的问题

1. **Python 3.9 兼容性**：本地系统 Python 3.9 不支持 `X | None` 语法。解决：为所有相关文件添加 `from __future__ import annotations`，State 中 `Optional[dict]` 兼容 LangGraph `get_type_hints()`。

2. **重依赖导入链**：orchestrator → subagents → tools → services (FlagEmbedding/pymilvus/torch)。解决：测试中 mock 第三方重依赖模块。

3. **pytest-asyncio 版本**：需要安装 `pytest-asyncio` 并使用 `@pytest.mark.asyncio` 标记异步测试。
