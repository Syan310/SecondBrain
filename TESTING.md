# 测试记录

## 运行方式

```bash
cd backend
MILVUS_HOST=localhost pytest tests/ -v
```

---

## 测试结果

### `test_milvus_client.py` — MilvusService 集成测试（需 Milvus 运行）

✅ 全部通过（insert / count / search / hybrid_search / filter / delete）

### `test_rag_search.py` — rag_search Tool 单元测试（纯 mock）

✅ 全部通过（基本链路 / 空结果 / 候选不足 / metadata 映射 / filter 透传）

---

### `test_orchestrator.py` — Orchestrator Plan & Execute with Replan（纯 mock）

✅ 39 个测试全部通过

- `TestParseJson` (4): JSON 解析
- `TestPlanNode` (7): 意图识别（单 QA / Gap / 多意图 / Quiz in steps / 纯 Quiz / fallback / 无效 intent 过滤）
- `TestExecuteStep` (4): 单步执行（QA / Quiz dispatch / 依赖参数改写 / 越界 noop）
- `TestReplanNode` (8): Replan 决策（LLM finish / max count / continue / revise 替换 plan / finish / 无效 action / 解析失败 / 无效 revised_steps fallback）
- `TestSynthesizeNode` (1): 多结果汇总
- `TestPassthroughNode` (2): 单结果直传 / 空结果
- `TestRouting` (7): 路由函数（plan / replan / output 三层路由）
- `TestGraphIntegration` (6): 端到端（单 QA / 纯 Quiz / 多意图 gap+qa / gap+quiz / replan-revise / replan-finish-early）

---

## 测试中发现的问题与修复

| 问题 | 原因 | 修复 |
|------|------|------|
| `hybrid_search()` TypeError | pymilvus 2.6 中 `rerank` 是位置参数，不能用 `ranker=` | 改为按位置传参 |
| `count()` 删除后不归零 | `num_entities` 是缓存值，软删除不更新 | 改用 `query(output_fields=["count(*)"])` |
