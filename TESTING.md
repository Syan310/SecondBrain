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

## 测试中发现的问题与修复

| 问题 | 原因 | 修复 |
|------|------|------|
| `hybrid_search()` TypeError | pymilvus 2.6 中 `rerank` 是位置参数，不能用 `ranker=` | 改为按位置传参 |
| `count()` 删除后不归零 | `num_entities` 是缓存值，软删除不更新 | 改用 `query(output_fields=["count(*)"])` |
