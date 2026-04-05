# Agent 记忆机制

## 概要

双层记忆：短期（SQLite，最近20轮对话全量注入 prompt）+ 长期（Milvus，LLM摘要→向量化，按需检索 Top-3）。超限时自动 consolidate：取最旧10轮→LLM摘要→写入长期→删除旧轮次。

---

## 详细设计

### 架构

```
用户请求 → memory_node (读取记忆) → plan_node (含记忆上下文) → execute → respond
                                                                          │
                                                                          ▼
                                                              save_conversation (写入短期)
                                                                          │
                                                              短期 > 20 轮? ─→ consolidate
                                                                                  │
                                                              LLM 摘要 → Milvus 长期记忆
                                                              删除已整合的旧轮次
```

### 短期记忆 (SQLite)

- 存储: `conversation_history` 表, 按 `user_id` 隔离
- 容量: 最近 20 轮 (可配置 `memory_short_term_limit`)
- 内容: user 原文 + assistant 最终回答 (不含中间推理)
- 注入: 每次请求全量注入 Orchestrator system prompt

### 长期记忆 (Milvus)

- 存储: `conversation_memory` Collection, dense + sparse 向量
- 内容: LLM 结构化摘要 (100-200字自然语言 + topics 列表)
- 检索: 按需检索 Top-3, 相似度阈值过滤, 按 user_id 隔离
- 写入: 短期 > 20 轮时, 取最早 10 轮 → LLM 摘要 → 向量化 → 写入
- 实现状态: `memory_manager` 直接消费 `EmbeddingOutput.dense` / `EmbeddingOutput.sparse`，与 embedding service 返回契约保持一致

### 作用范围

- **Orchestrator**: 注入点, memory_node 读取, plan_node 使用
- **QA / Gap SubAgent**: 透传 (从 state.memory_context 获取)
- **Quiz SubAgent**: 不使用 (已有 mastery 机制)

### 配置项

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `memory_short_term_limit` | 20 | 短期记忆最大轮次 |
| `memory_consolidate_count` | 10 | 触发整合时压缩的轮次数 |
| `memory_long_term_top_k` | 3 | 长期记忆检索数量 |
| `memory_similarity_threshold` | 0.3 | 长期记忆最低相似度 |
