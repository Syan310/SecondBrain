# Agent Tool 定义与分配

## 概要

5 个 Tool：rag_search（混合检索 + 可选 Rerank）、read/update_mastery（掌握度读写）、search_concepts（概念查询）、query_decompose（问题拆解）。Reranker 内嵌于 rag_search，LLM 推理/判断通过 prompt 实现不作为 Tool。

---

## 详细设计

### Tool 清单

| Tool | 功能 | 输入 | 输出 |
|------|------|------|------|
| `rag_search` | Dense+Sparse混合检索(RRF融合) + 可选 Rerank，返回相关文档片段 | query, top_k, expr(可选) | list[RankedChunk] (text, rerank_score, source_doc, heading_path, child_id, parent_id, chunk_index) |
| `read_mastery` | 读取用户掌握度（含时间衰减计算） | user_id, topics(可选) | topic + effective_mastery 列表 |
| `update_mastery` | 二元判分更新 mastery_score + last_reviewed | user_id, topic, is_correct | 更新后的 mastery 记录 |
| `search_concepts` | 查询知识库概念索引表 | scope/keyword(可选) | topic + heading_path 列表 |
### Agent × Tool 分配

| Agent | 可用 Tools | 说明 |
|-------|-----------|------|
| Orchestrator | 无 | 意图路由 + QA 子查询拆解（prompt 层面） + 结果汇总 |
| QA SubAgent | `rag_search` | ReAct 风格 (max 2轮)，LLM 通过 tool-calling 调用 rag_search |
| Quiz SubAgent | `read_mastery`, `update_mastery`, `rag_search`, `search_concepts` | 读写掌握度 + 检索出题素材 + 获取 topic 列表 |
| Gap Analysis SubAgent | `read_mastery`, `search_concepts` | 读取掌握度 + 知识库概念列表，交叉对比 |

### 设计说明

- **Reranker 不作为独立 Tool**：内嵌在 `rag_search` 中，通过 `enable_reranker` 配置开关控制；当前默认关闭。关闭时仍只返回 Milvus 原始排序的前 `top_k` 个 chunk，保证传给下游 LLM 的 chunk 数不变
- **LLM Generate / LLM Judge 不作为 Tool**：这是 Agent 自身的推理能力，通过 prompt 控制
- **query_decompose 已移除**：QA 子查询拆解由 Orchestrator plan_node 在规划阶段完成，QA SubAgent 只处理单概念查询
- **Gap Analysis 的 Cross Compare 不作为 Tool**：纯规则逻辑，写在 Agent prompt 或 node 代码中

### Gap Analysis Cross Compare 规则

| 情况 | 判定 | 优先级 |
|------|------|--------|
| knowledge_concepts 中有，但 user_mastery 中无记录 | **未接触盲区** | 高 |
| 有记录，effective_mastery < 0.3 | **薄弱盲区** | 高 |
| 有记录，effective_mastery 0.3~0.5 | **需巩固** | 中 |
| effective_mastery ≥ 0.5 | 掌握中/已掌握 | 低 |

Generate Report 时将结构化结果交给 LLM，生成自然语言诊断报告 + 推荐复习顺序。
