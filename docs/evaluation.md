# 评估指标体系

## 概要

6 维评估：检索（Recall@K/MRR/Reranker Lift）、生成（Faithfulness/Relevance/Citation）、多跳（Decomposition/Synthesis/Lift）、Quiz（Eval Accuracy/Convergence/Coverage）、Gap（Precision/Recall）、端到端（Latency/Intent Acc）。

---

## 详细设计

### 检索模块 (RAG + Reranker)

| 指标 | 定义 | 计算方式 | 目标 |
|------|------|----------|------|
| **Recall@K** | Top-K检索结果中包含正确文档的比例 | 命中查询数 / 总查询数 | ≥ 0.85 |
| **MRR** (Mean Reciprocal Rank) | 正确文档在结果中排名的倒数均值 | Σ(1/rank_i) / N | ≥ 0.70 |
| **Reranker Lift** | Reranker相对纯向量检索的MRR提升 | MRR_reranked - MRR_raw | > 0 |
| **Retrieval Latency (P95)** | 95分位检索延迟 | 从query到返回Top-K的耗时 | < 500ms |

### 生成模块 (知识问答)

| 指标 | 定义 | 计算方式 | 目标 |
|------|------|----------|------|
| **Faithfulness** | 生成答案是否忠于检索到的源文档 | LLM-as-Judge: 每个claim是否有source支撑 | ≥ 0.90 |
| **Answer Relevance** | 答案是否切题回答了用户问题 | LLM-as-Judge: 相关性评分 [1-5] | ≥ 4.0 |
| **Citation Accuracy** | 引用的来源是否真实且匹配 | source_doc是否确实包含对应内容 | ≥ 0.85 |

### 多跳QA (Multi-hop)

| 指标 | 定义 | 计算方式 | 目标 |
|------|------|----------|------|
| **Decomposition Quality** | 子查询是否覆盖原始问题的各个方面 | LLM-as-Judge: 子查询完备性评分 | ≥ 0.80 |
| **Synthesis Coherence** | 综合答案是否逻辑连贯、无矛盾 | LLM-as-Judge: 连贯性评分 [1-5] | ≥ 4.0 |
| **Multi-hop Lift** | 多跳相对单跳在复杂问题上的答案质量提升 | 复杂问题集上 Faithfulness 差值 | > 0 |
| **Avg Sub-queries** | 平均子查询数 | 总子查询数 / 复杂问题数 | 2-4 |

### Quiz自适应复习模块

| 指标 | 定义 | 计算方式 | 目标 |
|------|------|----------|------|
| **Answer Eval Accuracy** | LLM判分与人工判分的一致率 | 抽样验证一致的比例 | ≥ 0.90 |
| **Mastery Convergence** | 掌握度是否随复习次数合理收敛 | 多次session后mastery变化曲线 | 单调趋升 |
| **Topic Coverage** | 每次quiz覆盖的topic多样性 | unique_topics / total_questions | ≥ 0.60 |
| **Adaptive Accuracy** | 加权选题是否有效 | corr(weight, selection_freq) | 正相关 |
| **Decay Effectiveness** | 时间衰减是否促进了间隔复习 | 长时间未复习topic被选中率 | 高于无衰减基线 |

### Knowledge Gap Analysis 模块

| 指标 | 定义 | 计算方式 | 目标 |
|------|------|----------|------|
| **Gap Detection Precision** | 识别出的盲区是否确实是用户薄弱点 | 人工验证正确数 / 总识别数 | ≥ 0.80 |
| **Gap Detection Recall** | 用户实际薄弱点中被识别出的比例 | 识别数 / 人工标注的实际薄弱点数 | ≥ 0.70 |
| **Recommendation Relevance** | 推荐的复习顺序是否合理 | LLM-as-Judge + 人工抽检 | ≥ 0.80 |

### 端到端指标

| 指标 | 定义 | 目标 |
|------|------|------|
| **E2E Latency (P95)** | 用户输入到收到首token的延迟 | < 2s (流式) |
| **Intent Classification Acc** | Orchestrator意图分类准确率 | ≥ 0.95 |
| **Session Completion Rate** | Quiz session正常完成的比例 | ≥ 0.95 |
| **Complexity Judge Acc** | QA SubAgent复杂度判断准确率 | ≥ 0.85 |
