# 评估体系

## 概要

两条评估线：**RAG 评估**（RAGAS 框架，业界标准）+ **Agent 评估**（任务驱动）。Judge 模型使用 Kimi-K2.5，避免 Qwen 自评偏差。

---

## 评估架构

```
评估数据集 (半自动生成, 人工审核)
    │
    ├── RAG 评估 (RAGAS 框架)
    │   ├── 检索层: Context Precision, Context Recall, Recall@K, MRR, Reranker Lift
    │   └── 生成层: Faithfulness, Answer Relevancy
    │
    ├── Agent 评估 (任务驱动)
    │   ├── 路由层: Intent Classification Acc, Tool Selection Acc
    │   ├── 子任务: Task Success Rate, Quiz Quality, Gap Precision
    │   └── 多跳: Multi-hop Lift
    │
    └── 端到端: E2E Latency (P95), Overall Success Rate
```

---

## 一、RAG 评估（RAGAS 框架）

### 检索层

| 指标 | 定义 | 计算方式 | 目标 |
|------|------|----------|------|
| **Context Precision** | 检索到的 chunks 中相关比例 | RAGAS 自动计算 | ≥ 0.80 |
| **Context Recall** | ground truth 所需信息被检索到的比例 | RAGAS 自动计算 | ≥ 0.80 |
| **Recall@K** | Top-K 命中率 | 命中查询数 / 总查询数 | ≥ 0.85 |
| **MRR** | 正确文档排名倒数均值 | Σ(1/rank_i) / N | ≥ 0.70 |
| **Reranker Lift** | Reranker 前后 MRR 差值 | MRR_reranked - MRR_raw | > 0 |

### 生成层

| 指标 | 定义 | 计算方式 | 目标 |
|------|------|----------|------|
| **Faithfulness** | 答案中每个 claim 是否有 context 支撑 | RAGAS (Kimi-K2.5 as Judge) | ≥ 0.85 |
| **Answer Relevancy** | 答案是否切题 | RAGAS (Kimi-K2.5 as Judge) | ≥ 0.80 |

---

## 二、Agent 评估（任务驱动）

### 路由与决策

| 指标 | 定义 | 评估方式 | 目标 |
|------|------|----------|------|
| **Intent Classification Acc** | Orchestrator 意图分类准确率 | 标注 test set 对比 | ≥ 0.95 |
| **Tool Selection Acc** | Agent 工具选择准确率 | 记录 trajectory, LLM-as-Judge | ≥ 0.85 |
| **Replan Efficiency** | 平均 replan 次数 | 统计 | ≤ 1.5 |

### 子任务质量

| 指标 | 适用 | 评估方式 | 目标 |
|------|------|----------|------|
| **Task Success Rate** | QA/Quiz/Gap | LLM-as-Judge 判通过/不通过 | ≥ 0.80 |
| **Multi-hop Lift** | QA 复杂问题 | decompose 开启/关闭 Faithfulness 差值 | > 0 |
| **Quiz Question Quality** | Quiz | LLM-as-Judge 评分 (相关性+区分度+正确性) | ≥ 4.0/5 |
| **Gap Detection Precision** | Gap | 人工抽检盲区合理性 | ≥ 0.80 |

---

## 三、端到端

| 指标 | 定义 | 目标 |
|------|------|------|
| **E2E Latency (P95)** | 输入到首 token 延迟 | < 2s (流式) |
| **Overall Success Rate** | 全链路任务完成率 | ≥ 0.85 |

---

## 四、评估数据集

| 数据集 | 用途 | 条数 | 格式 |
|--------|------|------|------|
| QA 评估集 | RAG + 生成 | 30-40 (含 10 多跳) | `{question, ground_truth, contexts, source_doc}` |
| Intent 评估集 | 路由准确率 | 30 (qa/quiz/gap 各 10) | `{query, expected_intent}` |
| Quiz 评估集 | 出题质量 | 10 组 | 生成后 LLM-as-Judge 评分 |
| Gap 评估集 | 盲区检测 | 5-10 | 预设 mastery + 验证报告 |

**生成方式**：半自动（LLM 从 Milvus chunks 生成候选 → 人工审核定稿）。

---

## 五、技术实现

- **RAGAS 框架**: `pip install ragas`, 传入 `(question, answer, contexts, ground_truth)` 四元组
- **Judge 模型**: Kimi-K2.5 (Moonshot API, OpenAI-compatible)
- **评估脚本**: `backend/evaluation/` 目录, 每个维度独立模块, 统一 runner 输出 JSON 报告
