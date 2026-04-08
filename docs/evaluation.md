# 评估体系

## 概要

两条评估线：**RAG 评估**（RAGAS 框架，业界标准）+ **Agent 评估**（任务驱动）。Judge 模型使用 Kimi-K2.5，避免 Qwen 自评偏差。

当前实现约束：

- RAGAS 调 Kimi 时显式固定 `temperature=1`、`top_p=0.95`，避免 Moonshot 默认参数不兼容导致整列指标静默变成空值。
- RAGAS 四个指标按 **逐指标** 执行并分别记录状态；单个指标失败不会吞掉错误，也不会让其他指标一起失效。
- Agent 的 QA E2E 评测对每条样本使用独立 `user_id/session_id`，避免记忆串题污染结果。
- Agent 的 E2E 默认使用**分层子集**（single-hop + multi-hop），而不是简单取数据集前缀。
- RAG / Agent 评测脚本统一支持 `smoke / standard / full` 三档运行口径，用同一份数据集文件按不同规模抽样。
- RAG / Agent 评测结果文件名与时间字段使用 **洛杉矶时间**（`America/Los_Angeles`，ISO 含夏令时偏移）；每轮仅落盘一份带时间戳的 JSON。

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
    │   ├── 路由层: Intent Classification Acc
    │   ├── 子任务: Task Success Rate (含 Replan 诊断), Quiz Valid Rate, Gap Coverage
    │
    └── 端到端: E2E Latency (P95), Overall Success Rate
```

---

## 一、RAG 评估（RAGAS 框架）

### 运行档位

RAG 评估仅使用 **单跳** 样本（multi-hop 由 Agent 评估覆盖）。

| Profile | 用途 | 样本规模 |
|---------|------|----------|
| `smoke` | 开发期快速回归 | 4 单跳 |
| `standard` | 日常主评测 | 10 单跳 |
| `full` | 阶段性全量验收 | 全量单跳 |

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

### 运行档位

| Profile | Intent | QA E2E | Quiz | Gap | 用途 |
|---------|--------|--------|------|-----|------|
| `smoke` | 6 | 3 单跳 + 1 多跳 | 3 | 2 | 开发期快速回归 |
| `standard` | 15 | 10 单跳 + 5 多跳 | 6 | 4 | 日常主评测 |
| `full` | 全量 | 全量 | 全量 | 全量 | 阶段性全量验收 |

四个 phase 使用 `asyncio.gather` 四路并行，各自独立 `Semaphore(concurrency)` 控制并发。

### 路由与决策

| 指标 | 定义 | 评估方式 | 目标 |
|------|------|----------|------|
| **Intent Classification Acc** | Orchestrator `plan_node` 单意图规划准确率 | 标注 test set 对比 | ≥ 0.95 |

> **已分析后不实现**：Tool Selection Acc（subagent dispatch 是 intent→subagent 的确定性映射，与 Intent Acc 完全重复）。

### 子任务质量

| 指标 | 适用 | 评估方式 | 目标 |
|------|------|----------|------|
| **Task Success Rate** | QA 子集 E2E | LLM-as-Judge 判 pass/partial/fail | ≥ 0.80 |
| **Replan Efficiency** | QA E2E 附带诊断 | piggyback 统计 `replan_avg` / `replan_rate` / `replan_action_distribution`，仅在 action=revise 时计数 | 诊断信号，无硬目标 |
| **Quiz Valid Rate** | Quiz | 规则校验结构（4 选项 / A-D 答案）+ Kimi Judge 二元 pass/fail（题目可从 source_chunks 推导 + 答案正确） | ≥ 0.80 |
| **Gap Coverage** | Gap | SQLite 预设 mastery → 调 `run_gap_agent` → LLM 从报告提取 gap 列表 → 集合比较得 `coverage_rate` + `false_positive_rate` | coverage ≥ 0.80, FP ≤ 0.20 |

---

## 三、端到端

| 指标 | 定义 | 目标 |
|------|------|------|
| **E2E Latency (P95)** | 输入到首 token 延迟 | < 2s (流式) |
| **Overall Success Rate** | 全链路任务完成率 | ≥ 0.85 |

说明：

- `eval_agent.py` 当前输出的是 **QA 子集 E2E 成功率**，结果字段名为 `qa_e2e_task_success`，不是 quiz/gap 的综合成功率。
- Judge 异常（如 429、结构化输出失败）会单独统计为 `judge_error_*`，与真实 `pass/partial/fail` 分开。

---

## 四、评估数据集

| 数据集 | 文件 | 用途 | 条数 | 格式 |
|--------|------|------|------|------|
| QA 评估集 | `qa_dataset.json` | RAG 单跳 + Agent E2E（含多跳） | ~30 (含 10 多跳) | `{question, ground_truth, source_doc, question_type}` |
| Intent 评估集 | `intent_dataset.json` | 路由准确率 | 30 (qa/quiz/gap 各 10) | `{query, expected_intent}` |
| Quiz 评估集 | `quiz_eval_dataset.json` | 出题质量 | 10 组 | `{topic, query, source_doc}` |
| Gap 评估集 | `gap_eval_dataset.json` | 盲区检测 | 6 组 | `{scenario_id, mastery_preset, scope, expected_gap_mode}` |

**生成方式**：半自动（LLM 从 Milvus chunks 生成候选 → 人工审核定稿）。

---

## 五、技术实现

- **RAGAS 框架**: `pip install ragas`, 传入 `(question, answer, contexts, ground_truth)` 四元组
- **Judge 模型**: Kimi-K2.5 (Moonshot API, OpenAI-compatible)
- **评估脚本**: `backend/evaluation/` 目录, 每个维度独立模块, 统一 runner 输出 JSON 报告
- **运行方式**:
  - 并行跑 RAG + Agent: `cd backend && MILVUS_HOST=localhost python -m evaluation.eval_all --profile smoke`
  - 单独跑 RAG: `cd backend && MILVUS_HOST=localhost python -m evaluation.eval_rag --profile smoke`
  - 单独跑 Agent: `cd backend && MILVUS_HOST=localhost python -m evaluation.eval_agent --profile smoke`
  - 将 `smoke` 替换为 `standard` / `full` 切换档位
- **RAG 输出增强**: 每次运行仅写入 `evaluation/results/rag/rag_eval_YYYYMMDD_HHMMSS.json`（`YYYYMMDD_HHMMSS` 为洛杉矶本地），`meta.timezone` / `meta.started_at` / `meta.finished_at`（ISO 含偏移）、`meta.output_file`，并记录 `ragas_metric_status`
- **Agent 输出增强**: 每次运行仅写入 `evaluation/results/agent/agent_eval_YYYYMMDD_HHMMSS.json`（同上为洛杉矶本地），`meta` 字段同上，并记录 `metric_scope`、当前 `profile`、子集组成、judge error 统计
