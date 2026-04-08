# 评估与优化迭代记录

本文档按迭代轮次记录每次评估结果、问题分析和改进方案，展示系统从基准到优化的完整闭环。

评估框架：RAG 用 RAGAS 框架，Agent 用任务驱动评估。Judge 模型：Kimi K2.5（非 thinking 模式）。

---

## 迭代 0: 基准测试 (Baseline)

**日期**: 2026-04-06
**目的**: 建立初始基准，识别系统瓶颈

### RAG 测试结果

#### RAG — Smoke (6 samples: 4 单跳 + 2 多跳)

| 指标 | 得分 | 目标 | 状态 |
|---|---|---|---|
| Recall@K | 1.000 | >= 0.85 | 达标 |
| MRR | 1.000 | >= 0.70 | 达标 |
| Faithfulness | 0.748 | — | 偏低 |
| Answer Relevancy | 0.890 | — | 良好 |
| Context Precision | 0.875 | — | 良好 |
| Context Recall | 0.917 | — | 良好 |
| Reranker Lift | 0.000 | > 0 | 无增益 |
| Multi-hop Lift | -0.171 | > 0 | 不达标 |
| Latency P50 | 37,256 ms | — | 严重超标 |
| Latency P95 | 38,007 ms | < 500 ms | 超标 76x |

数据文件: `evaluation/results/rag/rag_eval_20260406_142834.json`

#### RAG — Full (30 samples: 20 单跳 + 10 多跳)

| 指标 | Smoke | Full | 变化 | 状态 |
|---|---|---|---|---|
| Recall@K | 1.000 | 0.950 | -0.050 | 达标 |
| MRR (reranked) | 1.000 | 0.950 | -0.050 | 达标 |
| MRR (raw) | 1.000 | 0.960 | -0.040 | 达标 |
| Faithfulness | 0.748 | 0.713 | -0.035 | 偏低 |
| Answer Relevancy | 0.890 | 0.888 | -0.002 | 良好 |
| Context Precision | 0.875 | 0.719 | -0.156 | 偏低 |
| Context Recall | 0.917 | 0.847 | -0.070 | 良好 |
| Reranker Lift | 0.000 | -0.010 | — | 负增益 |
| Multi-hop Lift | -0.171 | -0.316 | -0.145 | 恶化 |
| Latency P50 | 37,256 ms | 22,681 ms | -14,575 | 改善但仍慢 |
| Latency P95 | 38,007 ms | 44,895 ms | +6,888 | 长尾恶化 |

Multi-hop 分项:
- 单跳 Faithfulness: 0.819 | 多跳 Faithfulness: 0.503 (降幅 -0.316)
- 多跳 Context Precision: 10 样本中 6 个为 0.0（完全检索不到相关上下文）
- 多跳 Context Recall: 最低 0.333，均值约 0.59

数据文件: `evaluation/results/rag/rag_eval_20260406_173534.json`

#### RAG 问题分析

| 优先级 | 问题 | 现象 | 根因 |
|---|---|---|---|
| P1 | Reranker 负增益 (Lift = -0.01) | MRR_raw=0.960, MRR_reranked=0.950，reranker 反而打乱了正确排序 | bge-reranker-v2-m3 对已高质量的 embedding 结果做 cross-encoder 重排时，偶尔将正确结果降级。smoke Lift=0.000，full 确认为负增益 |
| P2 | Multi-hop 全面崩塌 | 单跳 faithfulness=0.819，多跳=0.503；多跳 Context Precision 10 样本中 6 个为 0.0 | 检索层单次 query 无法同时命中两个不同主题的 chunks；生成层在上下文不足时大量"补全" |
| P3 | Faithfulness 整体偏低 (0.713) | 全量 30 样本平均 0.713，多个样本低于 0.4 | Executor prompt 未强制限制"不编造"，LLM 倾向添加背景知识和延伸讨论 |
| P4 | Context Precision 下降 (0.875→0.719) | full profile 下 6 个样本完全为 0.0 | 主要由 multi-hop 问题拉低（单跳 precision 基本正常） |

#### RAG 改进方案

| 优先级 | 问题 | 改进方案 |
|---|---|---|
| P1 | Reranker 负增益 | 待定：考虑禁用或替换 reranker |
| P2 | Multi-hop 崩塌 | 待定：需改进检索策略（query decomposition 或多次检索） |
| P3 | Faithfulness 偏低 | 待定：QA prompt 强化"仅基于检索内容回答"约束 |

---

### Agent 测试结果

#### Agent — Smoke (6 intent + 4 QA E2E)

| 指标 | 得分 | 目标 | 状态 |
|---|---|---|---|
| Intent Accuracy | 100% (6/6) | >= 95% | 达标 |
| QA Pass Rate | 25% (1/4) | >= 80% | 严重不达标 |
| Orchestrator Error | 75% (3/4 超时) | 0% | — |
| Judge Error | 0% | — | 良好 |
| E2E Latency P50 (成功) | 89,833 ms | < 2,000 ms | 超标 45x |

数据文件: `evaluation/results/agent/agent_eval_20260406_144031.json`

#### Agent — Full (30 intent + 30 QA E2E)

| 指标 | Smoke | Full | 变化 | 状态 |
|---|---|---|---|---|
| Intent Accuracy | 100% (6/6) | 96.7% (29/30) | -3.3% | 达标 |
| QA Pass Rate | 25% (1/4) | 36.7% (11/30) | +11.7% | 严重不达标 |
| QA Pass Rate (evaluated) | — | 37.9% (11/29) | — | 严重不达标 |
| Orchestrator Timeout | 75% (3/4) | 60% (18/30) | -15% | 改善但仍严重 |
| Judge Error | 0% | 3.3% (1/30) | +3.3% | 可接受 |
| E2E Latency P50 (成功) | 89,833 ms | 86,972 ms | -2,861 | 仍然过慢 |
| E2E Latency P95 (成功) | — | 115,753 ms | — | 接近超时 |
| E2E Latency P50 (全量) | — | 120,005 ms | — | 中位数即超时 |

Intent 错误详情:
- 唯一错分: `"Transformer和RAG结合起来看的话，有哪些知识点是容易搞混或者遗漏的？"` -> 预期 gap_analysis，分类为 qa（边界 case，用户口语化表达盲区诊断需求）

QA E2E 成功 case 特征: 通过的 11 个全是简单单跳事实型/概念型问题，Timeout 的 18 个均为 120s 精确超时。

数据文件: `evaluation/results/agent/agent_eval_20260406_173539.json`

#### Agent 问题分析

| 优先级 | 问题 | 现象 | 根因 |
|---|---|---|---|
| P0 | E2E 超时率 60% — 系统不可用 | 30 个 QA 中 18 个超时 (120s)，QA Pass Rate 仅 36.7% | QA Agent pipeline 过重，单次请求需 5-10 次 LLM 调用（Orchestrator plan_node + QA Planner + N x Executor ReAct + Synthesize + QC + Replan），叠加每次 RAG 检索耗时，总耗时轻松超 120s。仅简单单跳问题能在 120s 内完成（P50=87s） |
| P5 | Intent Classification 边界 case | 1/30 错分 (gap_analysis → qa) | 用户口语化表达盲区诊断需求，planner 将其理解为 QA |

#### Agent 改进方案

| 优先级 | 问题 | 改进方案 |
|---|---|---|
| P0 | Agent 超时 60% | QA SubAgent 架构重构：Plan & Execute → ReAct + Reflection，LLM 调用从 5-10 次降到 2-4 次 |
| P0 | Agent 超时 60% | Orchestrator plan_node prompt 重写为原子任务拆解框架，将 query 拆解职责从 QA 内部上移到 Orchestrator |
| P0 | Agent 超时 60% | Orchestrator 辅助 prompt 优化（REWRITE/REPLAN/SYNTHESIZE），减少无效 replan 和计划膨胀 |

---

## 迭代 1: 架构优化 — 降低 LLM 调用链长度 + Prompt 重写

**日期**: 2026-04-06
**目标**: 解决 P0（Agent 超时率 60%），同时优化 Orchestrator prompt 质量

### 改进内容

#### Agent 改进

**1. QA SubAgent 架构重构**
- **旧架构**: Planner -> Executor (ReAct loop per step) -> Synthesize -> QC (max 2 retries)
- **新架构**: ReAct (max 2 轮, rag_search tool-calling) -> Reflection -> fail 时带 feedback 再跑 1 轮 ReAct
- **删除**: Planner、Synthesizer、QC retry 循环、query_decompose tool
- **LLM 调用次数**: 3-13 次 -> 2-4 次
- **影响文件**: `backend/app/agent/subagents/qa_agent.py`

**2. Orchestrator plan_node Prompt 重写**
- 从"意图分类 + 6 个 few-shot"改为"原子任务拆解框架 + 2 个精简示例"
- 定义原子任务 3 条判定标准：单一检索意图 + 单一概念聚焦 + 独立可回答
- 新增拆解判断三步流程：识别意图 -> 检查可拆性 -> 确定依赖
- 将 query 拆解职责从 QA SubAgent 内部上移到 Orchestrator
- **影响文件**: `backend/app/agent/nodes/orchestrator.py` (PLAN_SYSTEM_PROMPT)

**3. Orchestrator 辅助 Prompt 优化**
- **REWRITE_SYSTEM_PROMPT**: 补充改写目标（向量检索友好）、前序结果提取规则（gap->取最低 mastery 盲区）、原子性约束、fallback
- **REPLAN_SYSTEM_PROMPT**: continue 设为默认偏好、revise 上限 3 步防膨胀、示例从 4 个砍到 1 个、新增判断标准
- **SYNTHESIZE_SYSTEM_PROMPT**: 按意图组合给汇总策略、quiz JSON 透传不改写、去掉 400 字硬限制
- **影响文件**: `backend/app/agent/nodes/orchestrator.py`

#### RAG 改进

**4. 评估框架调整：RAG eval 移除 multi-hop + Prompt grounding 强化**

**背景**: 迭代 1 将 multi-hop query 拆解职责上移到 Orchestrator，RAG 层只处理单跳原子任务，因此 RAG eval 测多跳是测一个生产中不存在的场景。

**改动**:

| 文件 | 改动 |
|---|---|
| `evaluation/profiles.py` | 移除 `rag_multi_hop_limit`，RAG eval 只使用单跳样本 |
| `evaluation/eval_rag.py` | 移除 Phase 2 (multi-hop RAG) 和 Phase 4 (multi_hop_lift)；重写 QA_SYSTEM_PROMPT grounding 约束 |
| `evaluation/ragas_runner.py` | 移除 `compute_multi_hop_lift` 函数 |
| `app/agent/subagents/qa_agent.py` | 重写 REACT_SYSTEM_PROMPT：从"不要编造"升级为"允许推理综合，禁止引入外部知识" |

**Prompt grounding 策略**:
- 允许：基于检索内容进行归纳、对比、因果推理
- 禁止：引入检索内容中未提及的概念、数据、案例或结论
- Faithfulness 目标设为 0.85（而非 1.0），为合理推理留空间

**Multi-hop 评估归属调整**:
- RAG eval：只测单跳检索+生成质量
- Agent eval：E2E 覆盖 multi-hop（Orchestrator 拆解 → 多次单跳 QA → 合成）

---

### RAG 测试结果

#### RAG — Full (20 samples: 仅单跳，multi-hop 已移除)

| 指标 | 得分 | 状态 |
|---|---|---|
| Recall@K | 0.950 | 达标 |
| MRR (reranked) | 0.950 | 达标 |
| MRR (raw) | 0.960 | 达标 |
| Faithfulness | **0.934** | **优秀 (↑)** |
| Answer Relevancy | 0.808 | 一般 (↓) |
| Context Precision | 0.822 | 良好 |
| Context Recall | 0.925 | 优秀 |
| Reranker Lift | -0.010 | 仍为负增益 |
| Latency P50 | 19,064 ms | 改善 |
| Latency P95 | 41,181 ms | 改善 |

数据文件: `evaluation/results/rag/rag_eval_20260406_213345.json`

#### RAG 对比分析 (vs Baseline)

> 为公平对比，Baseline 取旧版 full 中的 20 条单跳子集指标。

| 指标 | Baseline (单跳) | 迭代 1 | Δ | 评价 |
|---|---|---|---|---|
| Recall@K | 0.950 | 0.950 | +0.000 | 不变 |
| MRR | 0.950 | 0.950 | +0.000 | 不变 |
| **Faithfulness** | 0.819 | **0.934** | **+0.115** | **大幅提升** |
| Answer Relevancy | 0.904 | 0.808 | -0.096 | 下降 |
| Context Precision | 0.831 | 0.822 | -0.009 | 基本持平 |
| Context Recall | 0.925 | 0.925 | +0.000 | 不变 |
| Reranker Lift | -0.010 | -0.010 | +0.000 | 仍为负增益 |
| Latency P50 | 22,681 ms | 19,064 ms | -3,617 | 改善 |
| Latency P95 | 44,895 ms | 41,181 ms | -3,714 | 改善 |

**改善点**:
1. **Faithfulness +11.5%** — 本次最显著改善。新 prompt 的 grounding 约束（"每个论述必须追溯到检索结果"+"禁止引入未提及的概念"）生效。旧版 6 个样本 < 0.7，新版仅 1 个。
2. **Latency 改善 ~3.5s** — QA SubAgent 从 Plan & Execute 改为 ReAct + Reflection，LLM 调用链缩短。
3. **Q2 (MEMORY.md 记忆类型) faithfulness 0.50 → 1.00** — 之前 LLM 编造额外信息，现在严格基于检索内容。
4. **Q13 (LLM 生成测试集) faithfulness 0.60 → 0.75** — 同上，减少编造。

**退步点**:
1. **Answer Relevancy 下降 -9.6%** — 主要由 3 个样本拉低：
   - Q16 (FFN 维度变化): **relevancy=0.00** — 检索完全失败 (context_recall=0)，新 prompt 下模型诚实拒答"信息不足"，而非编造。旧版编造回答 faithfulness=0.33 但 relevancy 看起来更高 → 这是 faithfulness 提升的正确副作用。
   - Q15 (上下文爆炸解法): relevancy=0.51 — 检索不精准 (context_precision=0)，回答信息发散。
   - Q12 (Causal Mask): relevancy=0.65 — 回答偏长。

**持续存在的问题**:
1. **Q15 + Q16 两次测试都全面低分** — 检索层问题，非生成层：
   - Q16 "FFN 内部维度变化流程": context_recall=0, context_precision=0 → chunk 中可能未包含 `d_model → 4d_model → d_model` 的完整描述，或该信息被切断
   - Q15 "上下文爆炸解法": context_recall=0.5, context_precision=0 → 检索到部分内容但排序不佳
2. **Reranker 持续负增益 (-0.01)** — 两轮测试一致，bge-reranker-v2-m3 对高质量 embedding 结果无正贡献

---

### Agent 测试结果

> 本轮未单独跑 Agent eval，Agent 改进效果待迭代 2 验证。

---

## 迭代 2: Embedding Heading Augmentation — 检索精度优化

**日期**: 2026-04-07
**目标**: 解决 Q15/Q16 检索排名不佳的问题

### 根因分析

排查发现 Q15/Q16 的正确 chunk **存在于 Milvus 中但检索排名不够高**：

- **Q16 (FFN 维度变化)**: 正确 chunk (`6. Feed Forward Network（FFN）/工作流程`) 在 hybrid search 中排第 3，但 chunk 文本不含 "FFN" 关键词（只在 heading 中），rerank 后被挤出 top 5
- **Q15 (上下文爆炸解法)**: 正确 chunk (`Agent 常见失败模式/3. 上下文爆炸（Context Overflow）`) 未进入 hybrid search top 10，被来自另一篇文档的大量"上下文"相关 chunk 淹没。chunk 文本不含"上下文爆炸"或"Context Overflow"（只在 heading 中）

**共性根因**: embedding 只编码 `child.text`，不包含 `heading_path` 中的关键词，导致 chunk 向量与 query 语义距离过远。

### 改进内容

**Embedding Heading Augmentation** — 在 indexing 阶段，embedding 编码时将 `heading_path` 拼接到 chunk text 前面。

| 文件 | 改动 |
|---|---|
| `backend/app/api/routes/ingest.py` | 新增 `_build_embed_texts()` 辅助函数，两处 embedding 调用改为用 `heading_path\n{text}` 编码 |

- Milvus 中 `text` 字段仍存原文（给 LLM 和 reranker 用）
- `embedding` 和 `sparse_embedding` 基于拼接后的文本生成
- 无 heading 时退化为原文，不影响无结构文档
- 需重新 ingest 所有文档使新 embedding 生效

### 测试结果

#### RAG — Full (20 samples: 单跳)

| 指标 | 迭代 2 | 迭代 1 | Δ | 状态 |
|---|---|---|---|---|
| Recall@K | 0.950 | 0.950 | +0.000 | 达标 |
| MRR (reranked) | 0.950 | 0.950 | +0.000 | 达标 |
| MRR (raw) | **1.000** | 0.960 | **+0.040** | **提升** |
| Faithfulness | 0.898 | **0.934** | -0.036 | 仍达标但回落 |
| Answer Relevancy | 0.792 | 0.808 | -0.016 | 略低于目标 0.80 |
| Context Precision | 0.831 | 0.822 | +0.008 | 小幅提升 |
| Context Recall | **0.950** | 0.925 | **+0.025** | **提升** |
| Reranker Lift | **-0.050** | -0.010 | **-0.040** | 明显恶化 |
| Latency P50 | 21,807 ms | 19,064 ms | +2,743 | 中位变慢 |
| Latency P95 | 38,702 ms | 41,181 ms | -2,479 | 长尾改善 |

成功数据文件: `evaluation/results/rag/rag_eval_20260407_140426.json`

#### RAG — Full 首次运行失败（稳定性问题）

- 首次 full run (`evaluation/results/rag/rag_eval_20260407_135500.json`) 在 IR phase 并发执行时失败
- 报错: `RuntimeError: Already borrowed`
- 调用栈定位到 reranker tokenizer (`transformers.tokenization_utils_fast.py -> enable_truncation`)
- 影响：IR phase 无结果，后续 RAGAS phase 因 0 samples 全部报 `IndexError`

失败数据文件: `evaluation/results/rag/rag_eval_20260407_135500.json`

### 对比分析 (vs 迭代 1)

#### 改善点

1. **Heading augmentation 确实提升了原始检索排序质量**
   - `MRR raw: 0.960 -> 1.000`
   - `Context Recall: 0.925 -> 0.950`
   - `Context Precision: 0.822 -> 0.831`
   说明把 `heading_path` 拼进 embedding 后，query 与 chunk 的语义对齐更好，尤其有利于关键词主要出现在标题中的样本。

2. **Q16 (FFN 维度变化) 的召回被修复**
   - `context_recall: 0.0 -> 1.0`
   这是本轮改动最直接命中的目标 case，证明 heading augmentation 对 heading-sensitive 检索问题有效。

3. **P95 延迟下降**
   - `41.2s -> 38.7s`
   长尾略有改善，说明并行化和 warmup 对尾部请求有帮助。

#### 退步点

1. **Faithfulness 回落**
   - `0.934 -> 0.898`
   检索覆盖变好后，生成并没有同步提升，部分样本出现“拿到更多/更相关上下文，但回答组织变差”的情况。

2. **Answer Relevancy 再次跌破目标**
   - `0.808 -> 0.792`
   主要问题仍集中在少数难例，检索侧修复没有自然传导到最终回答质量。

3. **Reranker 负增益显著恶化**
   - `Reranker Lift: -0.01 -> -0.05`
   - `MRR raw` 已到 1.000，但 `MRR reranked` 仍停在 0.950
   说明 reranker 正在破坏已经被 heading augmentation 修好的原始排序，而不是带来增益。

#### case 级结论

1. **Q16 (FFN 内部维度变化流程)**  
   目标 case 的 `context_recall` 从 `0.0 -> 1.0`，但 `faithfulness` 从 `0.833 -> 0.4`，`answer_relevancy` 仍为 `0.0`。  
   结论：召回修复了，但最终回答仍失败，问题已从“检索不到”转向“rerank/生成没有用好召回结果”。

2. **Q15 (上下文爆炸解法)**  
   `context_recall` 从 `0.5 -> 0.0`，未按预期改善。  
   结论：heading augmentation 不是通用解，这类中英别名/故障模式表达仍需 query rewrite、lexical boost 或更大的候选池。

#### 结论

- **迭代 2 成功改善了检索层 raw ranking，但未改善最终 reranked 结果**
- **本轮最大瓶颈已从 embedding 召回转移到 reranker**
- 下一轮应优先做 **“关闭 reranker 的公平 A/B”**，保持最终传给 LLM 的 chunk 数固定为 `top_k`，只比较排序来源差异

### 遗留待观察

| 问题 | 状态 |
|---|---|
| Reranker 负增益 (-0.05) | 已确认需要做 reranker off A/B |
| Answer Relevancy 下降 | heading augmentation 未自然修复，需继续观察生成侧与排序侧联动 |
| Reranker 并发稳定性 (`Already borrowed`) | 已暴露，需结合 reranker off A/B 一并判断是否继续保留 |

### 本轮后续改动记录

为准备下一轮 A/B，已新增 reranker 配置开关：

- `backend/app/config.py`：新增 `enable_reranker: bool = True`
- `backend/app/agent/tools/rag_search.py`：关闭 reranker 时，直接返回 Milvus 原始排序的前 `top_k` 个 chunk
- 关键约束：**关闭 reranker 后，最终传给 LLM 的 chunk 数保持不变，仍为 `top_k`**

这样下一轮 `eval_rag` 可以公平比较：
- reranker on：Milvus `top_k*3` 候选 → reranker → 前 `top_k`
- reranker off：Milvus `top_k*3` 候选 → 直接取原始排序前 `top_k`

---

## 迭代 3: Reranker Off 公平 A/B

**日期**: 2026-04-07
**目标**: 验证当前 reranker 是否仍有正向价值，并排除“最终传给 LLM 的 chunk 数变化”这个混杂变量

### 改动前提

- 已新增 `enable_reranker` 开关
- 关闭 reranker 时，仍先从 Milvus 取 `top_k * 3` 候选，但最终只保留原始排序前 `top_k` 个 chunk 传给 LLM
- 因此本轮对比只改变排序来源，不改变最终上下文长度

### 测试结果

#### RAG — Full (20 samples: 单跳, reranker off)

| 指标 | 迭代 3 | 迭代 2 | Δ | 状态 |
|---|---|---|---|---|
| Recall@K | **1.000** | 0.950 | **+0.050** | **提升** |
| MRR | **1.000** | 0.950 | **+0.050** | **提升** |
| MRR (raw) | 1.000 | 1.000 | +0.000 | 持平 |
| Faithfulness | 0.896 | 0.898 | -0.002 | 基本持平 |
| Answer Relevancy | **0.834** | 0.792 | **+0.042** | **提升** |
| Context Precision | **0.967** | 0.831 | **+0.136** | **显著提升** |
| Context Recall | **1.000** | 0.950 | **+0.050** | **提升** |
| Latency P50 | **19,364 ms** | 21,807 ms | **-2,443** | **改善** |
| Latency P95 | **28,212 ms** | 38,702 ms | **-10,490** | **显著改善** |

数据文件: `evaluation/results/rag/rag_eval_20260407_142828.json`

运行日志: `evaluation/results/logs_rag/eval_rag_20260407_142828.log`

### 对比分析 (vs 迭代 2)

#### 改善点

1. **关闭 reranker 后，检索指标全面转正**
   - `Recall@K: 0.95 -> 1.00`
   - `MRR: 0.95 -> 1.00`
   - `Context Recall: 0.95 -> 1.00`
   - `Context Precision: 0.831 -> 0.967`
   这说明 heading augmentation 带来的原始排序收益，在 reranker 关闭后被完整释放出来了。

2. **生成相关性显著回升**
   - `Answer Relevancy: 0.792 -> 0.834`
   Faithfulness 基本不变，说明这次提升不是靠“更敢编”，而更像是检索上下文更贴题带来的自然改善。

3. **延迟明显下降**
   - `P50: 21.8s -> 19.4s`
   - `P95: 38.7s -> 28.2s`
   关闭 cross-encoder 后，时延和长尾都得到实质改善。

#### case 级结论

1. **Q15 (上下文爆炸解法) 被修复**
   - `context_recall: 0.0 -> 1.0`
   - `context_precision: 0.0 -> 1.0`
   - `answer_relevancy: 0.51 -> 0.71`
   说明这类 heading-sensitive / 别名敏感问题，之前更像是被 reranker 重新排坏了。

2. **Q16 (FFN 内部维度变化流程) 被实质修复**
   - `answer_relevancy: 0.0 -> 0.819`
   - `faithfulness: 0.4 -> 0.857`
   这证明问题不只是“召回到了没用上”，而是 reranker 关闭后最终回答也一起恢复。

3. **仍有少数题生成波动**
   - 如 `Checkpointer 四种核心能力`、`构建测试集方案` 等样本的 faithfulness 有回落
   说明 raw 排序虽然整体更优，但少数 case 仍存在上下文组织和回答压缩问题。

### 重要说明

- 本轮真实在线 RAG pipeline 是 **reranker off**
- 但当前 `eval_rag.py` 的 `compute_reranker_lift()` 仍会单独调用 reranker 做离线对照
- 因此结果 JSON 中的 `mrr_reranked` / `reranker_lift` 仍是诊断指标，**不是本轮在线配置本身的直接反映**

### 结论

- **当前阶段 reranker 已确认是负贡献项，关闭后整体效果更好**
- **Heading augmentation + raw hybrid retrieval 已经把单跳 RAG 检索质量推到较高水平**
- 后续 RAG 优化重点不应再放在“是否保留现有 reranker”，而应转向：
  - 回答组织 / prompt 压缩
  - 结果报表补充 `meta.reranker_enabled`
  - 如需重启 reranker，必须让其输入与 embedding 一样包含 `heading_path`

---

## 迭代 4: Agent 架构优化 — 降低编排开销 + 消除 LLM 冗余调用

**日期**: 2026-04-07
**目标**: 解决最新 Agent eval 暴露的 P0（QA 编排过度导致超时）、P1（Quiz 证据绑定失败、Gap agent 大输入超时）、P2（通过样本延迟仍过高）

### 上一轮 Agent 测试结果回顾

数据文件: `evaluation/results/agent/agent_eval_20260407_141329.json`

| 指标 | 得分 | 状态 |
|---|---|---|
| Intent Accuracy | 93.3% (14/15) | 达标 |
| QA Pass Rate | 40% (6/15) | 严重不达标 |
| Orchestrator Timeout | 9/15 | 严重 |
| Judge Error | 0% | 良好 |
| Quiz Valid Rate | 16.7% (2/12) | 严重不达标 |
| Gap Timeout | 2/4 场景 | 严重 |
| E2E Latency P50 (通过) | 97,490 ms | 过慢 |

### 问题分析

| 优先级 | 问题 | 现象 | 根因 |
|---|---|---|---|
| P0 | QA 编排过度 | 9/15 超时，全部是 comparison / multi-hop 类型；通过样本全为单步 plan | plan_node prompt 将对比类/因果类问题强制拆为 2-3 个 qa 步骤，每步走完整 ReAct + Reflection 流水线（~4 次 LLM 调用），总计 ~9 次 LLM 调用必然超时 |
| P1 | Quiz 评测证据绑定失败 | quiz_valid_rate=16.7%，12 道题中 10 道 fail，原因全是"知识材料为空" | `_generate_questions()` 按 LLM 生成的题目 topic 匹配 source_chunks，但 topic 漂移（如 "Self-Attention" → "Multi-Head Attention"）导致匹配失败，judge 收到空材料 |
| P1 | Gap agent 大输入超时 | all_gap (259 topics) 和 partial_gap 场景均超时 120s | LLM 自行调用 search_concepts + read_mastery 后对 259 个 topic 做交叉分析，本质是规则计算却让 LLM 做推理 |
| P2 | 通过样本延迟过高 | P50=97s，最快通过样本也需 41s | 单步 plan 仍需 5 次 LLM 调用：plan_node + ReAct(tool_call + answer) + reflection + replan，其中 replan 对单步 plan 始终返回 finish，纯浪费 |

### 改进内容

#### Agent 改进

**1. Plan Prompt 重写 — 默认不拆**

- **旧策略**: "原子任务拆解框架"，对比类/因果类/并列类强制拆分为多个 qa
- **新策略**: "默认不拆，qa+qa 默认禁止，只有不同意图类型才拆分"
- 核心规则：对比类、并列类、因果类问题都归为单个 qa
- 新增 gap_analysis 区分规则：诊断用户个人掌握情况 → gap_analysis；问通用知识 → qa
- 示例从 3 个详细 JSON 精简为 2 个单行 JSON（不拆 + 拆）
- **影响文件**: `backend/app/agent/nodes/orchestrator.py` (PLAN_SYSTEM_PROMPT)

**2. 单步 Plan 跳过 Replan 节点**

- 新增 `route_after_execute()` 路由函数
- 编译图路由: `execute_batch → route_after_execute`
  - 单步 plan → 直接进 passthrough（跳过 replan LLM 调用）
  - 多步 plan → 进 replan 节点（保持原逻辑）
- 流式路径同步更新
- 预期每次单步请求省 1 次 LLM 调用 (~10-15s)
- **影响文件**: `backend/app/agent/nodes/orchestrator.py`, `backend/app/agent/graph.py`

**3. Quiz 证据绑定 — 全量附加**

- **旧逻辑**: 按 LLM 生成的题目 topic 匹配 materials（`t == topic or topic in content`），漂移时 source_chunks 为空
- **新逻辑**: 直接将全部 materials 附加到每道题的 source_chunks，不再按 topic 匹配
- **影响文件**: `backend/app/agent/subagents/quiz_agent.py`

**4. Gap Agent 重构 — Python 硬编码分类 + LLM 报告生成**

- **旧架构**: LLM tool-calling 循环（search_concepts + read_mastery → LLM 交叉分析 259 个 topic → 生成报告），MAX_ITERATIONS=10
- **新架构**: Python 直接调用 search_concepts + read_mastery → 确定性规则分类（4 个桶：未接触盲区/薄弱盲区/需巩固/已掌握）→ LLM 只负责格式化报告文本
- 分类规则硬编码：无 mastery 记录 → 未接触盲区；effective_mastery < 0.3 → 薄弱盲区；< 0.5 → 需巩固；>= 0.5 → 已掌握
- 消除 LLM tool-calling 循环，分类耗时从 120s+ 降为毫秒级
- **影响文件**: `backend/app/agent/subagents/gap_agent.py`

### Agent 测试结果

> 待测试。
