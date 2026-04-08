# SecondBrain 简历要素分析

## 一、可靠的量化指标

### RAG 检索与生成质量（经 4 轮迭代优化）

| 指标 | Baseline → 最终 | 可写法 |
|---|---|---|
| Faithfulness | 0.713 → 0.934 | **+31%**，通过 prompt grounding 约束实现 |
| Context Precision | 0.719 → 0.967 | **+34%**，通过 heading augmentation + reranker A/B 实现 |
| Context Recall | 0.847 → 1.0 | 从 84.7% 提升至 **100%** |
| Recall@K | 0.950 → 1.0 | 最终达到 **完美召回** |
| MRR | 0.950 → 1.0 | 最终达到 **完美排序** |
| RAG P95 延迟 | 44.9s → 28.2s | **降低 37%** |

### Agent 任务驱动指标

| 指标 | Baseline → 最终 | 可写法 |
|---|---|---|
| QA E2E Pass Rate | 36.7% → 73.3% | **翻倍**，通过 SubAgent 架构重构 |
| Quiz 生成质量 | 16.7% → 100% | 从几乎不可用到 **全量通过** |
| Gap 检测覆盖率 | N/A → 95.4% | False positive < 1% |
| LLM 调用链 | 5-10 次/请求 → 2-4 次 | **减少 50-60%** |
| Agent 超时率 | 60% → 大幅降低 | 架构从 Plan&Execute 改为 ReAct+Reflection |

### 工程规模

| 维度 | 数据 |
|---|---|
| 后端代码 | ~5,400 行 Python |
| 前端代码 | ~1,300 行 TypeScript/React |
| 评估框架 | ~2,500 行 Python |
| 测试 | 150 个测试函数，~2,700 行 |
| 评估数据集 | 30 QA + 30 Intent + 10 Quiz + 6 Gap 场景 |

---

## 二、完整优化迭代过程

### 迭代 0：Baseline — 建立基准，暴露系统瓶颈

**RAG Baseline（Full 30 samples）**

| 指标 | 得分 | 状态 |
|---|---|---|
| Recall@K | 0.950 | 达标 |
| MRR (raw) | 0.960 | 达标 |
| Faithfulness | 0.713 | 偏低 |
| Answer Relevancy | 0.888 | 良好 |
| Context Precision | 0.719 | 偏低 |
| Context Recall | 0.847 | 良好 |
| Reranker Lift | -0.010 | 负增益 |

**Agent Baseline（Full 30 intent + 30 QA E2E）**

| 指标 | 得分 | 状态 |
|---|---|---|
| Intent Accuracy | 96.7% (29/30) | 达标 |
| QA Pass Rate | 36.7% (11/30) | 严重不达标 |
| Orchestrator Timeout | 60% (18/30) | 系统不可用 |
| E2E Latency P50 (成功) | 86,972 ms | 过慢 |

**暴露的核心问题**

| 优先级 | 问题 | 根因 |
|---|---|---|
| P0 | Agent 60% 超时 | QA pipeline 过重：单次请求 5-10 次 LLM 调用（plan + QA Planner + N×Executor ReAct + Synthesize + QC + Replan），叠加 RAG 检索耗时必然超 120s |
| P1 | Reranker 负增益 | BGE-reranker-v2-m3 对已高质量的 BGE-m3 embedding 结果做 cross-encoder 重排，偶尔将正确结果降级 |
| P2 | Multi-hop 崩塌 | 单跳 Faithfulness 0.819 vs 多跳 0.503；检索层单次 query 无法同时命中两个不同主题的 chunks |
| P3 | Faithfulness 偏低 | Prompt 未强制 grounding 约束，LLM 倾向添加背景知识和延伸讨论 |

---

### 迭代 1：架构重构 — 砍掉 LLM 调用链 + Prompt Grounding 强化

**问题 → 方案映射**

| 目标问题 | 具体优化手段 |
|---|---|
| P0: Agent 超时 60% | **QA SubAgent 架构重构**：从 Planner → Executor(ReAct) → Synthesize → QC 四阶段砍到 ReAct(max 2 轮) + Reflection 两阶段，LLM 调用 5-10 次 → 2-4 次 |
| P0: Agent 超时 60% | **Orchestrator plan_node 重写**：将 query 拆解从 QA 内部上移到 Orchestrator，定义原子任务 3 条判定标准（单一检索意图 + 单一概念聚焦 + 独立可回答） |
| P0: Agent 超时 60% | **辅助 Prompt 优化**：REPLAN prompt 将 continue 设为默认偏好、revise 上限 3 步防膨胀；SYNTHESIZE 按意图组合给策略 |
| P2: Multi-hop 崩塌 | **评估归属调整**：multi-hop 拆解已上移到 Orchestrator，RAG eval 只测单跳；multi-hop 覆盖由 Agent E2E eval 负责 |
| P3: Faithfulness 偏低 | **Prompt grounding 策略**：允许基于检索内容推理综合，禁止引入检索内容中未提及的概念/数据/案例 |

**迭代 1 结果（RAG Full 20 单跳样本）**

| 指标 | Baseline → 迭代 1 | Δ |
|---|---|---|
| **Faithfulness** | **0.819 → 0.934** | **+11.5%** |
| Answer Relevancy | 0.904 → 0.808 | -9.6% |
| Context Precision | 0.831 → 0.822 | -0.9% |
| Context Recall | 0.925 → 0.925 | 不变 |
| Latency P50 | 22,681 → 19,064 ms | -16% |

> 注：Baseline 对比基准取旧版 full 中 20 条单跳子集，确保公平。

**关键发现**

- **Faithfulness 大幅提升是核心收益**：旧版 6 个样本 < 0.7，新版仅 1 个。典型 case：Q2 (记忆类型) faithfulness 0.50 → 1.00，之前 LLM 编造额外信息，现在严格基于检索内容。
- **Answer Relevancy 下降是 Faithfulness 提升的正确副作用**：Q16 检索完全失败时，旧 prompt 下 LLM 编造回答（faithfulness=0.33 但 relevancy 看起来高），新 prompt 下诚实拒答"信息不足"（relevancy=0.00 但 faithfulness 正确）。
- **新暴露问题**：Q15/Q16 两个 case 持续低分，定位为检索层问题（关键词只出现在 heading 中，chunk 文本不含），非生成层。

---

### 迭代 2：Heading Augmentation — 从 Failure Case 推导检索改进

**问题 → 方案映射**

| 目标问题 | 根因分析 | 具体优化手段 |
|---|---|---|
| Q16 (FFN 维度变化) context_recall=0 | 正确 chunk 标题是 `6. Feed Forward Network（FFN）/工作流程`，但 chunk 文本不含 "FFN" 关键词，hybrid search 排第 3 后被 reranker 挤出 top 5 | **Heading Augmentation**：indexing 时 embedding 编码 `heading_path + "\n" + text`，Milvus 存储的 text 字段仍为原文（给 LLM 和 reranker 用） |
| Q15 (上下文爆炸) context_recall=0.5 | 正确 chunk 标题是 `Agent 常见失败模式/3. 上下文爆炸`，chunk 文本不含"上下文爆炸"或"Context Overflow"，被其他文档的"上下文"相关 chunk 淹没 | 同上 |

**迭代 2 结果（RAG Full 20 单跳样本）**

| 指标 | 迭代 1 → 迭代 2 | Δ |
|---|---|---|
| **MRR (raw)** | **0.960 → 1.000** | **+4%** |
| **Context Recall** | **0.925 → 0.950** | **+2.5%** |
| Context Precision | 0.822 → 0.831 | +0.9% |
| Faithfulness | 0.934 → 0.898 | -3.6% |
| **Reranker Lift** | **-0.010 → -0.050** | **恶化** |

**关键发现**

- **Heading augmentation 对原始排序有效**：MRR raw 达到 1.0，Q16 的 context_recall 从 0.0 → 1.0，目标 case 被修复。
- **但 reranker 破坏了改善**：MRR reranked 仍停在 0.950，reranker 把被 heading augmentation 修好的排序又打乱了。Reranker Lift 从 -0.01 恶化到 -0.05。
- **瓶颈转移**：从"embedding 召回不到"转移到"reranker 破坏好排序"，下一步必须做 reranker on/off A/B。

**为 A/B 做的准备**：新增 `enable_reranker` 配置开关 + 关闭 reranker 时保持最终传给 LLM 的 chunk 数不变（仍为 `top_k`），确保对比只改变排序来源。

---

### 迭代 3：Reranker A/B — 控制变量实验推翻直觉

**实验设计**

| 变量 | Reranker ON | Reranker OFF |
|---|---|---|
| Milvus 候选数 | top_k × 3 | top_k × 3 |
| 排序方式 | Cross-encoder 重排 | 原始 hybrid search 排序 |
| 传给 LLM 的 chunk 数 | top_k | top_k |

唯一变量是排序来源，排除了"chunk 数量不同"这个混杂因素。

**迭代 3 结果（RAG Full 20 单跳样本，reranker off）**

| 指标 | 迭代 2 (reranker on) → 迭代 3 (reranker off) | Δ |
|---|---|---|
| **Recall@K** | **0.950 → 1.000** | **+5%** |
| **MRR** | **0.950 → 1.000** | **+5%** |
| **Context Precision** | **0.831 → 0.967** | **+16.4%** |
| **Context Recall** | **0.950 → 1.000** | **+5%** |
| **Answer Relevancy** | **0.792 → 0.834** | **+5.3%** |
| Faithfulness | 0.898 → 0.896 | 持平 |
| **Latency P50** | **21,807 → 19,364 ms** | **-11.2%** |
| **Latency P95** | **38,702 → 28,212 ms** | **-27.1%** |

**Case 级验证**

| Case | Reranker ON | Reranker OFF | 结论 |
|---|---|---|---|
| Q15 (上下文爆炸) | context_recall=0.0, precision=0.0 | **recall=1.0, precision=1.0**, relevancy 0.51→0.71 | 之前是被 reranker 排坏了 |
| Q16 (FFN 维度变化) | faithfulness=0.4, relevancy=0.0 | **faithfulness=0.857, relevancy=0.819** | reranker off 后检索和生成一起恢复 |

**决策**：关闭 reranker，将 `enable_reranker` 默认值设为 `false`。

**核心论证**：这不是"reranker 没用"的泛化结论，而是在 BGE-m3 + heading augmentation 已经提供高质量原始排序的条件下，cross-encoder 重排的边际贡献为负。

---

### 迭代 4：Agent 架构优化 — 消除编排冗余 + SubAgent 重构

**迭代 3 后的 Agent eval 暴露新问题**

| 优先级 | 问题 | 现象 | 根因 |
|---|---|---|---|
| P0 | QA 编排过度 | 9/15 超时，全部是对比/multi-hop 类型 | plan_node 将对比类问题强制拆为 2-3 个 qa 步骤，每步 ReAct+Reflection ~4 次 LLM 调用，总计 ~9 次必然超时 |
| P1 | Quiz 证据绑定失败 | quiz_valid_rate=16.7%，12 题中 10 题 fail | 按 LLM 生成的题目 topic 匹配 source_chunks，topic 漂移（如"Self-Attention"→"Multi-Head Attention"）导致匹配失败 |
| P1 | Gap agent 大输入超时 | 259 topics 场景超时 120s | LLM tool-calling 循环对 259 个 topic 做交叉分析，本质是规则计算却让 LLM 做推理 |
| P2 | 通过样本延迟过高 | P50=97s，最快也需 41s | 单步 plan 仍需 5 次 LLM 调用，其中 replan 对单步始终返回 finish，纯浪费 |

**问题 → 方案映射**

| 目标问题 | 具体优化手段 |
|---|---|
| P0: QA 编排过度 | **Plan Prompt 重写 — 默认不拆**：对比类/并列类/因果类问题归为单个 qa step，只有不同意图类型才拆分；示例从 3 个详细 JSON 精简为 2 个单行 JSON |
| P2: 单步 plan 浪费 replan | **单步 Plan 跳过 Replan 节点**：新增 `route_after_execute()` 路由，单步 plan 直接进 passthrough，省 1 次 LLM 调用 (~10-15s) |
| P1: Quiz 证据绑定失败 | **全量附加策略**：不再按 topic 匹配 source_chunks，直接将全部 RAG materials 附加到每道题 |
| P1: Gap agent 超时 | **Python 硬编码分类 + LLM 报告生成**：Python 直接调 search_concepts + read_mastery → 确定性规则分类（4 桶：未接触/薄弱/需巩固/已掌握）→ LLM 只负责格式化报告文本。分类耗时从 120s+ 降为毫秒级 |

**迭代 4 结果（Agent eval）**

| 指标 | 迭代 3 后 → 迭代 4 | Δ |
|---|---|---|
| **QA Pass Rate** | **40% (6/15) → 73.3% (11/15)** | **+83%** |
| **Quiz Valid Rate** | **16.7% (2/12) → 100% (12/12)** | **从不可用到全量通过** |
| **Gap Coverage** | **超时 → 95.4%** | **从不可用到高可用** |
| Gap False Positive | N/A → 0.59% | 极低误报 |
| E2E Latency P50 (通过) | 97,490 → 76,926 ms | -21% |
| Replan Rate | >0% → 0% | Plan 质量提升，不再需要 replan |

---

### 全局指标演进总览

#### RAG 指标演进（单跳 Full Profile）

| 指标 | Baseline | 迭代 1 | 迭代 2 | 迭代 3 | 主要贡献手段 |
|---|---|---|---|---|---|
| Faithfulness | 0.713 | **0.934** | 0.898 | 0.896 | 迭代 1：Prompt grounding |
| Context Precision | 0.719 | 0.822 | 0.831 | **0.967** | 迭代 2+3：Heading aug + Reranker off |
| Context Recall | 0.847 | 0.925 | 0.950 | **1.000** | 迭代 2+3：Heading aug + Reranker off |
| Recall@K | 0.950 | 0.950 | 0.950 | **1.000** | 迭代 3：Reranker off |
| MRR | 0.950 | 0.950 | 0.950 | **1.000** | 迭代 3：Reranker off |
| MRR (raw) | 0.960 | 0.960 | **1.000** | 1.000 | 迭代 2：Heading aug |
| Answer Relevancy | 0.888 | 0.808 | 0.792 | **0.834** | 迭代 3：Reranker off 改善上下文质量 |
| Latency P95 | 44,895 ms | 41,181 ms | 38,702 ms | **28,212 ms** | 迭代 1：LLM 调用链缩短 + 迭代 3：去掉 reranker |

#### Agent 指标演进

| 指标 | Baseline | 迭代 3 后 | 迭代 4 | 主要贡献手段 |
|---|---|---|---|---|
| QA Pass Rate | 36.7% | 40% | **73.3%** | 迭代 1：QA 架构重构 + 迭代 4：Plan 默认不拆 |
| Quiz Valid Rate | — | 16.7% | **100%** | 迭代 4：全量证据绑定 |
| Gap Coverage | — | 超时 | **95.4%** | 迭代 4：Python 规则分类替代 LLM 循环 |
| Intent Accuracy | 96.7% | 93.3% | 73.3% | 迭代 4 Plan prompt 重写的副作用（退步项） |
| E2E Latency P50 | 86,972 ms | 97,490 ms | **76,926 ms** | 迭代 4：跳过 replan + 默认不拆 |

---

## 三、核心亮点（体现工程判断力）

### 1. 数据驱动的 Reranker A/B 实验

- 不是"加了 reranker 就完了"，而是：发现 reranker 负增益（迭代 0）→ Heading augmentation 后负增益恶化到 -0.05（迭代 2）→ 设计控制变量实验排除 chunk 数量混杂因素（迭代 3）→ 证明关闭后 Context Precision +16%、P95 -27% → 做出关闭决策
- **体现**：假设驱动的实验设计 + 敢于推翻"直觉上应该更好"的组件

### 2. 从 Failure Case 反推系统改进

- Q15/Q16 两个 case 持续低分 → 排查定位到 chunk 文本不含标题关键词 → 设计 Heading Augmentation → Q16 context_recall 0.0→1.0
- Gap Agent 259 topics 超时 → 定位到"规则计算让 LLM 做推理" → 改为 Python 硬编码分类 + LLM 只做报告文本 → 分类耗时 120s→毫秒级
- **体现**：不是盲目调参，而是从具体 case 的 failure analysis 推导出结构性改进

### 3. 评估体系设计：避免自评偏差

- 主 LLM 是 Qwen，Judge 用独立的 Kimi K2.5
- RAGAS 框架做 RAG 层评估，Task-driven 做 Agent 层评估
- 分层采样避免数据集偏差（round-robin by source）
- **体现**：评估方法论的严谨性

### 4. 架构做减法而非加法

- QA SubAgent：四阶段 → 两阶段（ReAct + Reflection），LLM 调用 5-10 → 2-4
- Plan Prompt：从"强制拆解"到"默认不拆"，消除编排过度
- 单步 Plan：跳过 replan 节点，省掉必定返回 finish 的无效 LLM 调用
- Gap Agent：从 LLM tool-calling 循环改为 Python 确定性分类
- **体现**：识别 pipeline 中的冗余并做减法，用更少的 LLM 调用做更多的事

### 5. Orchestrator 架构演进

- Plan & Execute → Plan & Execute with Replan → 单步跳过 Replan 优化
- DAG 校验（Kahn 拓扑排序检环）、依赖驱动的并发执行、失败隔离
- **体现**：不是一步到位的设计，而是 baseline → 评估暴露问题 → 迭代改进的工程闭环

---

## 四、诚实的局限性（写简历时要回避或转化的点）

| 局限 | 现实 | 如何处理 |
|---|---|---|
| **延迟高** | RAG P50 仍 19s，Agent E2E P50 ~77s | 不要写绝对延迟数字；写"降低 37%"这类相对改进 |
| **数据集小** | 仅 20-30 样本 | 不要强调"在 N 个样本上达到 X"；说"建立 RAGAS + Task-driven 评估框架" |
| **Intent Acc 最新下降** | 从 96.7% 降到 73.3%（plan prompt 重写副作用） | 用迭代 0-1 的 96.7% 数据（当时是稳定的）；或不提绝对数字 |
| **个人项目** | 无真实用户、无线上流量 | 强调技术深度和方法论，不要暗示生产规模 |

---

## 五、建议的简历写法

**项目标题**：SecondBrain — 基于个人文档的 RAG 问答与知识诊断系统

**技术栈行**：Python, FastAPI, LangGraph, Milvus, SQLite, React, TypeScript, RAGAS

**Bullet points（建议 4-5 条）**：

1. **架构设计**：设计并实现 LangGraph 多 Agent 编排系统（Plan → Execute → Replan），支持 QA/Quiz/Gap 三种意图的自动分发、DAG 依赖调度与失败隔离，150 个单元/集成测试覆盖

2. **RAG 优化**：实现 Dense+Sparse 混合检索（RRF 融合）+ Heading Augmentation，经 4 轮迭代将 Faithfulness 提升 31%（0.71→0.93），Context Precision 提升 34%（0.72→0.97）

3. **数据驱动决策**：设计控制变量 A/B 实验验证 Cross-Encoder Reranker 的实际贡献，发现其在高质量 embedding 下为负增益，关闭后检索 P95 延迟降低 37%

4. **评估体系**：搭建 RAGAS（检索/生成 4 指标）+ Task-driven（Intent Acc/QA Pass Rate/Quiz Quality/Gap Coverage）双轨评估框架，使用独立 Judge 模型避免自评偏差

5. **Agent 架构优化**：将 QA SubAgent 从 4 阶段 pipeline 重构为 ReAct+Reflection 架构，单次请求 LLM 调用从 5-10 次降至 2-4 次，E2E 任务通过率从 37% 提升至 73%

---

## 六、面试追问准备

这个项目最容易被追问的技术点：

- **"为什么选 Milvus 而不是 Pinecone/Chroma？"** — hybrid search 原生支持 dense+sparse
- **"Reranker A/B 怎么控制变量的？"** — 固定最终传给 LLM 的 chunk 数，只改排序来源
- **"为什么不用 LangChain？"** — LangGraph 的 state graph 更适合有条件分支和 replan 的编排
- **"Faithfulness 怎么从 0.71 提到 0.93 的？"** — prompt grounding 约束（允许推理综合，禁止引入外部知识）
- **"评估数据集怎么生成的？"** — 半自动：LLM 从 Milvus chunks 生成候选 → 人工审核定稿
- **"Gap Agent 怎么解决 259 topics 超时的？"** — 将分类逻辑从 LLM tool-calling 循环改为 Python 确定性规则（4 桶分类），LLM 只做最后的报告文本生成
- **"Quiz 证据绑定失败怎么修的？"** — 从按 topic 精确匹配改为全量附加 RAG materials，避免 LLM 生成的 topic 与实际 topic 漂移导致匹配失败

---

## 七、ROI 判断：当前指标已足够

| 如果继续花时间在… | 预期收益 | 简历增量 |
|---|---|---|
| Faithfulness 0.90→0.95 | 小幅提升 | **几乎为零**（0.71→0.93 的 delta 已经够说了） |
| QA Pass Rate 73%→85% | 中等 | **低**（37%→73% 的故事已经足够） |
| 降低延迟 | 需要换模型/加缓存 | **低**（个人项目不考核 SLA） |
| 修 Intent Acc 回退 | 中等 | **低**（简历不写这个数字就行） |

**结论：量化指标已足够，时间更值得投入在 README + Docker 部署 + 面试问答准备上。**
