# LangGraph 图结构详细设计

## 概要

顶层 StateGraph：plan_node → execute_step → replan_node 循环，max_replan=3。QA SubAgent 内含 Complexity Judge + 简单/复杂双路径。Quiz 为单次出2题 step。Gap 为 Load→Search→Compare→Report 线性流程。

---

## 详细设计

### 节点一览

| 节点 ID | 所属 | 类型 | 说明 |
|---------|------|------|------|
| `plan_node` | 顶层 | 规划 | LLM few-shot 意图识别 + 执行计划生成 |
| `execute_step` | 顶层 | 执行 | 单步执行 plan[current_step_index]，调度 SubAgent |
| `replan_node` | 顶层 | 决策 | LLM 审视 step 结果，决定 continue/revise/finish |
| `synthesize_node` | 顶层 | 汇总 | 多结果 LLM 合并为最终回答 |
| `Complexity Judge` | QA | 判定 | 判断问题是否需要多跳分解 |
| `RAG Search` | QA | 处理 | 向量检索 Top-K |
| `Reranker` | QA | 处理 | bge-reranker-v2-m3 重排 |
| `Generate Answer` | QA | 处理 | 基于检索结果生成答案 |
| `Decompose Query` | QA | 处理 | 将复杂问题拆成子查询 |
| `Synthesize` | QA | 处理 | 综合多次检索结果生成最终答案 |
| `Sufficiency Check` | QA | 判定 | 判断当前结果是否足够回答原始问题 |
| `Load Memory` | Quiz | 处理 | 读 SQLite 掌握度 + 时间衰减计算 |
| `Select Question` | Quiz | 处理 | weight = 1 - decayed_mastery，加权随机选题 |
| `Evaluate Answer` | Quiz | 处理 | LLM 语义判分 |
| `Quiz Router` | Quiz | 判定 | 答错同topic再出 / 答对继续 / 结束 |
| `Update Memory` | Quiz | 处理 | 更新 mastery_score + last_reviewed |
| `Load Mastery` | Gap | 处理 | 读取指定范围的 mastery 数据 |
| `Search Concepts` | Gap | 处理 | 检索知识库中相关概念列表 |
| `Cross Compare` | Gap | 处理 | 交叉对比，识别盲区 |
| `Generate Report` | Gap | 处理 | 生成诊断报告 + 推荐复习顺序 |

### 顶层流程 (Plan & Execute with Replan)

```
START → plan_node (LLM few-shot 意图识别+计划生成)
      → route_after_plan
          ├── no_steps → empty_response → END
          └── has_steps → execute_step (执行 plan[current_step_index] 单个步骤)
                → replan_node (LLM 审视结果, 决定下一步)
                    ├── continue → execute_step (执行下一个 planned step)
                    ├── revise   → execute_step (替换剩余 plan 后继续, 可追加/删减/修改步骤)
                    └── finish   → route_to_output
                                     ├── single → passthrough_node (直传结果) → END
                                     └── multi  → synthesize_node (LLM 汇总) → END

Replan 安全机制: max_replan_count=3, 超过强制 finish

多意图编排示例:
  "分析NLP盲区然后讲讲最弱的" → plan: [gap(NLP), qa(depends_on gap结果)]
    → exec gap → replan(continue) → exec qa(params rewritten) → replan(finish) → synthesize
  "分析DL盲区然后讲讲" → plan: [gap(DL)]
    → exec gap → replan(revise, 发现3个盲区追加3个qa step) → exec qa1 → replan(continue) → ...
  "来几道NLP题" → plan: [quiz(NLP)] → exec quiz → replan(finish) → passthrough
```

### QA SubAgent 内部流程

```
入口 → 检查 Milvus 文档数
  ├── 知识库为空 → 纯 LLM 通用问答 (提示知识库为空) → 返回 Orchestrator
  └── 有文档 → Complexity Judge
        ├── 简单 → RAG Search → Reranker → Generate Answer → Sufficiency Check
        │                                                       ├── 充分 → 返回 Orchestrator
        │                                                       └── 不充分 → RAG Search (追加检索)
        └── 复杂 → Decompose Query → [对每个子查询: RAG Search → Reranker] → Synthesize → 返回 Orchestrator
```

### Quiz SubAgent 内部流程

```
入口 → Load Memory → Select Question → (等待用户作答) → Evaluate Answer → Quiz Router
  ├── 答错 → 同 topic 再出 → Select Question
  ├── 答对且未结束 → Select Question
  └── 答对且结束 / 用户主动结束 → Update Memory → 返回 Orchestrator
```

### Gap Analysis SubAgent 内部流程

```
入口 → Load Mastery → Search Concepts → Cross Compare → Generate Report → 返回 Orchestrator
```
