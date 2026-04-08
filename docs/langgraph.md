# LangGraph 图结构详细设计

## 概要

顶层 StateGraph：plan_node → execute_batch → replan_node 循环，max_replan=3。plan_node 负责意图识别和 QA 子查询拆解（对比/关联类拆多步），生成的 plan 经 `_validate_plan` 结构校验（编号唯一、依赖合法、Kahn 检环、params schema）后进入执行。execute_batch 并发调度同批 ready steps，全局 Semaphore(2) 跨请求限流，单步失败隔离（status=ok/failed）。QA SubAgent 为 ReAct (max 2轮) + Reflection，专注单概念检索回答。Quiz 为单次出2题 step。Gap 为 Agent 驱动诊断。

---

## 详细设计

### 节点一览

| 节点 ID | 所属 | 类型 | 说明 |
|---------|------|------|------|
| `plan_node` | 顶层 | 规划 | LLM few-shot 意图识别 + 执行计划生成 + QA 子查询拆解 + `_validate_plan` 结构校验 |
| `execute_batch` | 顶层 | 执行 | 并发执行所有 deps 已满足的 steps, 全局 `Semaphore(2)` 限流, 单步失败隔离 |
| `replan_node` | 顶层 | 决策 | LLM 审视 step 结果（含失败信息），决定 continue/revise/finish; revised_steps 走完整结构校验 |
| `synthesize_node` | 顶层 | 汇总 | 多结果 LLM 合并为最终回答（仅取 status=ok 的结果） |
| `ReAct LLM` | QA | 处理 | LLM + rag_search tool-calling，max 2 轮 |
| `Reflection` | QA | 判定 | 审查回答质量，pass/fail + feedback |
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
START → plan_node (LLM few-shot 意图识别+计划生成+QA子查询拆解)
      → _validate_plan (编号唯一/依赖合法/Kahn检环/params schema, 不合法→单步QA fallback)
      → route_after_plan
          ├── no_steps → empty_response → END
          └── has_steps → execute_batch (并发执行所有 deps 已满足的 steps)
                             全局 Semaphore(2) 跨请求限流
                             单步 try/except → status=ok/failed, 失败不影响同批
                → replan_node (LLM 审视结果+失败信息, 决定下一步)
                    ├── continue → execute_batch (下一批 ready steps)
                    ├── revise   → _is_plan_structure_valid + params校验
                    │                → 通过: 替换剩余 plan 后继续
                    │                → 不通过: 降级为 finish
                    └── finish   → route_to_output (仅计 status=ok 的结果)
                                     ├── single → passthrough_node (直传; 全失败时展示错误) → END
                                     └── multi  → synthesize_node (LLM 汇总 ok 结果) → END

安全机制:
  - max_replan_count=3, 超过强制 finish
  - 全局 Semaphore(2) 防下游资源打满
  - plan/revised_steps 结构校验 (编号/依赖/环/params)
  - 单步失败隔离, replan 可见失败信息做决策

QA 拆解示例:
  "self-attention 和 cross-attention 的区别" → plan: [qa(self-attention), qa(cross-attention)]
    → execute_batch(两步并行) → replan(finish) → synthesize
  "Token Embedding 缩放与 LayerNorm 的协同" → plan: [qa(缩放原因), qa(协同关系, depends_on step1)]
    → execute_batch(step1) → replan(continue) → execute_batch(step2, params rewritten) → replan(finish) → synthesize

多意图编排示例:
  "分析NLP盲区然后讲讲最弱的" → plan: [gap(NLP), qa(depends_on gap结果)]
    → execute_batch(gap) → replan(continue) → execute_batch(qa, params rewritten) → replan(finish) → synthesize
  "来几道NLP题" → plan: [quiz(NLP)] → execute_batch(quiz) → replan(finish) → passthrough
```

### QA SubAgent 内部流程

```
入口 → 检查 Milvus 文档数
  ├── 知识库为空 → 纯 LLM 通用问答 (提示知识库为空) → 返回 Orchestrator
  └── 有文档 → ReAct Loop (max 2 轮):
        LLM 分析问题 → 调用 rag_search → 观察结果 → 生成回答
        (如第1轮检索不足, 第2轮可换 query 补充检索)
              ↓
        Reflection:
          审查回答质量 (完整性/准确性/相关性)
          ├── pass → 返回 Orchestrator
          └── fail → 带 feedback 再跑 1 轮 ReAct → 返回 Orchestrator
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
