# Agent 分层架构

## 概要

Orchestrator 采用 Plan & Execute with Replan 模式，调度三个自治 SubAgent：QA（RAG问答）、Quiz（自适应出题）、Gap Analysis（知识盲区诊断）。单结果直传，多结果 LLM 汇总。Orchestrator plan_node 负责将用户查询拆解为原子任务序列（每个 step 对应单一检索意图、单一概念、可独立执行），QA SubAgent 专注单概念检索回答。

---

## 详细设计

```
Orchestrator (Plan & Execute with Replan, LangGraph StateGraph)
  │  原子任务拆解(三步流程: 识别意图→检查可拆性→确定依赖) → 单步执行 → Replan(continue/revise/finish) → 汇总结果
  │  拆解粒度：每个 step 满足单一检索意图 + 单一概念聚焦 + 独立可回答
  │
  ├── QA SubAgent (被 Orchestrator 调度, 每次只处理单概念查询)
  │     入口检查 Milvus 文档数:
  │     ├── 知识库为空 → 跳过 RAG, 纯 LLM 通用问答 + 提示"知识库为空"
  │     └── 有文档 → ReAct (max 2轮) + Reflection:
  │           ReAct: LLM 通过 tool-calling 调用 rag_search → 基于检索结果生成回答
  │           Reflection: 审查回答质量 → pass 返回 / fail 带反馈再跑 1 轮 ReAct
  │
  ├── Gap Analysis SubAgent (被 Orchestrator 调度)
  │     读取 mastery 数据 → 检索知识库概念列表 →
  │     交叉对比(有知识但mastery低/无记录 = 盲区) → 生成诊断报告
  │
  └── Synthesizer (多 SubAgent 结果汇总, 单结果时跳过)

独立调用 (不经过 Orchestrator, API 层直接调用):
  └── Quiz SubAgent (多轮交互, 独立生命周期)
        Load Memory(含时间衰减) → 加权选题 → 等待作答 →
        LLM判分 → 路由(答错同topic再出/答对继续/结束) → Update Memory
```
