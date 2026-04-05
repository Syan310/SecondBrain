# Agent 分层架构

## 概要

Orchestrator 采用 Plan & Execute with Replan 模式，调度三个自治 SubAgent：QA（RAG问答）、Quiz（自适应出题）、Gap Analysis（知识盲区诊断）。单结果直传，多结果 LLM 汇总。

---

## 详细设计

```
Orchestrator (Plan & Execute with Replan, LangGraph StateGraph)
  │  LLM few-shot 意图识别 → 生成执行计划 → 单步执行 → Replan(continue/revise/finish) → 汇总结果
  │
  ├── QA SubAgent (被 Orchestrator 调度)
  │     入口检查 Milvus 文档数:
  │     ├── 知识库为空 → 跳过 RAG, 纯 LLM 通用问答 + 提示"知识库为空"
  │     └── 有文档 → 自行判断问题复杂度:
  │           ├── 简单问题 → 单次 Retrieve + Rerank → Generate
  │           └── 复杂问题 → Decompose → 多轮 Retrieve + Rerank → Synthesize
  │           自行判断结果是否充分, 不充分则追加检索
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
