# 协作规则

## 项目快照

SecondBrain：基于个人笔记的 RAG + Quiz + 知识盲区诊断系统。FastAPI + LangGraph + Milvus + React/Vite。
Step 1-7 已完成（Services → Ingestion → Tools → SubAgents → Orchestrator → Memory → API → Frontend），当前在 Step 8 评估 & 优化阶段。

## 文档参考

- `ARCHITECTURE.md` — 系统架构总览 + 文档索引（**新对话必读**）
- `TASKS.md` — 任务清单（**新对话必读**）
- `docs/*.md` — 各模块详细设计（**按需查阅**，见 ARCHITECTURE.md 索引表）
- `DESIGN_ITERATIONS.md` — 设计迭代存档（**只追加，不读取**）

**新对话启动时：读 `ARCHITECTURE.md` 和 `TASKS.md`，不要读 `docs/` 下的文件，除非任务涉及对应模块。**

## 核心原则

我来提供设计思路和实现逻辑，你负责将我的思路翻译成代码。
禁止在我没有提供实现细节的情况下自行补全核心逻辑。
**如果不确定某个模块是否需要等我说清楚，默认等待，主动向我提问。**

## 你必须等我说清楚再动手的模块（包括但不限于）：

- LangGraph状态图结构和条件边路由
- Intent Router的意图分类逻辑和分支策略
- Quiz自适应出题的权重计算和路由规则
- Flashcard QC的判断标准和重生成策略
- Memory更新规则（mastery分数的奖惩逻辑）

## 对于以上模块，收到需求后你应该

1. 告诉我你理解到的需求是什么
2. 列出你需要我补充的实现细节
3. 等我确认后再写代码

## 可以直接生成的模块（包括但不限于）：

- Notion API调用封装和数据解析
- Milvus连接和Collection初始化
- SQLite建表和读写封装
- FastAPI路由和Schema
- Gradio/React UI布局
- 日志、异常处理等工程胶水代码

## 每次写代码前

用一句话复述我给你的实现逻辑，在我确认之后确保理解一致再开始写。

## 测试规则

每完成一个大型模块（如一个 service、一个 tool、一个 SubAgent）后，必须编写并运行测试，确认模块功能正常后再继续下一个模块。
每次测试完毕后，将测试结果和遇到的问题/修复方案更新到 `backend/TESTING.md`。

## 文档同步规则（必须执行）

每轮对话结束前，满足以下任一条件时必须更新文档：

- **完成代码实现** → 更新 `ARCHITECTURE.md` 或 `docs/` 对应模块文档
- **确定设计决策** → 更新 `ARCHITECTURE.md` 和 `TASKS.md`；向 `DESIGN_ITERATIONS.md` **追加**条目（用 bash append，不要先读取）

`ARCHITECTURE.md` 和 `TASKS.md` 需要检查并按需更新；`DESIGN_ITERATIONS.md` 只追加不读取。
