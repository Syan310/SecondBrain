# 协作规则

## 文档参考

`ARCHITECTURE.md` 项目的总体框架和设计
`DESIGN_ITERATIONS.md` 包含对项目细节想法的迭代过程（**只追加，不读取**，纯存档用）
`TASKS.md` 包含待处理的任务和已完成的任务
**如果当前是开启的新对话，务必先将 `ARCHITECTURE.md`、`TASKS.md` 看一遍来了解项目当前状态。不要读取 `DESIGN_ITERATIONS.md`。**

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

## 文档同步规则（必须执行）

每轮对话结束前，满足以下任一条件时必须更新文档：

- **完成代码实现** → 更新 `ARCHITECTURE.md` 对应模块的接口说明或实现状态
- **确定设计决策** → 更新 `ARCHITECTURE.md` 和 `TASKS.md`；向 `DESIGN_ITERATIONS.md` **追加**条目（用 bash append，不要先读取）

`ARCHITECTURE.md` 和 `TASKS.md` 需要检查并按需更新；`DESIGN_ITERATIONS.md` 只追加不读取。
