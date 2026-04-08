---

---
# ReAct 框架

**ReAct**（Reasoning and Acting）是 Agent 最经典的工作框架，核心思想是将推理（Reasoning）与行动（Acting）交替进行，形成一个循环。它由三个模块构成：

1. **Thought（思考）模块**
基于当前的问题和已有的 Observation，推理下一步需要做什么，判断是否需要调用工具。
2. **Action（行动）模块**
接收 Thought 的决策，执行具体的工具调用（Tool Call）。
3. **Observation（观察）模块**
接收工具调用的返回结果，将其作为新的上下文传回给 Thought 模块。

Thought 模块基于 Observation 的内容判断任务是否完成：若未完成，继续循环；若已完成，则直接输出最终答案。

> ⚠️ **注意**：ReAct 框架的三个模块是 **Thought / Action / Observation**，不是 Plan/Action/Observation。"Plan 模块"是另一种架构（Plan-and-Execute）的概念，两者不要混淆。

---

# Reflexion 框架

**Reflexion** 是在 ReAct 框架的基础上加入了自我反思（Self-Reflection）能力的增强版框架。

Reflection 模块的作用：

4. 对 Observation 输出的内容进行评估，判断行动结果的质量是否满足要求
5. 生成自然语言形式的反思建议，指出当前轨迹的错误或不足
6. 将反思建议注入下一轮 Thought 的 Prompt 中，引导 Agent 修正策略

相比原始 ReAct 框架，Reflexion 的核心优势是：**对中间结果的质量做显式检查**，允许 Agent 从失败中学习并在同一任务内自我纠错，这是原始 ReAct 框架无法做到的。

---

# Plan-and-Execute 架构

**Plan-and-Execute** 是另一种常见的 Agent 规划架构，与 ReAct 的逐步决策不同，它将任务分为两个阶段：

7. **Plan 阶段**：LLM 一次性将用户的复杂任务拆解为一个有序的子任务列表（步骤计划）
8. **Execute 阶段**：按计划逐步执行每个子任务，每步执行后可选择性地重新规划（Re-Plan）

**优缺点**：

- 优点：对复杂长任务结构更清晰，减少中间跑偏
- 缺点：初始 Plan 质量强依赖 LLM 能力，中途出错时重新规划代价较高

---

# Orchestrator 架构（多 Agent 架构）

这个架构是一种典型的 **Multi-Agent** 系统，主要由以下角色构成：

- **Orchestrator（调度者）**：接收用户任务，将其拆分后分配给各个 Sub-Agent
- **Sub-Agent（子 Agent）**：并行执行各自负责的子任务
- **Critic / Reviewer（审核者）**：汇总 Sub-Agent 的输出，判断结果是否满足要求；若不满足，则将任务重新交回 Orchestrator，形成循环

适用场景：任务复杂、可并行分解、对质量有较高要求的场景。

---

# LangGraph

**LangGraph** 是目前工业界最主流的 Agent 框架之一，基于**有向图（Directed Graph）**来表达工作流，天然支持循环（cycle），适合构建复杂的 Agent 和 Multi-Agent 系统。

它由四个核心部分组成：

9. **State（状态）**
整个工作流的全局状态对象（通常是 `TypedDict`），所有节点共享并通过它进行信息交换。
10. **Node（节点）**
工作流中的执行单元，通常是一个 Python 函数或 Agent，负责读取 State 并写回更新后的 State。
11. **Edge（边）**
分为普通边（固定跳转）和**条件边（Conditional Edge）**（根据 State 动态决定下一个节点），负责控制工作流的路由逻辑。
12. **Graph（图）**
将 State、Node、Edge 组合起来的整体结构。

## LangGraph 主要实现流程

13. 定义 State（数据结构）
14. 定义 Node（函数/Agent）
15. 创建 `StateGraph` 实例，注册节点
16. 根据逻辑添加 Edge（普通边）和 Conditional Edge（条件边）
17. 设置入口点（`set_entry_point`）和结束节点
18. **编译图**（`graph.compile()`）—— 生成可执行的 `CompiledGraph`
19. **运行图**（`graph.invoke()` 或 `graph.stream()`）—— 传入初始 State 开始执行

> ⚠️ 第 6 步（编译）和第 7 步（运行）是容易遗漏的关键步骤，不执行 `compile()` 无法运行图。

---

# Function Call（工具调用）

**Function Call** 是赋予大模型调用外部工具能力的机制，是现代 Agent 的核心基础能力。

主要流程：

20. 将所有可调用的工具定义写入 tool list（JSON Schema 格式描述每个工具的名称、参数、功能）
21. 调用 LLM API 时，将 tool list 通过 `tools` 参数传入
22. LLM 判断当前问题是否需要调用工具；若需要，返回包含工具名称和参数的结构化响应（JSON 格式，而非自然语言）
23. **外部代码**解析 LLM 的响应，实际执行工具调用
24. 将工具执行结果以 `tool` 角色消息的形式拼回对话上下文
25. LLM 基于工具结果继续生成最终回答

## LangGraph 中的 Function Call 实现

在 LangGraph 中，Function Call 的典型实现方式如下：

- **Agent 节点**：调用 LLM，判断是否需要工具，输出工具调用请求
- **Tool 节点**：接收 Agent 节点的请求，执行工具，将结果返回给 Agent 节点

关于 Tool 节点的两种设计方案：

26. **独立 Tool 节点（推荐）**：为每个 Sub-Agent 配备独立的 Tool 节点，逻辑清晰，维护简单
27. **共享 Tool 节点**：所有 Sub-Agent 共用一个统一的 Tool 节点，需要额外编写路由逻辑来判断工具属于哪个 Agent 并将结果正确返回——实现复杂度更高，一般不推荐

---

# MCP（Model Context Protocol）

**MCP** 是 Anthropic 于 2024 年底提出的开放协议，用于标准化 LLM 应用与外部工具/数据源之间的连接方式，可以理解为"AI 领域的 USB 接口"。

核心思想：

- **MCP Server**：对外暴露工具（Tools）、资源（Resources）、提示模板（Prompts）的服务端
- **MCP Client**：Agent/LLM 应用作为客户端，通过统一协议调用各种 MCP Server
- 解决了过去每个工具都需要单独开发集成代码的问题，实现**工具生态的标准化**

在工程实践中，LangGraph、Claude、Cursor 等工具均已支持 MCP，是 Agent 工具调用体系的重要趋势。

---

# Agent 记忆模块

Agent 的记忆系统通常分为两类：

## 短期记忆（Short-term Memory）

- 本质是**上下文窗口（Context Window）**内的对话历史
- 每轮对话都会将完整的消息历史传入 LLM
- 受限于 Context Window 大小，无法无限增长

## 长期记忆（Long-term Memory）

- 通过**外部向量数据库**存储，不受 Context Window 限制
- 运作机制：将短期记忆中的内容整合、向量化后存入数据库
- 检索方式：用户提问时，通过混合检索（语义检索 + 关键词检索）从长期记忆中提取相关内容，拼入 Prompt

## 短期记忆的溢出处理策略

当短期记忆接近 Context Window 上限时，常见的处理方式有：

28. **滑动窗口**：只保留最近 N 轮对话，丢弃更早的内容
29. **摘要压缩**：将较旧的对话摘要后压缩存储（可写入长期记忆），保留摘要 + 最近对话
30. **重要性过滤**：按重要性评分保留关键消息

---

# 长期记忆管理：Memory Consolidation（记忆整合）

在一些高级 Agent 系统（如 **MemGPT / Letta**）中，存在专门的后台记忆整合机制，会定期对长期记忆库执行以下操作：

31. **聚类（Clustering）**：将语义相近的记忆归为同一类
32. **去重（Deduplication）**：删除冗余或高度重复的记忆条目
33. **提炼（Abstraction）**：将具体的情景记忆（Episodic Memory）提炼为更通用的语义记忆（Semantic Memory）
34. **矛盾消解（Conflict Resolution）**：对相互矛盾的记忆进行判断，保留最准确的版本
35. **归档（Archiving）**：将低优先级或长期未访问的记忆转移到冷存储

> 📌 此机制来源于 MemGPT/Letta 的设计，并非所有 Agent 框架的通用标准功能，实际使用时需根据框架选择。

---

# LangGraph 深入：State Reducer 机制

LangGraph 的 State 并非简单赋值覆盖，而是支持**自定义 Reducer**，用于控制多个节点对同一字段并发写入时的合并行为。

## 基本用法

```python
from typing import Annotated
import operator
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[list, operator.add]  # 追加模式，不覆盖
    result: str  # 普通覆盖模式
```

- `operator.add`：新值追加到列表末尾，不覆盖原有内容——多用于 `messages` 字段
- 不加 `Annotated`：直接覆盖，后写入的值覆盖先写入的值

## 为什么在 Multi-Agent 中很重要

在 Orchestrator 架构中，多个 Sub-Agent 并行运行并写回同一个 State。如果没有 Reducer，后完成的 Agent 会直接覆盖先完成的 Agent 的结果，导致数据丢失。通过 `operator.add` 这类 Reducer，可以安全地将多个 Sub-Agent 的输出合并到同一个列表字段中。

---

# LangGraph 深入：条件边的实现

条件边（Conditional Edge）是 LangGraph 实现动态路由的核心机制。

## 典型用法

```python
def should_continue(state: State) -> str:
    if state["messages"][-1].tool_calls:  # LLM 请求了工具调用
        return "tools"
    return END  # 不需要工具，直接结束

graph.add_conditional_edges(
    "agent",           # 从哪个节点出发
    should_continue,   # 路由函数，返回下一节点的名称
    {"tools": "tools", END: END}  # 映射表（可选）
)
```

- 路由函数接收当前 State，返回字符串（下一个节点名）
- `END` 是 LangGraph 内置的终止节点标识，表示工作流结束
- 映射表可以将路由函数的返回值映射为节点名，提高可读性

---

# LangGraph 深入：Checkpointer 与持久化

**Checkpointer** 是 LangGraph 的持久化机制，用于在每一步执行后保存 State 快照，实现以下能力：

- **跨会话记忆**：同一用户下次对话时恢复上次的 State，无需重新开始
- **Human-in-the-Loop**：执行到某一步时暂停，等待人工确认后继续
- **错误恢复**：从最后一个成功的 Checkpoint 重新执行，而非从头开始
- **时间旅行（Time Travel）**：回退到历史某个 State，重新分叉执行

## 常用 Checkpointer

| Checkpointer | 存储方式 | 适用场景 |
| --- | --- | --- |
| `MemorySaver` | 内存（进程内） | 开发调试、单次会话 |
| `SqliteSaver` | SQLite 文件 | 本地持久化、轻量生产 |
| `PostgresSaver` | PostgreSQL | 生产环境、多实例部署 |

## 使用方式

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# 每次调用传入 thread_id，区分不同会话
config = {"configurable": {"thread_id": "user_123"}}
app.invoke({"messages": [...]}, config=config)
```

---

# Human-in-the-Loop（人工介入）

**Human-in-the-Loop** 是 Agent 工程实践中的重要安全机制，允许在工作流执行到关键节点时暂停，等待人工确认或修改后再继续。

## 在 LangGraph 中的实现

通过 `interrupt_before` 或 `interrupt_after` 在 `compile()` 时指定需要暂停的节点：

```python
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"]  # 在执行 tools 节点前暂停
)

# 第一次运行：执行到 tools 节点前自动暂停
app.invoke(input, config)

# 人工审查 State，确认后恢复执行（传入 None 表示继续）
app.invoke(None, config)
```

## 典型使用场景

- 执行危险操作前（发送邮件、删除文件、调用付费 API）需要人工确认
- Agent 给出中间规划，人工审查计划后再执行
- 需要人工补充信息才能继续的场景

---

# Multi-Agent 通信方式

在多 Agent 系统中，Sub-Agent 之间的通信主要有三种模式：

## 1. 共享 State（Shared State）

所有 Agent 共用同一个 LangGraph State 对象，通过读写 State 间接通信。

- **优点**：实现简单，天然适合 LangGraph 架构
- **缺点**：State 字段设计需仔细规划，字段过多容易混乱
- **适用**：同一图内的 Agent 协作

## 2. Handoff（控制权移交）

Agent A 完成自己的任务后，通过返回特定的 Tool Call 或消息，将控制权显式移交给 Agent B。

- **优点**：职责边界清晰，适合流水线式任务
- **缺点**：不支持并行，只能串行传递
- **适用**：有明确先后顺序的子任务

## 3. 消息传递（Message Passing）

Agent 之间通过消息队列或专用通信通道传递信息，彼此解耦。

- **优点**：解耦程度高，可跨进程/服务部署
- **缺点**：实现复杂度较高
- **适用**：分布式、大规模 Multi-Agent 系统

---

# Tool 设计规范

Function Call 中，LLM 完全依赖工具的**文字描述**来决定是否调用以及如何调用该工具。工具描述的质量直接影响 Agent 的行为准确性。

## 高质量 Tool Description 的要素

36. **明确说明功能**：工具做什么，不做什么
37. **说明输入参数的含义和格式**：尤其是可选参数的默认行为
38. **说明返回值的结构**：LLM 需要知道结果长什么样
39. **给出典型使用场景**：帮助 LLM 判断何时应该调用
40. **说明限制和边界条件**：避免 LLM 在不该调用时调用

## 示例对比

❌ **差的描述**：`搜索工具，用于搜索信息`

✅ **好的描述**：`在公司内部知识库中搜索技术文档。输入关键词（字符串），返回最相关的 5 篇文档标题和摘要。仅用于查询内部文档，不适用于搜索互联网信息。`

---

# Agent 的 Prompt Engineering

Agent 的 System Prompt 设计与普通 LLM 的 Prompt 有所不同，需要覆盖以下内容：

## System Prompt 的核心组成

41. **角色定义（Role）**：明确 Agent 是谁、具备什么能力、负责什么任务
42. **工具使用说明（Tool Instructions）**：何时调用工具、调用哪个工具、工具调用的优先级
43. **行为约束（Constraints）**：不能做什么、遇到歧义时如何处理
44. **输出格式要求（Output Format）**：最终答案的格式规范
45. **Few-shot 示例（可选）**：提供 1-3 个完整的 Thought/Action/Observation 示例，显著提升复杂任务的准确率

## 关键注意点

- Agent 的 System Prompt 通常比普通 LLM 的更长、更结构化
- 工具描述和 System Prompt 共同决定工具选择行为，两者要保持一致
- 在 Multi-Agent 系统中，每个 Sub-Agent 都应有独立的、针对其职责定制的 System Prompt

---

# Agent 评估体系

衡量 Agent 质量的指标体系与普通 LLM 不同，需要关注**过程（轨迹）**和**结果**两个维度。

## 结果评估指标

| 指标 | 说明 |
| --- | --- |
| **任务成功率（Task Success Rate）** | Agent 完全正确完成任务的比例，最核心的指标 |
| **部分成功率** | 任务完成度评分（0-1），适用于复杂任务 |
| **答案准确率** | 最终输出与 Ground Truth 的匹配程度 |

## 轨迹评估指标（Trajectory Evaluation）

| 指标 | 说明 |
| --- | --- |
| **工具调用准确率** | 是否调用了正确的工具（对的工具、对的参数） |
| **冗余步骤数** | Agent 绕路、重复调用的次数，越少越好 |
| **幻觉工具调用率** | 调用了不存在或不相关工具的比例 |
| **平均步骤数** | 完成任务所需的平均轮次，衡量效率 |

## 常用评估框架

- **LangSmith**：LangChain 官方的 Agent 追踪与评估平台，可记录完整轨迹
- **RAGAS**：原用于 RAG 评估，部分指标可迁移到 Agentic RAG 场景
- **自定义 LLM-as-Judge**：用另一个 LLM 对 Agent 的轨迹进行打分评估

---

# Agent 常见失败模式

初级面试中高频考察：了解 Agent 在哪里容易出问题，体现工程经验。

## 1. 幻觉工具调用（Hallucinated Tool Call）

**现象**：LLM 调用了一个不存在的工具，或传入了错误格式的参数。

**原因**：工具描述不够清晰，或 LLM 过度自信。

**解法**：完善工具描述；对工具调用结果做参数校验；使用 strict 模式的 JSON Schema。

## 2. 无限循环（Infinite Loop）

**现象**：Agent 在 Thought → Action → Observation 循环中无法跳出，持续重复调用工具。

**原因**：终止条件设计不合理，或 Observation 结果始终未满足 LLM 的预期。

**解法**：设置**最大步骤数限制（max_iterations）**；设计明确的终止信号；添加重复检测逻辑。

## 3. 上下文爆炸（Context Overflow）

**现象**：随着工具调用轮次增加，消息历史超出 Context Window，导致报错或信息截断。

**原因**：未对 message history 做长度管理。

**解法**：使用滑动窗口或摘要压缩策略；将中间结果存入长期记忆而非全部保留在上下文中。

## 4. 工具选择错误（Wrong Tool Selection）

**现象**：面对某类问题，LLM 持续选择错误的工具。

**原因**：工具描述相似、歧义，或 Few-shot 示例不足。

**解法**：优化工具描述，明确区分各工具的适用边界；增加 Few-shot 示例。

## 5. 过度工具调用（Over-Calling）

**现象**：对于可以直接回答的问题，Agent 仍然调用工具，浪费资源。

**原因**：System Prompt 未明确「什么时候不需要调用工具」。

**解法**：在 System Prompt 中显式说明直接回答的条件。

---

# Parallel Tool Call（并行工具调用）

部分 LLM（如 GPT-4o、Claude 3.5）支持在一次响应中同时返回**多个工具调用请求**，而非逐个串行执行。

## 工作流程

46. LLM 判断当前步骤可以并行执行多个工具
47. 在同一次响应中返回多个 `tool_call` 对象
48. 外部代码并行执行所有工具
49. 将所有结果一次性拼回上下文
50. LLM 基于所有结果继续生成

## 优势与注意点

- **优势**：显著减少总 RTT（Round-Trip Time），加快 Agent 执行速度
- **注意**：并行工具之间不能有数据依赖（工具 B 的输入不能依赖工具 A 的输出）；需要在代码层面实现并发执行逻辑（如 `asyncio.gather`）