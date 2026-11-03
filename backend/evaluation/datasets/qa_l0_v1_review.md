# 题集 qa_l0_v1 核对表

- 状态：**已核对**（2026-11-02，总 Agent 逐题核对，用户确认）。40 道新题全部保留，不需要修改或补题；20 道原题的 gold_sections 全部正确
- 起草人：Agent B（Claude），没有调用任何模型 API
- 语料：`C:\Users\Shaoyu Yan\Desktop\LLM_Learn` 下的 4 份笔记
- `heading_path`：用 `DocProcessor().process_file()` 解析笔记得到，和系统切块一致

## 怎么核对

1. 逐题看第 1 节的 40 道新题。
2. 在最后一列填 **保留**、**修改** 或 **删除**。
3. 填「修改」时，在同一格写明改成什么。
4. 第 2 节是 20 道原题的 `gold_sections`。原题的问题和答案不改，只请确认章节对不对。

你删掉的题，我按同一题型补齐，再请你核对一次。核对完以后我生成 `qa_l0_v1.json`。

## 0. 题目组成

| 题型 | `type` | 题数 |
|---|---|---|
| 术语 | `term` | 10 |
| 换说法 | `paraphrase` | 10 |
| 标题敏感 | `heading` | 5 |
| 单文档事实 | `factual` | 5 |
| 拒答 | `refusal` | 10 |
| 原题 | `original` | 20 |
| **合计** | | **60** |

新增的 30 道可回答题在 4 份笔记上的分布（要求每份至少 6 道）：

| 笔记 | 新题数 | 原题数 |
|---|---|---|
| Transformer 复习.md | 7 | 5 |
| RAG复习.md | 9 | 5 |
| Agent 复习.md | 7 | 5 |
| 你不知道的Agent-原理架构与工程实践.md | 7 | 5 |

### 程序检查的结果

下面 5 项都由脚本检查。40 道题全部通过，0 个问题。

| 检查 | 规则 | 结果 |
|---|---|---|
| 章节存在 | 每个 `gold_sections` 的 `heading_path` 都在 `DocProcessor` 的输出里 | 30/30 通过 |
| 答案有出处 | 参考答案的关键原文出现在 `gold_sections` 的正文里 | 30/30 通过 |
| 换说法题 | 问题和对应章节正文没有共同的 3 字以上中文片段，也没有共同的英文词（排除下面列出的疑问词） | 10/10 通过 |
| 标题敏感题 | 关键词出现在 `heading_path` 里，不出现在这个章节的正文里 | 5/5 通过 |
| 拒答题 | 关键词在 4 份笔记里都搜不到（RoPE、MoE、GQA、ToT 区分大小写，其余不区分） | 10/10 通过 |

**换说法检查排除的词：** 什么、为什么、是什么、哪些、哪三、分别、如何、怎样、Agent。
前 8 个是疑问词和虚词，不是检索关键词。「Agent」在 4 份笔记里到处都有，起不到区分作用。
排除这些词以后，有 2 道题和正文共用了被排除的词：

- q033 共用「为什么」「什么」（正文有「为什么 HyDE 能起作用？」）
- q036 共用「分别」

其他 8 道换说法题即使不排除这些词，也能通过检查。2 字的普通词（如「模型」「问题」）不在检查范围内。

> **注意：** 系统会把代码块里的 `# 注释` 当成 Markdown 标题，所以有些 `heading_path` 看起来很奇怪。
> 例如 q049 的章节是 `人工审查 State，确认后恢复执行（传入 None 表示继续）/典型使用场景`。
> 这是系统实际的切块结果，题集按它来写。原题 q007、q008 也是这种情况。

## 1. 新题（q021–q060）

| 编号 | 题型 | 问题 | 参考答案 | gold_sections | 程序检查用的关键词 | 核对（保留 / 修改 / 删除） |
|---|---|---|---|---|---|---|
| q021 | 术语 | BPE 分词器是按照什么流程构建词表的？ | 先把词表中的所有词拆成最小的字符单元；统计每两个相邻字符对的出现频率；把频率最高的字符对合并为一个新 token 加入词表；重复统计和合并，直到词表达到预设大小。 | `Transformer` › 1. BPE 分词器 | `BPE` || 保留 |
| q022 | 术语 | Encoder-Decoder 架构的 Cross-Attention 中，Q、K、V 分别来自哪里？ | Q 来自 Decoder 的隐层输出，K 和 V 来自 Encoder 的最终输出。 | `Transformer` › 9. 三种架构：Encoder、Decoder、Encoder-Decoder/Encoder-Decoder 的 Cross-Attention | `Cross-Attention` || 保留 |
| q023 | 术语 | IVF 索引中把 nprobe 调大会带来什么影响？ | nprobe 越大，搜索的簇越多，召回率越高，但速度越慢；nprobe 等于 nlist 时退化为暴力搜索。 | `RAG` › 存入数据库的 document 不需要加前缀（或加不同前缀）/IVF 原理详解 | `IVF`、`nprobe` || 保留 |
| q024 | 术语 | HNSW 的 ef_search 参数控制什么？调整它需要重建索引吗？ | ef_search 是检索阶段的动态候选集大小，控制检索精度；可以在不重建索引的情况下动态调整。 | `RAG` › 存入数据库的 document 不需要加前缀（或加不同前缀）/HNSW 原理详解 | `HNSW`、`ef_search` || 保留 |
| q025 | 术语 | 混合检索用 RRF（Reciprocal Rank Fusion）融合时，公式是什么？k 通常取多少？ | RRF 不依赖原始分数，而是按各检索结果在各自列表中的排名融合，公式为 score = Σ 1/(k + rank_i)，k 通常为 60。 | `RAG` › 检索环节/混合检索（Hybrid Search） | `RRF` || 保留 |
| q026 | 术语 | LangGraph 的 MemorySaver、SqliteSaver 和 PostgresSaver 分别适合什么场景？ | MemorySaver 存在进程内存里，适合开发调试和单次会话；SqliteSaver 存在 SQLite 文件里，适合本地持久化和轻量生产；PostgresSaver 存在 PostgreSQL 里，适合生产环境和多实例部署。 | `Agent 复习` › LangGraph 深入：Checkpointer 与持久化/常用 Checkpointer | `MemorySaver`、`SqliteSaver`、`PostgresSaver` || 保留 |
| q027 | 术语 | 在 LangGraph 的 State 定义里，Annotated[list, operator.add] 有什么作用？ | 它是追加模式：新值追加到列表末尾，不覆盖原有内容，多用于 messages 字段；不加 Annotated 时，后写入的值直接覆盖先写入的值。 | `Agent 复习` › LangGraph 深入：State Reducer 机制/基本用法 | `operator.add`、`Annotated` || 保留 |
| q028 | 术语 | MCP 中的 MCP Server 对外暴露哪三类内容？ | 工具（Tools）、资源（Resources）和提示模板（Prompts）。 | `Agent 复习` › MCP（Model Context Protocol） | `MCP Server` || 保留 |
| q029 | 术语 | Agent 评测中，Pass@k 和 Pass^k 分别适合什么阶段？ | Pass@k 适合开发阶段，用来回答能力边界；Pass^k 适合上线前，用来保证质量。 | `你不知道的Agent` › 你不知道的 Agent：原理、架构与工程实践/8. Agent 评测 | `Pass@k`、`Pass^k` || 保留 |
| q030 | 术语 | Cursor 在 MCP 工具上验证 Dynamic Context Discovery 时，A/B 测试的结果是什么？ | Cursor 把工具描述同步到文件夹，Agent 默认只看到工具名，需要时再查询具体定义；A/B 测试中，调用 MCP 工具的任务总 token 消耗减少了 46.9%。 | `你不知道的Agent` › 你不知道的 Agent：原理、架构与工程实践/文件系统为什么适合做上下文接口 | `Dynamic Context Discovery`、`Cursor` || 保留 |
| q031 | 换说法 | 词向量在送进模型之前，为什么要先放大一个跟模型宽度有关的倍数？ | Embedding 向量会乘以 √d_model（模型维度的平方根），目的是确保 Embedding 的数值尺度不会被位置编码淹没。 | `Transformer` › 2. Token Embedding | — || 保留 |
| q032 | 换说法 | 让每一层的原始信息绕过该层、原样并入它的结果，这种设计主要为了解决什么训练难题？ | 这是残差连接，核心目的是缓解深层网络中的梯度消失问题：把子层的输入直接加到输出上（output = x + sublayer(x)），即使某一层学得不好，也能保留原始输入的信息，保证梯度顺利回传。 | `Transformer` › 7. 残差连接（Residual Connection） | — || 保留 |
| q033 | 换说法 | 提问只有一句话、资料却是大段说明时，先让模型凭空写一份设想的回复，再拿这份回复去查找资料，为什么能找得更准？ | 这是 HyDE。短问题的向量与长文档的向量天然存在语义差距；先让 LLM 生成一段假设性回答，这段回答的向量与数据库中真实答案的向量更接近，所以检索效果更好。 | `RAG` › Query-Doc 语义鸿沟/HyDE 的完整原理与适用场景 | — || 保留 |
| q034 | 换说法 | 在把每个文本片段写入库之前，先让模型给它补一两句交代来龙去脉的话，这样做能把找不到资料的情况减少多少？ | 这是 Contextual Retrieval。Anthropic 的测试数据显示，它配合 BM25 可以把检索失败率降低约 49%。 | `RAG` › Contextual Retrieval（上下文检索）/效果 | — || 保留 |
| q035 | 换说法 | 用户换一种说法问同一件事时，只按原文逐字比对来复用旧结果的机制为什么会失灵？应该改成什么做法？ | 普通缓存（如 Redis）基于精确字符串匹配，语义相同但表述不同的问题无法命中。Semantic Cache 改为按向量相似度匹配：新问题与缓存中某个问题的相似度高于阈值时，直接返回缓存结果，跳过检索和 LLM 调用。 | `RAG` › 并行对多个子问题检索/语义缓存（Semantic Cache） | — || 保留 |
| q036 | 换说法 | 把同一份资料分别按不同粒度切开、各建一套库，怎样比较出哪种粒度最好？ | 给定几个不同量级的块大小（如 128、256、512 token），按每个量级分别切块并存入对应的向量数据库，针对用户提问生成回复，再用「忠诚度测试」和「相关性测试」给不同块大小的回复打分，选出最优量级。 | `RAG` › RAG 分块（Chunking）/切块大小如何选择？ | — || 保留 |
| q037 | 换说法 | 先让模型把整件复杂工作排出完整的步骤单，然后照单逐项完成，这种智能体设计的长处和短处各是什么？ | 这是 Plan-and-Execute 架构。优点是对复杂长任务结构更清晰，减少中间跑偏；缺点是初始计划的质量强依赖 LLM 能力，中途出错时重新规划的代价较高。 | `Agent 复习` › Plan-and-Execute 架构 | — || 保留 |
| q038 | 换说法 | 几个子智能体并发干完活后，各自把产出存进一个共用字段，如果不做合并设置会出什么问题？ | 后完成的 Agent 会直接覆盖先完成的 Agent 的结果，导致数据丢失。用 operator.add 这类 Reducer，可以把多个 Sub-Agent 的输出安全地合并到同一个列表字段中。 | `Agent 复习` › LangGraph 深入：State Reducer 机制/为什么在 Multi-Agent 中很重要 | — || 保留 |
| q039 | 换说法 | 为什么一份很少改动、篇幅很长的开场指令，反而可能比经常改动的短指令更省钱？ | Prompt Caching 要求输入前缀精确一致才能命中。稳定的大系统提示写入缓存的成本只付一次，后续每次调用读取可以打到约 90% 的折扣；经常改动的提示会破坏前缀匹配，无法命中缓存。 | `你不知道的Agent` › 你不知道的 Agent：原理、架构与工程实践/Prompt Caching 减少重复开销 | — || 保留 |
| q040 | 换说法 | 搭建大模型应用时，有哪几种把模型调用串起来、分流或同时跑的基础套路？ | 五种：提示链（Prompt Chaining）、路由（Routing）、并行（Parallelization）、编排器-工作者（Orchestrator-Workers）、评估器-优化器（Evaluator-Optimizer）。 | `你不知道的Agent` › 你不知道的 Agent：原理、架构与工程实践/五种常见控制模式 | — || 保留 |
| q041 | 标题敏感 | Transformer 为什么需要 LayerNorm？ | 如果不做标准化，各层输出的数值范围可能差异很大（部分值极大、部分极小），下一层难以处理，训练过程会非常不稳定。 | `Transformer` › 8. Layer Normalization/为什么需要 LayerNorm？ | `LayerNorm` || 保留 |
| q042 | 标题敏感 | Self-Attention 中 Softmax 起什么作用？ | 把注意力分数（logits）转化为归一化的注意力权重（概率分布），使所有权重之和为 1。 | `Transformer` › 4. Self-Attention 机制/Softmax 的作用 | `Softmax`、`Self-Attention` || 保留 |
| q043 | 标题敏感 | Agent 出现幻觉工具调用时，有哪些解法？ | 完善工具描述；对工具调用结果做参数校验；使用 strict 模式的 JSON Schema。 | `Agent 复习` › Agent 常见失败模式/1. 幻觉工具调用（Hallucinated Tool Call） | `幻觉工具调用` || 保留 |
| q044 | 标题敏感 | Agent 压缩上下文时，保留优先级从高到低是怎样的？ | 1. 架构决策，不得摘要；2. 已修改文件和关键变更；3. 验证状态（pass/fail）；4. 未解决的 TODO 和回滚笔记；5. 工具输出，可删，只保留 pass/fail 结论。 | `你不知道的Agent` › 你不知道的 Agent：原理、架构与工程实践/保留优先级 | `保留优先级` || 保留 |
| q045 | 标题敏感 | Agent 的可观测性与追踪要记录哪三个关键节点？ | tool_start、tool_end、turn_end。完整 Trace 同步落盘，并分发给日志系统、UI 更新、在线评测等下游。 | `你不知道的Agent` › 你不知道的 Agent：原理、架构与工程实践/9. 可观测性与追踪 | `可观测性`、`追踪` || 保留 |
| q046 | 单文档事实 | Encoder、Decoder 和 Encoder-Decoder 三种架构的代表模型分别是什么？ | Encoder 的代表是 BERT；Decoder 的代表是 GPT 系列；Encoder-Decoder 的代表是 T5 和 BART。 | `Transformer` › 9. 三种架构：Encoder、Decoder、Encoder-Decoder | — || 保留 |
| q047 | 单文档事实 | 解析 PDF 时，为什么表格模块需要单独做专项解析？ | 仅靠 OCR 很难完整识别出表格的准确含义，所以要筛选出标签为「表格」的模块，用专门的模型和工具进行深度解析。 | `RAG` › 文档解析 | — || 保留 |
| q048 | 单文档事实 | 重排序模型有哪三种训练目标？大多数常用的重排序模型采用哪种？ | Pointwise（单点评分）、Pairwise（对比学习）、Listwise（列表级优化）。大多数常用的 Reranking 模型（如 bge-reranker、ms-marco-MiniLM）采用 Pointwise 或 Pairwise。 | `RAG` › 重排序（Reranking）/Reranking 模型的训练目标 | — || 保留 |
| q049 | 单文档事实 | Human-in-the-Loop 的典型使用场景有哪些？ | 执行危险操作前（发送邮件、删除文件、调用付费 API）需要人工确认；Agent 给出中间规划后，人工审查计划再执行；需要人工补充信息才能继续的场景。 | `Agent 复习` › 人工审查 State，确认后恢复执行（传入 None 表示继续）/典型使用场景 | — || 保留 |
| q050 | 单文档事实 | 在 OpenAI 的 Agent 优先开发实践中，AGENTS.md 保留多少行？它起什么作用？ | 约 100 行，只作为索引，细节拆到各 docs 目录按需引用。 | `你不知道的Agent` › 你不知道的 Agent：原理、架构与工程实践/OpenAI 的 Agent 优先开发实践 | — || 保留 |
| q051 | 拒答 | RAPTOR 方法如何通过递归聚类和摘要构建树状索引来改进 RAG 检索？ | 知识库中没有这个问题的答案 | （空） | `RAPTOR` || 保留 |
| q052 | 拒答 | Corrective RAG（CRAG）中，检索评估器判断检索结果质量差时会采取什么动作？ | 知识库中没有这个问题的答案 | （空） | `Corrective`、`CRAG` || 保留 |
| q053 | 拒答 | RoPE（旋转位置编码）是如何把相对位置信息注入注意力计算的？ | 知识库中没有这个问题的答案 | （空） | `RoPE`、`旋转` || 保留 |
| q054 | 拒答 | FlashAttention 通过什么方式减少注意力计算中的显存读写？ | 知识库中没有这个问题的答案 | （空） | `FlashAttention`、`Flash` || 保留 |
| q055 | 拒答 | GQA（Grouped-Query Attention）相比 Multi-Head Attention，如何减少推理时的显存占用？ | 知识库中没有这个问题的答案 | （空） | `GQA`、`Grouped`、`分组查询`、`显存` || 保留 |
| q056 | 拒答 | MoE（混合专家）模型中的路由器如何为每个 token 选择专家？ | 知识库中没有这个问题的答案 | （空） | `MoE`、`混合专家`、`Mixture` || 保留 |
| q057 | 拒答 | SPLADE 模型是如何生成用于检索的稀疏向量的？ | 知识库中没有这个问题的答案 | （空） | `SPLADE`、`稀疏向量` || 保留 |
| q058 | 拒答 | DiskANN 是如何把向量索引放在 SSD 上，同时保持较低检索延迟的？ | 知识库中没有这个问题的答案 | （空） | `DiskANN`、`SSD` || 保留 |
| q059 | 拒答 | Tree of Thoughts（思维树）框架在推理时如何对多个候选思路进行搜索和回溯？ | 知识库中没有这个问题的答案 | （空） | `Tree of Thought`、`思维树`、`ToT` || 保留 |
| q060 | 拒答 | Google 提出的 A2A（Agent2Agent）协议中，Agent Card 用来描述什么？ | 知识库中没有这个问题的答案 | （空） | `A2A`、`Agent2Agent`、`Agent Card` || 保留 |

## 2. 原题的 gold_sections（q001–q020）

问题、答案和来源笔记都不改（`origin` = `original_v1`）。章节是用参考答案里的原文在切块结果里定位的。
q020 的参考答案用到了好坏示例，所以多加了 `Tool 设计规范/示例对比`。

| 编号 | 原 `question_type` | 问题 | gold_sections | 核对（对 / 不对） |
|---|---|---|---|---|
| q001 | factual | 原始 Transformer 使用什么函数生成位置编码？这些编码在训练过程中是否参与梯度更新？ | `Transformer` › 3. 位置编码（Positional Encoding） || 对 |
| q002 | conceptual | 为什么多头注意力（Multi-Head Attention）机制在增加 head 数量的同时不会增加整体计算量？ | `Transformer` › 5. Multi-Head Attention || 对 |
| q003 | comparison | Pre-Norm 和 Post-Norm 在 Layer Normalization 的放置位置上有何区别？为什么现代大模型普遍采用 Pre-Norm？ | `Transformer` › 8. Layer Normalization/Pre-Norm vs Post-Norm || 对 |
| q004 | application | 在训练用于文本生成的 Decoder 模型时，为了防止模型“偷看答案”，应在注意力分数矩阵的哪个具体区域填充 -∞？这属于哪种 Mask 机制？ | `Transformer` › 5. Multi-Head Attention/Mask 机制 || 对 |
| q005 | factual | Feed Forward Network (FFN) 在处理数据时，其内部维度的标准变化流程是怎样的？ | `Transformer` › 6. Feed Forward Network（FFN）/工作流程 || 对 |
| q006 | factual | 在 Naive RAG（朴素 RAG）架构中，标准的处理流程固定为哪三个步骤？ | `RAG` › RAG 三代架构演进/Naive RAG（朴素 RAG） || 对 |
| q007 | conceptual | 为什么 E5 系列模型不能混用 query 和 document 的编码策略？ | `RAG` › 存入数据库的 document 不需要加前缀（或加不同前缀） || 对 |
| q008 | comparison | OpenAI 的 text-embedding 系列模型与 E5 系列模型在处理输入前缀时有什么主要区别？ | `RAG` › 存入数据库的 document 不需要加前缀（或加不同前缀） || 对 |
| q009 | application | 在实际工程中，如果缺乏历史数据来评估 RAG 系统，文档推荐的最常用构建测试集方案是什么？ | `RAG` › Advanced RAG 进阶模式/解决方案 || 对 |
| q010 | factual | 在计算 RAGAS 的 Faithfulness（忠实度）指标时，具体的计算公式是什么？ | `RAG` › RAG 系统评估/RAGAS 核心指标的计算方式 || 对 |
| q011 | factual | 在 Agent 的记忆系统设计中，MEMORY.md 对应的是哪一种记忆类型？ | `你不知道的Agent` › 你不知道的 Agent：原理、架构与工程实践/5. 记忆系统设计 || 对 |
| q012 | comparison | 根据文档，Workflow 和 Agent 的核心区别是什么？ | `你不知道的Agent` › 你不知道的 Agent：原理、架构与工程实践/Workflow 和 Agent 有什么区别 || 对 |
| q013 | conceptual | 为什么在上下文分层管理中强调“别把确定性逻辑放进上下文”？正确的处理方式是什么？ | `你不知道的Agent` › 你不知道的 Agent：原理、架构与工程实践/上下文为什么要分层 || 对 |
| q014 | application | 根据 Harness 的关键结论，若要让 Agent 在开发任务中发挥最大效能，团队应如何调整任务属性并设定验收标准？ | `你不知道的Agent` › 你不知道的 Agent：原理、架构与工程实践/Harness 的关键结论 || 对 |
| q015 | factual | 在工具设计的第三代演进中，采用“Programmatic Tool Calling（代码编排）”方式后，token 消耗大约能从多少降低到多少？ | `你不知道的Agent` › 你不知道的 Agent：原理、架构与工程实践/工具设计如何演进 || 对 |
| q016 | factual | LangGraph 的 Checkpointer 机制主要提供哪四种核心能力？ | `Agent 复习` › LangGraph 深入：Checkpointer 与持久化 || 对 |
| q017 | conceptual | 相比原始的 ReAct 框架，Reflexion 框架的核心优势是什么？ | `Agent 复习` › Reflexion 框架 || 对 |
| q018 | comparison | 在 LangGraph 中实现 Function Call 时，为什么推荐使用“独立 Tool 节点”而非“共享 Tool 节点”？ | `Agent 复习` › Function Call（工具调用）/LangGraph 中的 Function Call 实现 || 对 |
| q019 | application | 当 Agent 运行出现“上下文爆炸（Context Overflow）”问题时，文档推荐了哪些具体解法？ | `Agent 复习` › Agent 常见失败模式/3. 上下文爆炸（Context Overflow） || 对 |
| q020 | application | 根据文档中的正反示例对比，一个“好的工具描述”应当包含哪些关键要素？ | `Agent 复习` › Tool 设计规范/高质量 Tool Description 的要素<br>`Agent 复习` › Tool 设计规范/示例对比 || 对 |

## 3. 备注

q049、q023、q024、q007、q008 的 heading_path 受 DocProcessor 把代码注释当成标题的问题影响（例如 q049 的真实标题 `# Human-in-the-Loop（人工介入）` 被代码块里的 `# 人工审查 State…` 截断）。报告里要把这 5 题单独列出。
