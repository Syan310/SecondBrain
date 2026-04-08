---

---
# RAG 三代架构演进

理解 RAG 的整体发展脉络，是面试中体现全局观的重要方式。

## Naive RAG（朴素 RAG）

最早期的 RAG 范式，流程固定为：**索引 → 检索 → 生成**。

- **索引**：文档解析 → 分块 → Embedding → 存入向量数据库
- **检索**：用户 Query 向量化后做相似度检索，取 Top-K
- **生成**：将检索结果拼入 Prompt，送入 LLM 生成回答

**局限性**：

- 检索精度低，query 与文档语义差距大时效果差
- 没有查询优化，没有重排序
- 生成时对检索结果的质量没有验证，幻觉率高

## Advanced RAG（增强 RAG）

针对 Naive RAG 的痛点，在**检索前**、**检索中**、**检索后**三个阶段引入优化：

- **检索前**：查询改写、HyDE、Query Decomposition 等
- **检索中**：混合检索（BM25 + 向量）、元数据过滤
- **检索后**：Reranking、上下文压缩、Lost in the Middle 应对策略

## Modular RAG（模块化 RAG）

最新一代架构，将 RAG 中的每个功能拆解为**可替换、可组合的独立模块**，支持灵活编排。

- 检索模块、记忆模块、融合模块、路由模块、预测模块等均可单独替换
- 不再局限于单一的"检索→生成"线性流程，可以根据任务动态组合
- Agentic RAG 是 Modular RAG 的典型代表：Agent 自主决定何时检索、检索什么、是否需要重新检索

**三代对比**：

|   | Naive RAG | Advanced RAG | Modular RAG |
| --- | --- | --- | --- |
| 流程 | 固定线性 | 线性 + 各阶段优化 | 动态可编排 |
| 灵活性 | 低 | 中 | 高 |
| 复杂度 | 低 | 中 | 高 |
| 适用场景 | 原型验证 | 大多数生产场景 | 复杂多源、多步推理场景 |

---

# RAG vs Fine-tuning：如何选择？

这是初级 Agent/LLM 工程师面试中**出现频率最高**的问题之一。

## 核心判断维度

| 维度 | 倾向 RAG | 倾向 Fine-tuning |
| --- | --- | --- |
| **知识更新频率** | 知识需要频繁更新（如新闻、文档库） | 知识相对静态，一次训练长期有效 |
| **知识来源** | 有明确的外部知识库，需要引用溯源 | 需要改变模型的行为/风格/推理方式 |
| **数据量** | 无需大量标注数据 | 需要高质量的（指令, 回答）对 |
| **成本** | 推理成本较高（每次检索 + 生成） | 训练成本高，推理成本低 |
| **幻觉风险** | 可通过引用溯源降低幻觉 | 难以控制模型"记忆"的准确性 |
| **领域适应** | 适合知识密集型任务 | 适合格式/风格/能力迁移 |

## 实践结论

- **优先考虑 RAG**：当任务是"根据给定文档回答问题"、知识需要更新、需要引用来源时
- **优先考虑 Fine-tuning**：当任务是改变模型输出风格、适配特定格式、提升特定推理能力时
- **两者结合**：在实际生产中，往往是先 Fine-tuning 让模型具备特定能力，再配合 RAG 提供实时知识——两者不是互斥关系

---

# 文档解析

RAG 的第一步是文档解析。这一步的目的是将各种格式文档中的内容全部解析为文本格式，以便进行后续步骤。

解析过程中主要会遇到一些问题，比如 PDF 文件。PDF 中包含大量的图片和表格，单纯依靠文本识别工具无法达到理想的效果。因此，有一套专门解析 PDF 的流程，具体步骤如下：

1. **OCR 识别与模块化**：将 PDF 全部进行 OCR 识别，把里面的所有内容都拆解成一个个小的模块。
2. **模块打标签**：给 OCR 识别后的模块贴上标签，区分其属性（如正文、标题、页眉、页脚等）。
3. **表格专项解析**：筛选出标签为"表格"的模块。由于仅靠 OCR 很难完整识别出表格的准确含义，因此需要使用专门的模型和工具进行深度解析。
4. **模块拼接**：(a) 横向拼接：将属于同一行的模块进行拼接。(b) 纵向拼接：将属于同一段落或同一页面的行进行拼接。
5. **冗余清洗**：清洗掉多余的模块，例如页眉、页脚等无关部分。
6. **提取与分离**：将表格、图片等模块提取出来。它们需要与文本模块分开，独立进行后续的解析处理。

最后完成拼接，整个解析流程就结束了。

---

# RAG 分块（Chunking）

RAG 分块的步骤是将解析好的文档分成一个个小块，然后进行向量化，存储进向量数据库。

**为什么需要分块？**

7. **向量模型的输入限制**：向量模型的输入大小是有长度限制的。
8. **语义精确度**：如果将一个长文本全部变成一整个向量，语义精确度会很低，检索准确度不够。

## 切块方法

9. **按照规则切块**
    - 设定固定的 token 数量（如 256 个 token），通常还会设置**重叠窗口（Overlap）**（如相邻块重叠 50 个 token），避免切块边界处丢失上下文语义。
    - 按照章节或段落切分。
    - **递归切分（Recursive Splitting）**：先按大粒度（章节）切分，若块仍过长，再按更小粒度（段落、句子）递归切分，直到满足长度限制。这是 LangChain `RecursiveCharacterTextSplitter` 的默认策略。
10. **按照语义切块**：通过计算句子之间的语义相似度来判断是否切分。相似度高于阈值则合并；低于阈值则切开。阈值判断方式包括标准差法、百分位法、梯度法。

## 切块大小如何选择？

11. 给定几个不同量级的块大小（如 128、256、512 token）。
12. 按每个量级分别切块，存入对应的向量数据库。
13. 针对用户提问生成回复。
14. 用"忠诚度测试"和"相关性测试"对不同块大小的回复打分，选出最优量级。

## 切块时的元数据记录

如果今后需要追溯原文，切块时需要将当前块在原文中的位置信息（所在段落、章节等）存入元数据，方便后续检索。

## 复杂结构文档的层级保留

对于有多层嵌套标题、表格、图例的复杂文档，保留层级信息的解决思路：

15. **结构化存储**：将章节、标题、图片、表格全部转化为可被索引的元数据，存入数据库。
16. **增强 Chunk 上下文**：在每个 Chunk 中附带当前内容所在的层级关系上下文（处于哪一段、哪一页、哪一章节），存入 Metadata。
17. **元数据索引检索**：检索时根据用户问题先对元数据进行索引过滤，快速定位到特定层级再检索。

---

# Query-Doc 语义鸿沟

当用户的 query 与原文的差距过大时，常见策略：

18. **文档端增强（Document-side）**：在每个 chunk 中用大模型构建几个 QA 问答对，检索时同时检索原文和问答对，弥补语义鸿沟。
19. **查询端增强（Query-side）**：对用户 query 进行改写或扩展（详见查询改写部分）。
20. **HyDE（Hypothetical Document Embeddings）**：先让大模型根据 query 生成一个假设性的回答文档，再用假设文档的向量去检索真实文档，缩小语义差距。

### HyDE 的完整原理与适用场景

**为什么 HyDE 能起作用？**

用户的原始 query 往往是一个短问题（如「RAG 如何优化？」），而数据库中存储的是详细的技术文档。短问题的向量与长文档的向量天然存在语义差距。HyDE 的思路是：如果我们先让 LLM 生成一段假设性的回答，这段回答的向量与数据库中真实答案的向量就会更接近，从而实现更好的检索。

**HyDE 完整流程**：

```javascript
用户 Query → LLM 生成假设性局部答案 → 假设文档进行 Embedding
    ↓
用假设文档向量检索向量数据库
    ↓
取回真实文档 chunk，送入 LLM 生成最终回答
```

**最适合的场景**：

- 专业知识库问答（如内部文档、技术文档）
- Query 很短但数据库中的内容较长，语义差距显著
- 通用性知识域，其中 LLM 天然具备较准确的背景知识

> ⚠️ **HyDE 的局限性**：

> 1. 在专业性强的垂直领域（如医疗、法律、金融），LLM 生成的假设文档本身可能包含错误，导致向量方向偏移，反而比直接检索效果更差。

> 2. HyDE 引入了额外的 LLM 调用（生成假设文档），增加了延迟和成本。

> 3. 建议：在使用前，先用 RAGAS 等工具对比开启 HyDE 前后的检索质量，由指标决定是否启用。

---

# Contextual Retrieval（上下文检索）

**Contextual Retrieval** 是 Anthropic 于 2024 年提出的检索增强方案，通过在每个 chunk 写入向量库之前，**用 LLM 为其自动生成一段上下文前缀**，显著提升检索命中率。

## 核心思路

传统分块后，chunk 往往缺乏上下文，导致语义孤立。例如：

- 原始 chunk：`"该季度的收入同比增长了 12%"`
- 上下文前缀：`"本文档是 Acme 公司 2023 年 Q3 财报。该季度的收入同比增长了 12%"`

加上前缀后，chunk 的语义更完整，检索向量更准确。

## 实现流程

21. 对每个 chunk，将原始文档（或文档摘要）+ chunk 内容一起送入 LLM
22. 让 LLM 生成一段 1-2 句的上下文描述，说明该 chunk 在整篇文档中的位置和含义
23. 将"上下文描述 + chunk 原文"拼接后再做 Embedding 和 BM25 索引

## 效果

Anthropic 测试数据显示，Contextual Retrieval 配合 BM25 可以将检索失败率降低约 **49%**，是目前成本较低且效果显著的检索优化手段。

> 📌 成本考量：为每个 chunk 调用 LLM 生成前缀会增加索引阶段的成本，适合使用 Prompt Caching 来降低开销（Anthropic API 支持对文档部分做 Cache）。

---

# 向量数据库的 Metadata Filter

在实际工程中，**Metadata Filter（元数据过滤）** 是 RAG 系统中几乎必用的优化手段，也是面试中考察工程经验的高频点。

## 核心思路

在做向量相似度检索**之前**，先用元数据条件对候选集进行过滤，大幅缩小检索范围，同时提升结果的精准度。

## 典型元数据字段

| 字段 | 示例 |
| --- | --- |
| 文档来源 | `source: "annual_report_2023.pdf"` |
| 文档类型 | `doc_type: "legal" / "technical" / "faq"` |
| 时间戳 | `created_at: "2024-01-01"` |
| 章节/部门 | `department: "finance"` |
| 语言 | `language: "zh"` |
| 权限标签 | `access_level: "internal"` |

## 实现方式（以 Qdrant 为例）

```python
results = qdrant_client.search(
    collection_name="docs",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(key="doc_type", match=MatchValue(value="legal")),
            FieldCondition(key="language", match=MatchValue(value="zh"))
        ]
    ),
    limit=10
)
```

## 与 Hybrid Search 的关系

Metadata Filter 通常作为**前置过滤**步骤，在混合检索（BM25 + 向量）之前执行：

24. **Metadata Filter** → 过滤不相关文档
25. **Hybrid Search** → 在过滤后的候选集中进行混合检索
26. **Reranking** → 对检索结果精排

这个三阶段组合是目前工业界最主流的 RAG 检索管线设计。

---

# 向量化与索引

分块完成后，需要将每个 chunk 转化为向量（Embedding），并存入向量数据库进行索引。

## Embedding 模型的选择

常见的 Embedding 模型有：

- **OpenAI text-embedding-3-small / large**：通用性强，支持多语言。
- **BGE 系列（BAAI）**：开源模型中表现优异，有中文优化版本（如 bge-large-zh）。
- **E5 系列（Microsoft）**：在多个基准上表现优秀，支持 instruction-tuned 变体。
- **Cohere Embed v3**：支持多语言和多种输入类型（query vs document）。

选择时需要考虑：维度大小、最大输入长度、多语言支持、以及在目标领域的效果。

## Embedding 的非对称编码（Query/Document Asymmetry）

这是实际使用 BGE、E5 等模型时**最常见的踩坑点**，面试也会考察。

部分 Embedding 模型对 **query** 和 **document** 需要加不同的 **instruction prefix**，否则检索效果会显著下降：

**BGE 系列**：

```python
# 检索时的 query 需要加前缀
query = "Represent this sentence for searching relevant passages: " + user_query

# 存入数据库的 document 不需要加前缀（或加不同前缀）
document = chunk_text
```

**E5 系列**：

```python
query = "query: " + user_query
document = "passage: " + chunk_text
```

**原因**：这类模型在训练时对 query 和 document 使用了不同的编码策略（asymmetric training），混用会导致向量空间不对齐，相似度计算失准。

> ⚠️ OpenAI 的 text-embedding 系列不需要加前缀，对称使用即可。使用其他开源模型前务必查阅对应文档。

## 向量数据库与索引策略

常见的向量数据库包括：Milvus、Pinecone、Weaviate、Qdrant、Chroma、pgvector 等。

索引类型主要有：

27. **Flat（暴力搜索）**：精确但慢，适合小规模数据。
28. **IVF（Inverted File Index）**：将向量空间划分为多个簇，检索时只搜索最近的几个簇，是精度和速度的折中。
29. **HNSW（Hierarchical Navigable Small World）**：基于图的近似最近邻算法，检索速度快且精度高，是目前最常用的索引方式。
30. **PQ（Product Quantization）**：通过量化压缩向量以节省内存，适合超大规模数据集。

### IVF 原理详解

IVF 是一种**基于聚类的近似最近邻**算法，核心思路：

**索引构建阶段**：

31. 用 k-means 算法将所有向量聚类成 `nlist` 个簇（如 1024 个）
32. 记录每个向量属于哪个簇，构建"倒排列表"（每个簇 → 其包含的向量 ID 列表）

**检索阶段**：

33. 将 query 向量与所有 `nlist` 个簇中心做距离计算，找出最近的 `nprobe` 个簇（`nprobe` 是关键超参数，默认值通常为 1-64）
34. 只在这 `nprobe` 个簇内进行精确搜索
35. 返回 Top-K 结果

`**nprobe**`** 的 trade-off**：`nprobe` 越大，搜索的簇越多，召回率越高，但速度越慢；`nprobe = nlist` 时退化为暴力搜索。

### HNSW 原理详解

HNSW 是目前**最常用的 ANN 索引**，兼顾检索速度和精度。

**核心思想**：构建一个**多层次的跳表式图结构**，高层稀疏（用于快速导航），低层稠密（用于精确搜索）。

**图构建阶段**：

36. 每个新向量插入时，随机分配到某一层（层数越高概率越低，服从指数分布）
37. 在每一层，通过贪心搜索找到该层中与新向量最近的 `M` 个邻居（`M` 是关键超参数，控制每个节点的连接数），建立双向边
38. 最终形成一个从粗到细的分层图：顶层连接少但跨度大，底层连接多且精细

**检索阶段**：

39. 从最顶层的入口节点出发，贪心地向与 query 更近的节点移动
40. 到达每一层的局部最优后，下降到下一层继续搜索
41. 在最底层（第 0 层）执行精细搜索，返回 Top-K

**关键超参数**：

- `M`：每个节点在构建时的最大连接数，越大精度越高，内存和构建时间越多（典型值 16-64）
- `ef_construction`：构建阶段的动态候选集大小，控制构建质量（越大越好但越慢）
- `ef_search`：检索阶段的动态候选集大小，控制检索精度（可以在不重建索引的情况下动态调整）

**HNSW vs IVF 对比**：

|   | IVF | HNSW |
| --- | --- | --- |
| 检索速度 | 快 | 更快 |
| 召回精度 | 中等 | 高 |
| 内存占用 | 低 | 高（需存储图结构） |
| 增量插入 | 不友好（需重新聚类） | 友好（直接插入图） |
| 适用场景 | 大规模数据，内存受限 | 中等规模，追求高精度和低延迟 |

---

# 查询改写（Query Rewriting）

用户的原始 query 往往不够精确或过于口语化，直接用来检索效果可能不佳。查询改写是在检索前对 query 进行优化的过程。

常见的查询改写策略：

42. **Query Expansion（查询扩展）**：使用大模型将原始 query 扩展为多个不同角度的子查询，分别检索后合并结果。
43. **Query Decomposition（查询分解）**：将复杂问题拆分为多个简单的子问题，分别检索后整合答案。适用于需要多步推理的复杂问题。
44. **Step-back Prompting（回退提问）**：先让大模型生成一个更高层次、更抽象的问题，用这个抽象问题检索到更广泛的背景知识，再结合原始问题进行回答。
45. **Query Rewrite with LLM（大模型改写）**：直接让大模型将用户的口语化提问改写为更精确、更适合检索的表述。

---

# 检索环节

## 混合检索（Hybrid Search）

**BM25 与向量检索的融合流程**：

46. 拿到用户 query 后，分别进行一次关键词检索和向量相似度检索。
47. 拿到两种方式对应的分数。
48. 对分数进行归一化，然后使用加权融合或 **RRF（Reciprocal Rank Fusion）** 进行融合。
    - 加权融合示例：向量相似度占 0.7，关键词占 0.3（**注意：权重是可调超参数**，需根据实际场景调优）。
    - RRF 方法：不依赖原始分数，根据各检索结果在各自列表中的排名融合，公式为 $score = sum frac{1}{k + rank_i}$，其中 k 通常为 60。
49. 融合后对最终分数排序，取得最终 chunk。

## BM25 与向量检索各自优势

- **BM25（关键词检索）**：检索短名词、专有名词时效率和精准度高。
- **向量相似度检索**：检索较长语句、或用户提问模糊时优势明显。

## 检索质量评估指标

50. **Recall（召回率）**：在所有相关文档里，具体召回了多少？
51. **Precision（准确率）**：在所有召回文档里，真正相关的有多少？
52. **MRR（Mean Reciprocal Rank）**：第一个相关文档在结果列表中排名倒数的平均值，衡量将相关结果排在前面的能力。
53. **NDCG（Normalized Discounted Cumulative Gain）**：考虑文档相关性等级和排名位置的综合指标，越靠前的相关文档贡献越大。

---

# 重排序（Reranking）

初次检索返回的候选文档可能存在噪声。Reranking 是在初次检索之后，对候选文档进行二次精排的过程。

## 为什么需要 Reranking？

54. 初次检索使用的是 **Bi-Encoder（双塔模型）**，query 和 document 分别编码，效率高但精度有限。
55. Reranking 使用 **Cross-Encoder（交叉编码器）**，将 query 和 document 拼接后一起编码，能捕捉更细粒度的语义交互，精度更高但速度较慢。
56. 实践中通常先用 Bi-Encoder 快速召回 Top-K（如 50-100）个候选文档，再用 Cross-Encoder 精排，最终取 Top-N（如 5-10）送入大模型。

## Reranking 模型的训练目标

Cross-Encoder 的训练目标决定了其对相关性的建模方式，面试中可能被追问：

- **Pointwise（单点评分）**：对每个（query, document）对独立打分（相关/不相关二分类，或相关性分数），训练简单，但无法显式建模文档间的相对顺序。
- **Pairwise（对比学习）**：对两个文档进行相对比较（哪个更相关），直接优化排序目标，效果通常优于 Pointwise。
- **Listwise（列表级优化）**：对整个候选列表建模，直接优化排名指标（如 NDCG），效果最好但训练成本最高。

大多数常用的 Reranking 模型（如 bge-reranker、ms-marco-MiniLM）采用 **Pointwise** 或 **Pairwise** 训练。

## 常见的 Reranking 方案

57. **Cross-Encoder 模型**：如 Cohere Rerank、bge-reranker、ms-marco-MiniLM 等。
58. **LLM-based Reranking**：使用大模型对候选文档的相关性进行打分或排序。
59. **ColBERT（Late Interaction）**：介于 Bi-Encoder 和 Cross-Encoder 之间，token 级别的交互，兼顾效率和精度。

---

# 上下文压缩（Contextual Compression）

**Contextual Compression** 解决的是检索到的 chunk 中包含大量无关内容的问题。即使 Reranking 帮助排除了不相关的 chunk，剩下的 chunk 内部也可能只有一小段与问题相关，大量内容占据了宝贵的上下文窗口。

## 核心问题

假设一个 chunk 是一整段关于公司财务状况的文字，但用户的问题只关心其中关于“年化收益”的命题。如果我们将整个 chunk 送入 LLM，大量无关内容会干扰生成。Contextual Compression 就是在此步骤中，先将 chunk 压缩为与问题相关的小片段，再送入 LLM。

## 两种实现方式

**1. LLM 压缩提取（LLM Extractive Compression）**

将每个 chunk 和 query 一起送入 LLM，要求 LLM 提取其中与问题相关的关键句子或片段：

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 检索时自动压缩每个返回的 chunk
results = compression_retriever.get_relevant_documents(query)
```

**2. 基于 Embedding 的筛选（Embedding-based Filter）**

不使用 LLM，而是将每个句子分别计算与 query 的向量相似度，低于阈值的句子直接删除：

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter

embeddings_filter = EmbeddingsFilter(embeddings=embeddings_model, similarity_threshold=0.76)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=base_retriever
)
```

此方式没有额外的 LLM 调用开销，速度更快，但精细度不如 LLM 压缩。

## 与 Reranking 的关系

Contextual Compression 和 Reranking 不是互斥的，它们解决不同的问题：

- **Reranking**：解决“哪些 chunk 更相关”的问题，对候选 chunk **排序**
- **Contextual Compression**：解决“chunk 内部的无关内容”的问题，对单个 chunk 进行**内容精简**

典型的全链路排列：**Hybrid Search → Reranking → Contextual Compression → Generation**

---

# 生成环节（Generation）

检索到相关 chunk 后，将其与用户 query 一起送入大模型进行回答生成。

## Prompt 构建

典型的 RAG Prompt 结构：

60. **System Prompt**：定义模型角色和回答规范（如"请根据提供的上下文回答问题，如果上下文中没有相关信息，请明确说明"）。
61. **Context**：将检索到的 chunk 按相关性排序后拼接为上下文。
62. **User Query**：用户的原始问题。

## 生成环节中的关键问题

### 1. Lost in the Middle（中间信息丢失）

**现象**：研究表明，LLM 对上下文中间部分的信息利用率显著低于头部和尾部（即"Lost in the Middle"现象），导致关键信息被忽略。

**应对策略**：

- 将最相关的 chunk 放在上下文的**开头或末尾**，而非中间
- **位置感知排列（Position-aware Ordering）**：相关性最高的放首位，第二高的放末尾，其余按序填充中间
- 控制送入的 chunk 总数，不要一味堆砌

### 2. 幻觉（Hallucination）控制

- 在 Prompt 中明确要求模型"仅基于上下文回答，不要编造信息"。
- 要求模型在回答中标注信息来源（引用具体 chunk）。
- 设置较低的 temperature 以减少随机性。

### 3. Faithfulness（忠实度）

生成的回答应忠实于检索到的上下文，不应添加上下文中不存在的信息。可以通过后处理步骤（如 NLI 模型）检测生成内容与上下文的一致性。

### 4. 引用与溯源

在生成的回答中标注每段信息来自哪个 chunk / 文档，方便用户追溯验证。

### 5. 流式生成（Streaming）

在 RAG 场景中，`stream()` 可以让用户在 LLM 生成第一个 token 时就开始看到输出，而不需要等待完整回答，显著提升感知响应速度。

在 LangChain / LangGraph 中使用流式生成：

```python
# LangChain LCEL 链
for chunk in rag_chain.stream({"question": user_query}):
    print(chunk, end="", flush=True)

# LangGraph
for event in graph.stream({"messages": [user_msg]}, stream_mode="values"):
    print(event)
```

> ⚠️ 注意：流式生成与 Reranking 的结合需要特别处理——Reranking 需要等所有候选文档检索完成后才能执行，之后再启动流式生成。

---

# RAG 系统评估

一个完整的 RAG 系统评估需要覆盖检索和生成两个环节。常用的评估框架有 RAGAS、TruLens 等。

## 检索质量评估

63. **Context Relevancy（上下文相关性）**：检索到的 chunk 是否与用户问题相关？
64. **Context Recall（上下文召回率）**：回答所需的关键信息是否都被检索到了？

## 生成质量评估

65. **Faithfulness（忠实度）**：生成的回答是否忠实于检索到的上下文，有没有产生幻觉？
66. **Answer Relevancy（回答相关性）**：生成的回答是否切题地回答了用户的问题？
67. **Answer Correctness（回答正确性）**：如果有标准答案，生成的回答与标准答案的一致程度如何？

## RAGAS 核心指标的计算方式

面试中常被追问"指标是怎么算的"，需要掌握底层逻辑：

**Faithfulness（忠实度）**：

68. 将生成回答拆解为若干原子性陈述（atomic claims）
69. 对每个陈述，用 LLM 判断是否可以从检索到的上下文中推断出来（NLI 风格的判断）
70. `Faithfulness = 可被上下文支持的陈述数 / 总陈述数`

**Answer Relevancy（回答相关性）**：

71. 让 LLM 根据生成的回答逆向生成若干个可能的问题
72. 计算这些逆向生成的问题与原始用户问题的向量相似度
73. 取平均值作为 Answer Relevancy 分数
74. 核心思想：如果回答真的回答了问题，那么从回答中生成的问题应该与原始问题高度相似

**Context Recall（上下文召回率）**：

75. 将 Ground Truth 答案拆解为若干原子性陈述
76. 对每个陈述，用 LLM 判断是否可以从检索到的上下文中找到支持
77. `Context Recall = 被上下文支持的陈述数 / Ground Truth 总陈述数`

## 端到端评估方法

78. 构建标准的 QA 测试集（包含问题、标准答案、相关文档）。
79. 分别对检索和生成环节打分。
80. 使用更强大的 LLM（如 GPT-4）作为评判者（LLM-as-Judge）进行自动评估。
81. 结合人工评估进行校准，确保自动评估的可靠性。

---

# RAG 系统性能优化与 Latency 减少

面试中常被问到“如何提升 RAG 系统的响应速度”，这是工程化思维的考证。主要与三个层次优化：

## 检索层 Latency 优化

**1. 并行检索（Async Concurrent Retrieval）**

当一个问题被拆分为多个子问题（Query Decomposition），或者需要对多个知识库同时检索时，不要安排成串行调用，应该并行执行：

```python
import asyncio
from langchain_core.runnables import RunnableParallel

# 并行对多个子问题检索
async def parallel_retrieve(sub_queries):
    tasks = [retriever.ainvoke(q) for q in sub_queries]
    results = await asyncio.gather(*tasks)
    return results
```

**2. 和光 Embedding（On-the-fly Embedding 预计算）**

将常见 query 模式提前做好 embedding 并缓存，检索时直接读取。

## 语义缓存（Semantic Cache）

**Semantic Cache** 是 RAG 系统中最重要的 Latency 优化手段之一，面试高频考同。

**核心思路**：普通缓存策略（如 Redis）基于精确字符串匹配，如果用户问了一个语意相同但表述不同的问题（如「RAG 怎么优化？」 vs「怎样提升 RAG 效果？」），就无法命中缓存。Semantic Cache 则是基于**向量相似度**匹配，只要新问题的向量与缓存中某个问题的向量相似度高于阈值，就直接返回缓存的结果，完全跳过检索和 LLM 调用。

**实现示意**：

```python
class SemanticCache:
    def __init__(self, embedder, similarity_threshold=0.95):
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.cache = []  # [(query_embedding, answer), ...]

    def get(self, query):
        query_emb = self.embedder.embed_query(query)
        for cached_emb, cached_answer in self.cache:
            similarity = cosine_similarity(query_emb, cached_emb)
            if similarity >= self.threshold:
                return cached_answer  # 命中缓存，直接返回
        return None  # 未命中

    def set(self, query, answer):
        query_emb = self.embedder.embed_query(query)
        self.cache.append((query_emb, answer))
```

**GPTCache** 是目前最成熟的开源 Semantic Cache 实现，支持与 LangChain 集成。

**适用场景**：高重复查询场景（如客服机器人、FAQ 系统），相同或相似问题重复率高。

**局限性**：阈值选择需谨慎——阈值过低会返回错误结果（不同问题被当成相同）；对于时间敏感或知识频繁更新的场景，需要设置缓存过期时间（TTL）。

## 其他工程化优化手段

| 优化手段 | 具体操作 | 适用场景 |
| --- | --- | --- |
| **流式输出（Streaming）** | LLM 首个 token 出现时即展示 | 对感知延迟要求高，全部场景 |
| **小化 Chunk 数量** | Top-K 调低至 3-5，配合 Reranking | 小上下文窗口大小环境 |
| **Embedding 缓存** | 相同 query 的 embedding 结果缓存 | 多用户反复相同 query |
| **密集请求分批** | 多个 chunk 的 embedding 批处理 | 离线批量建库 |
| **负载均衡** | 多实例 LLM 、向量库分剧 | 高并发生产环境 |

---

# Advanced RAG 进阶模式

## Multi-hop RAG（多跳检索）

对于需要多步推理的复杂问题，单次检索可能无法获取所有必要信息。Multi-hop RAG 会进行多轮检索，每一轮根据上一轮的结果和中间推理来确定下一步检索什么。

## Self-RAG（自反思 RAG）

模型在生成过程中自行判断是否需要检索、检索到的内容是否有用、生成的内容是否被检索结果支持，从而动态决定检索和生成策略。

## Parent-Child Chunking（父子分块）

用小的 chunk（child）进行精确检索，但在送入大模型时，返回其所属的更大的 chunk（parent），以提供更完整的上下文。

## Agentic RAG（智能体 RAG）

将 RAG 与 Agent 框架结合，让模型自主决定何时检索、从哪个数据源检索、是否需要重新检索等，具有更强的灵活性和自主性。

## GraphRAG（知识图谱 RAG）

**GraphRAG** 是微软提出的方案，将知识图谱（Knowledge Graph）与 RAG 结合，解决传统向量检索在**多实体关联推理**场景下的局限。

**核心思路**：

82. **索引阶段**：用 LLM 从文档中抽取实体和实体间的关系，构建知识图谱；对图中的社区（紧密相关的节点群）生成摘要
83. **检索阶段**：根据查询类型选择检索策略：
    - **Local Search**：适合具体事实性问题，从知识图谱中检索相关实体及其邻居节点
    - **Global Search**：适合需要全局理解的问题（如"整篇文档的主要主题是什么"），利用社区摘要进行检索
84. **生成阶段**：将检索到的图谱上下文送入 LLM 生成回答

**相比传统 RAG 的优势**：

|   | 传统向量 RAG | GraphRAG |
| --- | --- | --- |
| 多实体关联推理 | 弱（chunk 之间割裂） | 强（图结构天然表达关系） |
| 全局主题理解 | 弱 | 强（社区摘要） |
| 构建成本 | 低 | 高（需要 LLM 抽取图谱） |
| 适用场景 | 通用文档问答 | 人际关系、知识网络、多跳推理 |

---

## RAG 系统冷启动问题

**冷启动问题**指的是新建的 RAG 系统缺乏历史 QA 数据和用户行为时，如何进行评估和调优的问题。

### 冷启动面临的三个核心挑战

85. **没有历史 QA 对**： RAGAS 等评估框架需要（问题, 标准答案, 相关文档）三元组数据集，新系统根本没有。
86. **没有用户反馈**：无法根据使用模式来优化分块大小、检索策略。
87. **超参数选择困难**：切块大小、Overlap、Top-K 等超参数无制评估基准。

### 解决方案

**方案 1：利用 LLM 自动生成测试集（最常用）**

在没有历史数据的情况下，可以直接让 LLM 从已有文档中生成问题-答案对：

```python
# 对每个 chunk 生成测试问题
prompt = f"""根据以下文档内容，生成 3 个有代表性的问题和对应答案。
输出格式为 JSON：[{{"question": ..., "answer": ...}}, ...]

文档内容：{chunk}"""
qa_pairs = llm.invoke(prompt)
```

这个测试集虽然是 "AI 生成的"，但已经可以用于初期评估分块质量和检索准确度。

**方案 2：AB 测试（分块策略对比）**

对同一批生成的 QA 对，分别用不同切块策略（128/256/512 token，固定切分 vs 语义切分）进行检索测试，选择指标最好的策略。

**方案 3：少量人工标注作为基准**

即使只有 20-50 个人工标注的高质量 QA 对，也可以对 AI 自动评估进行校准，确保评估可靠性。

**方案 4：渐进式上线**

初期以经验公式确定超参数（如切块 256 token，Top-K=5），小量上线后采集实际用户反馈（带赞 / 蹄 / 显式反馈），再逐步调优。

---

## Multi-modal RAG（多模态 RAG）

随着 GPT-4o、Gemini 、Claude 等多模态模型的普及，Multi-modal RAG 正成为面试考察点。

### 核心问题

传统 RAG 只能处理文本。但实际文档中全是图表、图片、流程图，这些内容如何检索？

### 三种主流方案

**方案 1：将图像转为文本，再索引**

用多模态 LLM 或专用模型（如表格 OCR、图像描述模型）将图片和图表转化为文本描述，再索引文本。优点是架构简单，使用现有文本检索管线；缺点是图像转文本可能损失视觉细节。

**方案 2：多模态 Embedding（图文统一向量空间）**

使用 CLIP、LLaVA-Embed 等多模态 Embedding 模型，将图像和文本映射到**同一向量空间**中，直接计算文本 query 与图像的相似度。优点是语义更准确；缺点是需要对应的多模态 Embedding 模型。

**方案 3：图文原始内容存储 + 摘要检索**（推荐）

这是目前工程中最常用的方案：

88. 用 LLM 为每张图片/表格生成文字摘要，索引摘要一便于检索
89. 存储图片/表格的原始内容（Base64 或存储路径）与摘要的对应关系
90. 检索时用摘要索引定位，但送入 LLM 时传递原始图片，让多模态 LLM 直接看图回答

```python
# 索引阶段：生成图片摘要并存储对应关系
for image in document_images:
    summary = llm.invoke(f"Describe this image in detail: <image>{image.base64}</image>")
    vector_store.add(text=summary, metadata={"image_ref": image.path})

# 检索阶段：用摘要检索，用原始图片生成
results = vector_store.search(query)  # 检索到摘要
for r in results:
    original_image = load_image(r.metadata["image_ref"])  # 取回原始图片
    answer = multimodal_llm.invoke(query, images=[original_image])  # 多模态生成
```

### 适用场景

- 包含大量图表和图像的财务报告问答
- 技术文档（包含流程图、架构图）问答
- 产品说明书、教材等图文混排场景