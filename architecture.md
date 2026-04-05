# SecondBrain 项目规划图

## 1. 系统整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React + Vite)                  │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌────────────┐  │
│  │ 聊天问答  │  │ 知识盲区诊断  │  │ Quiz界面  │  │ 文档上传    │  │
│  └────┬─────┘  └──────┬───────┘  └────┬─────┘  └─────┬──────┘  │
└───────┼───────────────┼───────────────┼───────────────┼─────────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Orchestrator (意图路由 + 结果汇总)              │   │
│  │                        │                                  │   │
│  │         ┌──────────────┼──────────────┐                   │   │
│  │         ▼              ▼              ▼                   │   │
│  │  ┌─────────────┐ ┌──────────┐ ┌──────────────┐           │   │
│  │  │ QA SubAgent │ │  Quiz    │ │ Gap Analysis │           │   │
│  │  │ (自治)      │ │ SubAgent │ │  SubAgent    │           │   │
│  │  │ 简单/多跳   │ │ (自治)   │ │  (自治)      │           │   │
│  │  └─────────────┘ └──────────┘ └──────────────┘           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Ingestion Pipeline (非Agent, 固定流程)            │   │
│  │  文档解析 → 层级分块 → 向量化 → Milvus                    │   │
│  │         └→ 概念抽取(从Heading提取topic) → knowledge_concepts│   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Embedding    │  │ Reranker     │  │ Document Processing  │   │
│  │ Service      │  │ Service      │  │ (Notion/Word/PDF)    │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
└─────────┼─────────────────┼─────────────────────┼───────────────┘
          │                 │                     │
          ▼                 ▼                     ▼
┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   Milvus     │  │ BGE-m3 (1024-dim) │  │    Notion API    │
│ (向量存储)    │  │ BGE-reranker-v2  │  │   (数据源)        │
└──────────────┘  └──────────────────┘  └──────────────────┘
                  ┌──────────────────┐
                  │   SQLite          │
                  │ (用户记忆/掌握度)  │
                  └──────────────────┘
```

## 2. Agent 分层架构

```
Orchestrator (Plan & Execute with Replan, LangGraph StateGraph)
  │  LLM few-shot 意图识别 → 生成执行计划 → 单步执行 → Replan(continue/revise/finish) → 汇总结果
  │
  ├── QA SubAgent (被 Orchestrator 调度)
  │     自行判断问题复杂度:
  │     ├── 简单问题 → 单次 Retrieve + Rerank → Generate
  │     └── 复杂问题 → Decompose → 多轮 Retrieve + Rerank → Synthesize
  │     自行判断结果是否充分, 不充分则追加检索
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

## 3. LangGraph 图结构详细设计

### 3.1 节点一览

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

### 3.2 顶层流程 (Plan & Execute with Replan)

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

### 3.3 QA SubAgent 内部流程

```
入口 → Complexity Judge
  ├── 简单 → RAG Search → Reranker → Generate Answer → Sufficiency Check
  │                                                       ├── 充分 → 返回 Orchestrator
  │                                                       └── 不充分 → RAG Search (追加检索)
  └── 复杂 → Decompose Query → [对每个子查询: RAG Search → Reranker] → Synthesize → 返回 Orchestrator
```

### 3.4 Quiz SubAgent 内部流程

```
入口 → Load Memory → Select Question → (等待用户作答) → Evaluate Answer → Quiz Router
  ├── 答错 → 同 topic 再出 → Select Question
  ├── 答对且未结束 → Select Question
  └── 答对且结束 / 用户主动结束 → Update Memory → 返回 Orchestrator
```

### 3.5 Gap Analysis SubAgent 内部流程

```
入口 → Load Mastery → Search Concepts → Cross Compare → Generate Report → 返回 Orchestrator
```

### 3.6 完整流程图

```
                            ┌─────────────┐
                            │  用户输入    │
                            └──────┬──────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │    Orchestrator      │
                        │  意图路由 + 结果汇总  │
                        └───┬───────┬───────┬──┘
                            │       │       │
              intent=qa     │       │       │  intent=gap_analysis
                            ▼       │       ▼
    ┌─ QA SubAgent ─────────────┐   │   ┌─ Gap Analysis SubAgent ──────┐
    │                           │   │   │                              │
    │  ┌───────────────────┐    │   │   │  ┌────────────┐              │
    │  │ Complexity Judge  │    │   │   │  │Load Mastery│              │
    │  │ 简单 or 复杂?     │    │   │   │  └─────┬──────┘              │
    │  └──┬────────────┬───┘    │   │   │        ▼                     │
    │     │            │        │   │   │  ┌──────────────┐            │
    │     ▼            ▼        │   │   │  │Search Concepts│           │
    │  [简单路径]   [复杂路径]   │   │   │  └─────┬────────┘            │
    │     │            │        │   │   │        ▼                     │
    │     ▼            ▼        │   │   │  ┌─────────────┐             │
    │  RAG Search   Decompose   │   │   │  │Cross Compare│             │
    │     │         Query       │   │   │  └─────┬───────┘             │
    │     ▼            │        │   │   │        ▼                     │
    │  Reranker        ▼        │   │   │  ┌───────────────┐           │
    │     │       Multi-RAG     │   │   │  │Generate Report│           │
    │     ▼       Search+Rerank │   │   │  └─────┬─────────┘           │
    │  Generate        │        │   │   │        │                     │
    │  Answer          ▼        │   │   └────────┼─────────────────────┘
    │     │        Synthesize   │   │            │
    │     ▼            │        │   │            │
    │  ┌──────────┐    │        │   │            │
    │  │Sufficiency│   │        │   │            │
    │  │Check     │    │        │   │            │
    │  └──┬───┬───┘    │        │   │            │
    │  充分│   │不充分   │        │   │            │
    │     │   └→ RAG   │        │   │            │
    │     │   Search↑  │        │   │            │
    └─────┼────────────┼────────┘   │            │
          │            │            │            │
          ▼            ▼            │            ▼
        ┌─────────────────────┐     │  ┌─────────────────────┐
        │    Orchestrator     │◄────┘  │    Orchestrator      │
        └─────────────────────┘        └──────────────────────┘
                                  ▲
              intent=quiz         │
                            ┌─────┘
    ┌─ Quiz SubAgent ───────────────────────────────────────┐
    │                                                       │
    │  ┌───────────┐    ┌─────────────┐    ┌─────────────┐  │
    │  │Load Memory│──▶│Select Question│──▶│等待用户作答  │  │
    │  │+时间衰减   │    │w=1-mastery   │    └──────┬──────┘  │
    │  └───────────┘    └──────▲───▲──┘           │          │
    │                         │   │               ▼          │
    │                         │   │     ┌─────────────────┐  │
    │                         │   │     │Evaluate Answer  │  │
    │                         │   │     │LLM语义判分      │  │
    │                         │   │     └────────┬────────┘  │
    │                         │   │              │           │
    │                         │   │              ▼           │
    │                         │   │     ┌──────────────┐     │
    │                         │   │     │ Quiz Router  │     │
    │                         │   │     └──┬─────┬─────┘     │
    │              答错同topic │   │答对    │     │           │
    │              再出一题    │   │继续    │     │结束       │
    │                         │   └────────┘     │           │
    │                         │                  ▼           │
    │                         │        ┌──────────────┐      │
    │                         │        │题目够了?     │      │
    │                         │        └──┬───────┬───┘      │
    │                     否  │           │       │ 是       │
    │                         └───────────┘       │          │
    │                                             ▼          │
    │                                    ┌──────────────┐    │
    │                                    │Update Memory │    │
    │                                    └──────┬───────┘    │
    └───────────────────────────────────────────┼────────────┘
                                                │
                                                ▼
                                     ┌─────────────────────┐
                                     │    Orchestrator      │
                                     └──────────┬──────────┘
                                                │
                                                ▼
                                        ┌──────────────┐
                                        │  返回结果     │
                                        └──────────────┘
```

## 4. Agent Tool 定义与分配

### 4.1 Tool 清单

| Tool | 功能 | 输入 | 输出 |
|------|------|------|------|
| `rag_search` | Dense+Sparse混合检索(RRF融合) + Rerank，返回相关文档片段 | query, top_k, expr(可选) | list[RankedChunk] (text, rerank_score, source_doc, heading_path, child_id, parent_id, chunk_index) |
| `read_mastery` | 读取用户掌握度（含时间衰减计算） | user_id, topics(可选) | topic + effective_mastery 列表 |
| `update_mastery` | 三档质量评估更新 mastery_score + last_reviewed | user_id, topic, quality("low"\|"medium"\|"high") | 更新后的 mastery 记录 |
| `search_concepts` | 查询知识库概念索引表 | scope/keyword(可选) | topic + heading_path 列表 |
| `query_decompose` | 将复杂问题拆解为子查询 | original_query | 子查询列表 |

### 4.2 Agent × Tool 分配

| Agent | 可用 Tools | 说明 |
|-------|-----------|------|
| Orchestrator | 无 | 只做意图路由 + 结果汇总 |
| QA SubAgent | `rag_search`, `query_decompose` | ReAct风格，自行决定是否需要拆解和多轮检索 |
| Quiz SubAgent | `read_mastery`, `update_mastery`, `rag_search`, `search_concepts` | 读写掌握度 + 检索出题素材 + 获取 topic 列表 |
| Gap Analysis SubAgent | `read_mastery`, `search_concepts` | 读取掌握度 + 知识库概念列表，交叉对比 |

### 4.3 设计说明

- **Reranker 不作为独立 Tool**：内嵌在 `rag_search` 中，对外暴露完整的"混合检索+RRF融合+重排"能力
- **LLM Generate / LLM Judge 不作为 Tool**：这是 Agent 自身的推理能力，通过 prompt 控制
- **Sufficiency Check / Complexity Judge 不作为 Tool**：属于 Agent 的推理判断，用 prompt 引导
- **Gap Analysis 的 Cross Compare 不作为 Tool**：纯规则逻辑，写在 Agent prompt 或 node 代码中

### 4.4 Gap Analysis Cross Compare 规则

| 情况 | 判定 | 优先级 |
|------|------|--------|
| knowledge_concepts 中有，但 user_mastery 中无记录 | **未接触盲区** | 高 |
| 有记录，effective_mastery < 0.3 | **薄弱盲区** | 高 |
| 有记录，effective_mastery 0.3~0.5 | **需巩固** | 中 |
| effective_mastery ≥ 0.5 | 掌握中/已掌握 | 低 |

Generate Report 时将结构化结果交给 LLM，生成自然语言诊断报告 + 推荐复习顺序。

## 5. 数据流与存储设计

```
═══════════════════════════════════════════════════════════════
  Ingestion Pipeline (非Agent, 固定流程)
═══════════════════════════════════════════════════════════════

  ┌───────────┐   ┌──────────────┐
  │ Notion API│   │Word/PDF上传  │
  └─────┬─────┘   └──────┬───────┘
        │                │
        └───────┬────────┘
                ▼
    ┌───────────────────────┐
    │   Document Processor  │
    │  层级分块 Heading-     │
    │  aware Split          │
    └───────────┬───────────┘
                │ parent chunk + child chunks
                ▼
    ┌───────────────────────┐
    │   BGE Embedding       │
    │  对 child chunks 编码  │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │      Milvus           │
    │  存储 child chunks     │
    │  (仅child有向量,       │
    │   parent_id+chunk_index│
    │   保留相邻扩展能力)     │
    └───────────────────────┘


═══════════════════════════════════════════════════════════════
  检索链路 (被 QA SubAgent / Gap Analysis SubAgent 调用)
═══════════════════════════════════════════════════════════════

  用户Query ──▶ BGE-m3 Encode (dense + sparse)
                    │                    │
                    ▼                    ▼
              Milvus Dense          Milvus Sparse
              ANN Search            Inverted Search
                    │                    │
                    └───── RRF 融合 ─────┘
                             │
                             ▼
                    Top-K×3 candidates
                             │
                             ▼
                    Reranker (bge-reranker-v2-m3)
                             │
                             ▼
                    Top-K reranked chunks (含原文)
                                        │
                                        ▼
                                   Qwen3-Plus 生成答案


═══════════════════════════════════════════════════════════════
  用户记忆 (SQLite)
═══════════════════════════════════════════════════════════════

  ┌──────────┐  读取mastery + 时间衰减  ┌────────────────┐
  │  SQLite  │─────────────────────────▶│ Quiz SubAgent  │
  │          │◀─────────────────────────│                │
  └────┬─────┘  更新mastery+last_reviewed└────────────────┘
       │
       │  读取mastery
       ▼
  ┌──────────────────┐
  │Gap Analysis      │
  │SubAgent          │
  └──────────────────┘
```

## 6. Mastery 时间衰减公式

```
effective_mastery = mastery_score × exp(-λ × Δt)

其中:
  mastery_score  = 上次复习后的原始掌握度 [0, 1]
  λ (decay_rate) = 衰减系数，默认0.05
  Δt             = 距上次复习的天数 (now - last_reviewed).days

出题权重:
  weight = 1 - effective_mastery

示例:
  mastery=0.8, 10天未复习 → effective = 0.8 × e^(-0.5) ≈ 0.49 → weight=0.51
  mastery=0.8, 1天未复习  → effective = 0.8 × e^(-0.05) ≈ 0.76 → weight=0.24

更新规则 (三档质量评估):
  高质量 (high):   mastery_score = min(1.0, effective_mastery + 0.1)
  中等质量 (medium): mastery_score = min(1.0, effective_mastery + 0.05)
  低质量 (low):    mastery_score = max(0.0, effective_mastery - 0.15)

  medium/high 计入 correct_count, low 不计入
  每次更新后 last_reviewed = now, 重置衰减起点
```

## 7. SQLite 表结构

```sql
-- 知识点掌握度（带时间衰减）
CREATE TABLE user_mastery (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    topic       TEXT NOT NULL,
    mastery_score REAL DEFAULT 0.3,     -- 原始掌握度 [0, 1]
    decay_rate  REAL DEFAULT 0.05,      -- 衰减系数 λ
    total_attempts INTEGER DEFAULT 0,
    correct_count  INTEGER DEFAULT 0,
    last_reviewed  TIMESTAMP,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, topic)
);

-- Quiz会话记录
CREATE TABLE quiz_sessions (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    started_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at    TIMESTAMP,
    total_questions INTEGER DEFAULT 0,
    correct_count   INTEGER DEFAULT 0,
    topics_covered  TEXT               -- JSON数组
);

-- 知识库概念索引（Ingestion时从文档Heading提取）
CREATE TABLE knowledge_concepts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    topic       TEXT NOT NULL UNIQUE,   -- 概念/主题名
    source_doc  TEXT,                   -- 来源文档
    heading_path TEXT,                  -- 层级路径，如 "NLP/Transformer/Attention"
    chunk_count INTEGER DEFAULT 1,      -- 该topic下的chunk数量
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 8. 项目目录结构

```
SecondBrain/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI入口
│   │   ├── config.py               # 配置管理
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── chat.py         # QA问答API
│   │   │   │   ├── quiz.py         # Quiz API
│   │   │   │   ├── gap.py          # 知识盲区诊断API
│   │   │   │   └── ingest.py       # 文档摄入API
│   │   │   └── deps.py             # 依赖注入
│   │   ├── agent/
│   │   │   ├── graph.py            # LangGraph主图 (Orchestrator)
│   │   │   ├── subagents/
│   │   │   │   ├── qa_agent.py     # QA SubAgent (含复杂度判断+多跳)
│   │   │   │   ├── quiz_agent.py   # Quiz SubAgent
│   │   │   │   └── gap_agent.py    # Gap Analysis SubAgent
│   │   │   ├── tools/
│   │   │   │   ├── rag_search.py       # 向量检索+Rerank tool
│   │   │   │   ├── query_decompose.py  # 查询拆解 tool
│   │   │   │   ├── read_mastery.py     # 读取用户掌握度 tool
│   │   │   │   ├── update_mastery.py   # 更新用户掌握度 tool
│   │   │   │   └── search_concepts.py  # 知识库概念索引查询 tool
│   │   │   ├── nodes/
│   │   │   │   ├── orchestrator.py
│   │   │   │   ├── quiz_select.py
│   │   │   │   ├── quiz_evaluate.py
│   │   │   │   └── memory_manager.py
│   │   │   └── state.py            # LangGraph State定义
│   │   ├── services/
│   │   │   ├── embedding.py        # BGE Embedding封装
│   │   │   ├── reranker.py         # BGE Reranker封装
│   │   │   ├── milvus_client.py    # Milvus操作
│   │   │   ├── llm_client.py       # Qwen3-Plus调用
│   │   │   ├── notion_loader.py    # Notion数据拉取
│   │   │   └── doc_processor.py    # 文档解析+层级分块
│   │   └── db/
│   │       ├── database.py         # SQLite连接管理
│   │       ├── models.py           # ORM模型
│   │       └── crud.py             # CRUD操作
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatPanel.tsx       # QA问答
│   │   │   ├── QuizPanel.tsx       # Quiz复习
│   │   │   ├── GapAnalysis.tsx     # 知识盲区诊断
│   │   │   └── FileUpload.tsx      # 文档上传
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── services/api.ts
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts
├── docker-compose.yml              # Milvus + Backend
└── README.md
```

## 9. 实现流程（自底向上）

```
Step 1: Services 层（基础能力封装）
├── embedding.py          ← 所有检索链路的前置
├── reranker.py           ← rag_search tool 的前置
├── llm_client.py         ← 所有 Agent 的前置
└── doc_processor.py      ← Ingestion 的前置
        │
        ▼
Step 2: Ingestion Pipeline（有数据才能测后续所有模块）
├── notion_loader.py      ← 拉取原始文档
├── doc_processor.py      ← 层级分块 + 概念抽取
├── 向量化写入 Milvus
└── 概念写入 knowledge_concepts 表
        │
        ▼
Step 3: Tools 层（Agent 的能力单元，可独立单测）
├── rag_search.py         ← 依赖 embedding + milvus + reranker
├── query_decompose.py    ← 依赖 llm_client
├── read_mastery.py       ← 依赖 SQLite
├── update_mastery.py     ← 依赖 SQLite
└── search_concepts.py    ← 依赖 SQLite knowledge_concepts
        │
        ▼
Step 4: SubAgents（三个互相独立，可并行开发）
├── qa_agent.py           ← 绑定 rag_search + query_decompose
├── quiz_agent.py         ← 绑定 read_mastery + update_mastery
└── gap_agent.py          ← 绑定 read_mastery + search_concepts
        │
        ▼
Step 5: Orchestrator（意图路由 + 分发 + 汇总）
└── graph.py              ← 依赖三个 SubAgent
        │
        ▼
Step 6: API 层
├── chat.py               ← 对接 Orchestrator (QA)
├── quiz.py               ← 对接 Orchestrator (Quiz)
├── gap.py                ← 对接 Orchestrator (Gap)
└── ingest.py             ← 对接 Ingestion Pipeline
        │
        ▼
Step 7: Frontend
└── ChatPanel / QuizPanel / GapAnalysis / FileUpload
        │
        ▼
Step 8: 评估 & 优化
```

### 当前进度

- [x] FastAPI 项目骨架
- [x] Milvus Standalone 部署 (docker-compose)
- [x] SQLite 数据库初始化
- [x] Step 1: Services 层 (embedding + reranker + llm_client)
- [ ] Step 2: Ingestion Pipeline (milvus_client 已升级支持 hybrid search)
- [x] Step 3: Tools 层 (rag_search + query_decompose + read_mastery + update_mastery + search_concepts)
- [x] Step 4: SubAgents (state.py + qa_agent + quiz_agent + gap_agent)
- [x] Step 5: Orchestrator (Plan & Execute, QA+Gap 编排, Quiz 独立)
- [ ] Step 6: API 层
- [ ] Step 7: Frontend
- [ ] Step 8: 评估 & 优化

## 10. 评估指标体系

### 10.1 检索模块 (RAG + Reranker)

| 指标 | 定义 | 计算方式 | 目标 |
|------|------|----------|------|
| **Recall@K** | Top-K检索结果中包含正确文档的比例 | 命中查询数 / 总查询数 | ≥ 0.85 |
| **MRR** (Mean Reciprocal Rank) | 正确文档在结果中排名的倒数均值 | Σ(1/rank_i) / N | ≥ 0.70 |
| **Reranker Lift** | Reranker相对纯向量检索的MRR提升 | MRR_reranked - MRR_raw | > 0 |
| **Retrieval Latency (P95)** | 95分位检索延迟 | 从query到返回Top-K的耗时 | < 500ms |

### 10.2 生成模块 (知识问答)

| 指标 | 定义 | 计算方式 | 目标 |
|------|------|----------|------|
| **Faithfulness** | 生成答案是否忠于检索到的源文档 | LLM-as-Judge: 每个claim是否有source支撑 | ≥ 0.90 |
| **Answer Relevance** | 答案是否切题回答了用户问题 | LLM-as-Judge: 相关性评分 [1-5] | ≥ 4.0 |
| **Citation Accuracy** | 引用的来源是否真实且匹配 | source_doc是否确实包含对应内容 | ≥ 0.85 |

### 10.3 多跳QA (Multi-hop)

| 指标 | 定义 | 计算方式 | 目标 |
|------|------|----------|------|
| **Decomposition Quality** | 子查询是否覆盖原始问题的各个方面 | LLM-as-Judge: 子查询完备性评分 | ≥ 0.80 |
| **Synthesis Coherence** | 综合答案是否逻辑连贯、无矛盾 | LLM-as-Judge: 连贯性评分 [1-5] | ≥ 4.0 |
| **Multi-hop Lift** | 多跳相对单跳在复杂问题上的答案质量提升 | 复杂问题集上 Faithfulness 差值 | > 0 |
| **Avg Sub-queries** | 平均子查询数 | 总子查询数 / 复杂问题数 | 2-4 |

### 10.4 Quiz自适应复习模块

| 指标 | 定义 | 计算方式 | 目标 |
|------|------|----------|------|
| **Answer Eval Accuracy** | LLM判分与人工判分的一致率 | 抽样验证一致的比例 | ≥ 0.90 |
| **Mastery Convergence** | 掌握度是否随复习次数合理收敛 | 多次session后mastery变化曲线 | 单调趋升 |
| **Topic Coverage** | 每次quiz覆盖的topic多样性 | unique_topics / total_questions | ≥ 0.60 |
| **Adaptive Accuracy** | 加权选题是否有效 | corr(weight, selection_freq) | 正相关 |
| **Decay Effectiveness** | 时间衰减是否促进了间隔复习 | 长时间未复习topic被选中率 | 高于无衰减基线 |

### 10.5 Knowledge Gap Analysis 模块

| 指标 | 定义 | 计算方式 | 目标 |
|------|------|----------|------|
| **Gap Detection Precision** | 识别出的盲区是否确实是用户薄弱点 | 人工验证正确数 / 总识别数 | ≥ 0.80 |
| **Gap Detection Recall** | 用户实际薄弱点中被识别出的比例 | 识别数 / 人工标注的实际薄弱点数 | ≥ 0.70 |
| **Recommendation Relevance** | 推荐的复习顺序是否合理 | LLM-as-Judge + 人工抽检 | ≥ 0.80 |

### 10.6 端到端指标

| 指标 | 定义 | 目标 |
|------|------|------|
| **E2E Latency (P95)** | 用户输入到收到首token的延迟 | < 2s (流式) |
| **Intent Classification Acc** | Orchestrator意图分类准确率 | ≥ 0.95 |
| **Session Completion Rate** | Quiz session正常完成的比例 | ≥ 0.95 |
| **Complexity Judge Acc** | QA SubAgent复杂度判断准确率 | ≥ 0.85 |
