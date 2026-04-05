# 数据流与存储设计

## 概要

三条主要数据流：Ingestion Pipeline（文档→分块→BGE编码→Milvus）、检索链路（Query→dense+sparse双路→RRF融合→Rerank→生成）、用户记忆（SQLite mastery 读写，含时间衰减）。

---

## 详细设计

### Ingestion Pipeline

```
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
```

### 检索链路

```
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
```

### 用户记忆数据流

```
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

### Mastery 时间衰减公式

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

更新规则 (二元判分):
  答对 (correct):   mastery_score = min(1.0, effective_mastery + 0.1)
  答错 (incorrect): mastery_score = max(0.0, effective_mastery - 0.15)

  correct 计入 correct_count, incorrect 不计入
  每次更新后 last_reviewed = now, 重置衰减起点
```

### SQLite 表结构

```sql
-- 知识点掌握度（带时间衰减）
CREATE TABLE user_mastery (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    topic       TEXT NOT NULL,
    mastery_score REAL DEFAULT 0.3,
    decay_rate  REAL DEFAULT 0.05,
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
    topics_covered  TEXT
);

-- 短期对话记录（FIFO 淘汰）
CREATE TABLE conversation_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 知识库概念索引（Ingestion时从文档Heading提取）
CREATE TABLE knowledge_concepts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    topic       TEXT NOT NULL UNIQUE,
    source_doc  TEXT,
    heading_path TEXT,
    chunk_count INTEGER DEFAULT 1,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
