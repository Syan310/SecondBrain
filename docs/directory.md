# 项目目录结构

## 概要

FastAPI backend + React/Vite frontend + Docker Compose 基础设施。backend 分 api/agent/services/db 四层，agent 下含 subagents/tools/nodes/state。

---

## 详细结构

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
│   │   │   ├── Layout.tsx          # sidebar导航 + 内容区布局
│   │   │   ├── ChatPanel.tsx       # 统一对话 (QA/Quiz/Gap 三合一, Markdown/LaTeX)
│   │   │   ├── QuizModal.tsx       # Quiz MCQ 弹窗
│   │   │   └── FileUpload.tsx      # 文档管理 (上传+列表+删除)
│   │   ├── services/api.ts         # API客户端
│   │   ├── types.ts                # TypeScript类型定义
│   │   └── App.tsx                 # 两面板切换 (chat/upload)
│   ├── package.json
│   └── vite.config.ts              # Tailwind v4 + dev proxy
├── docker-compose.yml              # Milvus + Backend
└── README.md
```
