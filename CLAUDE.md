# SecondBrain

基于个人笔记的 RAG 问答、出题和知识盲区诊断系统。功能和接口见 `README.md`。

## 怎么跑

- Milvus：`docker compose up -d etcd minio milvus`（端口 19530）。
- 后端（Windows，在 `backend/` 下）：`.venv/Scripts/python -m uvicorn app.main:app --port 8000`。`backend/.env` 里要有 `MILVUS_HOST=localhost`。
- 前端：`cd frontend && npm run dev`（端口 5173）。
- 单元测试（在 `backend/` 下）：`.venv/Scripts/python -m pytest -q`。
- 独立脚本（例如 `evaluation/stats.py`）要加 `PYTHONPATH=.`。

## 技术栈

FastAPI + LangGraph，Milvus 2.6（BGE-M3 dense + BM25，RRF），SQLite，React 19 + Vite。应用模型由 `LLM_MODEL` 决定，评测用 `deepseek-flash`；评分模型是 OpenRouter 上的 Claude Haiku 5.5。

## 目录与约定

- `backend/app/agent/`：Orchestrator 和 SubAgent。问答走 `subagents/qa_agent.py`（ReAct + Reflection）。
- `backend/app/services/citations.py`：引用模式 `off`、`prompt`、`filter`、`fallback`。线上默认值是 `fallback`（`app/config.py`）。
- `backend/evaluation/`：评测管线。`eval_rag.py` 跑评测，`stats.py` 做配对 bootstrap。结果在 `results/`。
- `docs/`：任务文档（`task-*.md`）、状态文件（`status-*.md`）、评测报告（`eval-report-*.md`）。
- 评测规则：
  - 每次运行在命令里显式写出 `MILVUS_COLLECTION_NAME`、`ENABLE_RERANKER`、`RERANKER_USE_HEADING`、`CITATION_MODE`。
  - 正式运行开始后不改代码。`eval_rag.py` 用 `git status --untracked-files=no` 判断 `dirty`，所以改任何已跟踪的文件（包括文档）都会让结果变成 dirty。
  - 不修改集合 `secondbrain_docs`（L0，187 行）和题集 `evaluation/datasets/qa_l0_v1.json`。
- API key 不能出现在代码、日志、报告和提交信息里。
- 提交信息末尾加 `Co-Authored-By` 行。

## 当前状态

- 最近完成的是回答兜底任务（`docs/task-fallback-answer.md`）。报告是 `docs/eval-report-2026-11-03-fallback.md`。
- 已知问题：兜底回答的第一句有时是变体（例如「笔记里没有找到 RAPTOR 相关内容。」），`is_not_found()` 只认完整的固定句，所以判不出来。
- 已知问题：回答提示词里还有「禁止引入检索内容中未提及的……」，和兜底规则矛盾。用户决定这次不修。
