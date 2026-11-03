# SecondBrain

A RAG + quiz + knowledge gap diagnosis system built on personal notes. Upload your study materials and let the AI answer questions, generate quizzes, and identify your weak points.

## Core Features

| Feature | Description |
| --- | --- |
| **Question Answering (QA)** | Document-grounded RAG QA for complex questions such as comparison, causality, and parallel analysis, using a ReAct + Reflection workflow. Answers cite chunk ids. If the notes have no answer, the reply starts with `笔记里没有找到相关内容。` and then answers from model knowledge (`CITATION_MODE=fallback`, the default). |
| **Quiz Generation** | Generates multiple-choice questions from document content and user weak points, then updates mastery in real time with a time-decay model. |
| **Knowledge Gap Diagnosis** | Compares knowledge base concepts with user mastery and produces a report with four levels: unseen, weak, needs reinforcement, and mastered. |
| **Document Management** | Supports PDF, Word, and Markdown upload, then automatically parses, chunks, extracts concepts, and writes vectors. |
| **Conversation Memory** | Uses short-term SQLite memory plus long-term Milvus memory with automatic consolidation and context injection. |

## Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | React 19 + Vite + TypeScript + Tailwind CSS v4 |
| Backend | FastAPI + LangGraph + Python 3.11 |
| Embedding | BGE-m3 dense (1024 dimensions, bilingual, heading-augmented input) |
| LLM | Qwen3-Plus (chat / stream / tool-calling) |
| Vector Store | Milvus Standalone v2.6: hybrid retrieval = BGE-m3 dense (HNSW) + BM25 (built-in Function, jieba analyzer), fused with RRF (k=60) |
| Database | SQLite (aiosqlite + SQLAlchemy Async, WAL mode) |
| PDF Parsing | magic-pdf (MinerU) |
| Deployment | Docker Compose (Milvus + etcd + MinIO + Backend) |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Node.js 18+ for local frontend development
- Local model files under `backend/models/`

### 1. Create `backend/.env`

This repository currently does not include `backend/.env.example`, so create `backend/.env` manually.

Minimum required:

```env
LLM_API_KEY=sk-xxxx
```

Recommended local settings:

```env
LLM_API_KEY=sk-ant-xxxx
LLM_BASE_URL=https://api.anthropic.com
LLM_MODEL=claude-opus-4-6

EMBEDDING_MODEL_PATH=./models/bge-m3
RERANKER_MODEL_PATH=./models/bge-reranker-v2-m3
ENABLE_RERANKER=false
CITATION_MODE=fallback

MILVUS_HOST=milvus
MILVUS_PORT=19530
SQLITE_DB_PATH=./data/secondbrain.db

APP_HOST=0.0.0.0
APP_PORT=8000
APP_ENV=development
```

### 2. Prepare Models

Place the following models under `backend/models/`:

```text
backend/models/
├── bge-m3/
└── bge-reranker-v2-m3/
```

Example download via Hugging Face:

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-m3', local_dir='backend/models/bge-m3')
snapshot_download('BAAI/bge-reranker-v2-m3', local_dir='backend/models/bge-reranker-v2-m3')
"
```

### 3. Start Backend Services with Docker Compose

```bash
docker compose --profile backend up -d
docker compose logs -f backend
```

After startup:

- Backend API: `http://localhost:18080`
- API Docs: `http://localhost:18080/docs`
- Health Check: `http://localhost:18080/health`
- Milvus: `localhost:19530`

### 4. Start the Frontend for Local Development

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

The frontend uses a Vite dev proxy, so `/api` requests go to `localhost:18080` by default.
If you want to run the backend locally instead of in Docker, set `VITE_PROXY_TARGET=http://localhost:8000` in `frontend/.env.local`.

### 5. Build the Frontend

```bash
cd frontend
npm run build
```

The build output is generated in `frontend/dist/`.

## Local Development Without Docker

### Start Milvus Only

```bash
docker compose up -d etcd minio milvus
```

### Start the Backend Locally

```bash
cd backend
pip install -r requirements.txt
export MILVUS_HOST=localhost
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Start the Frontend Locally

```bash
cd frontend
npm install
VITE_PROXY_TARGET=http://localhost:8000 npm run dev
```

## Main API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/chat/` | Main chat entry for QA, quiz triggering, and gap-analysis requests |
| `POST` | `/api/chat/stream` | Streaming chat response |
| `GET` | `/api/chat/history` | Load conversation history |
| `POST` | `/api/quiz/answer` | Submit a quiz answer and update mastery |
| `POST` | `/api/gap/analyze` | Run gap analysis directly |
| `POST` | `/api/ingest/upload` | Upload and ingest documents |
| `GET` | `/api/ingest/documents` | List uploaded documents |
| `DELETE` | `/api/ingest/documents/{filename}` | Delete a document and its indexed data |
| `GET` | `/health` | Service health check |

## Usage

1. Upload PDF, Word, or Markdown files in the document panel.
2. Ask a question in chat for document-grounded QA.
3. Type `quiz` to generate 2 MCQs and update mastery from your answers.
4. Ask for a knowledge gap analysis to get a diagnosis based on your materials and mastery history.

## Troubleshooting

- `backend/.env.example` not found:
  Create `backend/.env` manually. The repository does not currently ship an example file.
- Backend container does not start:
  Check `docker compose logs -f backend` and confirm `backend/models/` contains both BGE models.
- Backend cannot connect to Milvus in local mode:
  Set `MILVUS_HOST=localhost` before starting `uvicorn`.
- Upgrading from the Milvus 2.4 setup:
  The compose file now uses new volumes (`*_v26`), so 2.4 data is not reused. Re-upload your documents after the first start. When you no longer need the old data, remove the old volumes with `docker volume ls` and `docker volume rm`.
- API works but frontend requests fail:
  Check `VITE_PROXY_TARGET` and make sure the frontend is pointing to the same backend port you started.
- First request is slow:
  Initial model loading for embedding and reranker can take noticeable time, especially on CPU.

## Current Status

- Core QA, quiz, gap analysis, ingestion, and memory flows are implemented.
- Frontend and backend are both runnable locally.
- Evaluation tooling for RAG and agent performance exists under `backend/evaluation/`.
- `CITATION_MODE` values: `off` (no citations), `prompt` (strip markers), `filter` (drop uncited sentences; refuse if none left), `fallback` (default).
- Evaluation reports are in `docs/eval-report-*.md`. The latest is `docs/eval-report-2026-11-03-fallback.md`.
