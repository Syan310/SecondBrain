from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.database import init_db
from app.api.routes import chat, quiz, gap, ingest
from app.services.milvus_client import milvus_service, memory_milvus_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    milvus_service.connect()
    milvus_service.init_collection()
    memory_milvus_service.init_collection()
    yield
    # Shutdown: 清理资源（预留）


app = FastAPI(
    title="SecondBrain API",
    description="基于个人笔记的RAG + Quiz + 知识盲区诊断系统",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS - 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为前端域名
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router, prefix="/api")
app.include_router(gap.router, prefix="/api")
app.include_router(quiz.router, prefix="/api")
app.include_router(ingest.router, prefix="/api")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
