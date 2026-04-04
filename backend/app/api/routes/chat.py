from fastapi import APIRouter

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/")
async def chat():
    """知识查找 - RAG问答"""
    return {"message": "chat endpoint placeholder"}
