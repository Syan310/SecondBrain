from fastapi import APIRouter

router = APIRouter(prefix="/gap", tags=["gap-analysis"])


@router.post("/analyze")
async def analyze_knowledge_gap():
    """知识盲区诊断"""
    return {"message": "gap analysis endpoint placeholder"}
