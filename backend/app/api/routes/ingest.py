from fastapi import APIRouter

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/notion")
async def ingest_from_notion():
    """从Notion拉取并摄入文档"""
    return {"message": "notion ingest endpoint placeholder"}


@router.post("/upload")
async def upload_document():
    """上传Word/PDF文档"""
    return {"message": "document upload endpoint placeholder"}
