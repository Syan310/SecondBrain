"""Ingestion API: upload document → parse → embed → write to Milvus + SQLite."""

import logging
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.config import settings
from app.db.crud import delete_concepts_by_source, upsert_concepts
from app.services.doc_processor import doc_processor
from app.services.embedding import embedding_service
from app.services.milvus_client import milvus_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".md", ".markdown"}
_FILENAME_MAX_LEN = 255
_SAFE_FILENAME_RE = re.compile(r"[^\w\s\-\.\(\)\[\]（）\u4e00-\u9fff]", re.UNICODE)


def _sanitize_filename(raw: str) -> str:
    """Sanitize an uploaded filename: strip path components, limit length."""
    # Take only the basename (防止路径穿越)
    name = Path(raw).name
    # Remove potentially dangerous characters
    name = _SAFE_FILENAME_RE.sub("_", name)
    # Limit length
    if len(name) > _FILENAME_MAX_LEN:
        stem = Path(name).stem[:_FILENAME_MAX_LEN - 10]
        name = stem + Path(name).suffix
    return name


@router.post("/upload")
async def upload_document(file: UploadFile, session: AsyncSession = Depends(get_db)):
    """Upload a document, parse → chunk → embed → store in Milvus + knowledge_concepts."""
    # Validate filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    filename = _sanitize_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {_ALLOWED_EXTENSIONS}",
        )

    # Read with size limit
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_upload_size_mb} MB.",
        )

    # Save to temp file for processing
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 1. Parse & chunk
        doc = doc_processor.process_file(tmp_path)
        # Override source name to sanitized filename
        doc.source = filename

        all_children = [child for parent in doc.parents for child in parent.children]
        if not all_children:
            raise HTTPException(status_code=400, detail="No content extracted from document.")

        # 2. Embed child chunks (dense + sparse)
        texts = [child.text for child in all_children]
        embed_out = embedding_service.encode_documents(texts)
        embeddings = embed_out.dense.tolist()
        sparse_embeddings = embed_out.sparse

        # 3. Delete old data for this source (re-ingestion) — single session for atomicity
        milvus_service.delete_by_source(filename)
        await delete_concepts_by_source(session, filename)

        # 4. Insert into Milvus
        count = milvus_service.insert(
            parent_ids=[child.parent_id for child in all_children],
            source_docs=[filename] * len(all_children),
            heading_paths=[child.heading_path for child in all_children],
            chunk_indices=[child.chunk_index for child in all_children],
            texts=texts,
            embeddings=embeddings,
            sparse_embeddings=sparse_embeddings,
        )

        # 5. Upsert concepts into SQLite (same session as delete)
        for concept in doc.concepts:
            concept.source_doc = filename
        new_concepts = await upsert_concepts(session, doc.concepts)

        logger.info(
            "Ingested '%s': %d chunks → Milvus, %d new concepts → SQLite",
            filename, count, new_concepts,
        )
        return {
            "filename": filename,
            "parents": len(doc.parents),
            "chunks_inserted": count,
            "concepts_total": len(doc.concepts),
            "concepts_new": new_concepts,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ingestion failed for '%s': %s", filename, e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
