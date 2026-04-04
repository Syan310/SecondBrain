"""Ingestion API: upload document → parse → embed → write to Milvus + SQLite."""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.db.crud import delete_concepts_by_source, upsert_concepts
from app.db.database import async_session_factory
from app.services.doc_processor import doc_processor
from app.services.embedding import embedding_service
from app.services.milvus_client import milvus_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".md", ".markdown"}


@router.post("/upload")
async def upload_document(file: UploadFile):
    """Upload a document, parse → chunk → embed → store in Milvus + knowledge_concepts."""
    # Validate file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {_ALLOWED_EXTENSIONS}",
        )

    # Save to temp file for processing
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 1. Parse & chunk
        doc = doc_processor.process_file(tmp_path)
        # Override source name to original filename
        doc.source = file.filename

        all_children = [child for parent in doc.parents for child in parent.children]
        if not all_children:
            raise HTTPException(status_code=400, detail="No content extracted from document.")

        # 2. Embed child chunks (dense + sparse)
        texts = [child.text for child in all_children]
        embed_out = embedding_service.encode_documents(texts)
        embeddings = embed_out.dense.tolist()
        sparse_embeddings = embed_out.sparse

        # 3. Delete old data for this source (re-ingestion)
        milvus_service.delete_by_source(file.filename)
        async with async_session_factory() as session:
            await delete_concepts_by_source(session, file.filename)

        # 4. Insert into Milvus
        count = milvus_service.insert(
            parent_ids=[child.parent_id for child in all_children],
            source_docs=[file.filename] * len(all_children),
            heading_paths=[child.heading_path for child in all_children],
            chunk_indices=[child.chunk_index for child in all_children],
            texts=texts,
            embeddings=embeddings,
            sparse_embeddings=sparse_embeddings,
        )

        # 5. Upsert concepts into SQLite
        # Update source_doc to original filename
        for concept in doc.concepts:
            concept.source_doc = file.filename

        async with async_session_factory() as session:
            new_concepts = await upsert_concepts(session, doc.concepts)

        logger.info(
            "Ingested '%s': %d chunks → Milvus, %d new concepts → SQLite",
            file.filename, count, new_concepts,
        )
        return {
            "filename": file.filename,
            "parents": len(doc.parents),
            "chunks_inserted": count,
            "concepts_total": len(doc.concepts),
            "concepts_new": new_concepts,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)
