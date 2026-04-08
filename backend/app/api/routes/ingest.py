"""Ingestion API: upload document → parse → embed → write to Milvus + SQLite."""

import asyncio
import logging
import re
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.api.schemas import (
    DocumentInfo, DocumentListResponse, MultiUploadResult, MultiUploadResponse,
)
from app.config import settings
from app.db.crud import (
    delete_concepts_by_source, get_all_concept_topics, list_documents, upsert_concepts,
)
from app.services.concept_extractor import extract_concepts
from app.services.doc_processor import doc_processor
from app.services.embedding import embedding_service
from app.services.milvus_client import milvus_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".md", ".markdown"}
_FILENAME_MAX_LEN = 255
_SAFE_FILENAME_RE = re.compile(r"[^\w\s\-\.\(\)\[\]（）\u4e00-\u9fff]", re.UNICODE)


def _build_embed_texts(children: list) -> list[str]:
    """Build texts for embedding: prepend heading_path to improve semantic matching."""
    return [
        f"{c.heading_path}\n{c.text}" if c.heading_path else c.text
        for c in children
    ]


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


async def _validate_and_save_temp(file: UploadFile) -> tuple[str, str]:
    """Validate a single upload file and save to a temp path.

    Returns:
        (sanitized_filename, tmp_path)

    Raises:
        ValueError with message on validation failure.
    """
    if not file.filename:
        raise ValueError("Filename is required.")

    filename = _sanitize_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}. Allowed: {_ALLOWED_EXTENSIONS}")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise ValueError(f"File too large. Max size: {settings.max_upload_size_mb} MB.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        return filename, tmp.name


def _parse_file(tmp_path: str, filename: str):
    """Parse & chunk a single file (CPU-bound, runs in thread).

    Returns:
        (doc, all_children) tuple.
    """
    doc = doc_processor.process_file(tmp_path)
    doc.source = filename
    all_children = [child for parent in doc.parents for child in parent.children]
    return doc, all_children


async def _extract_concepts_for_doc(
    parents: list,
    filename: str,
    db_session: AsyncSession,
) -> tuple[int, int]:
    """Extract concepts via LLM for each parent chunk, upsert to DB.

    Returns:
        (total_concepts_found, new_concepts_inserted)
    """
    existing = await get_all_concept_topics(db_session)
    total_found = 0
    total_new = 0

    for parent in parents:
        topics = await extract_concepts(parent.full_text, existing)
        if not topics:
            continue

        total_found += len(topics)
        new_count = await upsert_concepts(
            db_session, topics, source_doc=filename, parent_chunk_id=parent.id,
        )
        total_new += new_count

        # Update running list so next chunk can deduplicate
        for t in topics:
            if t not in existing:
                existing.append(t)

    return total_found, total_new


async def _ingest_one(
    filename: str,
    tmp_path: str,
    db_session: AsyncSession,
) -> MultiUploadResult:
    """Full ingestion pipeline for a single file: parse → embed → store."""
    try:
        # 1. Parse & chunk (parallel-safe: CPU work in thread)
        doc, all_children = await asyncio.to_thread(_parse_file, tmp_path, filename)

        if not all_children:
            return MultiUploadResult(filename=filename, success=False, error="No content extracted.")

        # 2. Embed (serial — shared model, called sequentially by caller)
        texts = [child.text for child in all_children]
        embed_texts = _build_embed_texts(all_children)
        embed_out = embedding_service.encode_documents(embed_texts)
        embeddings = embed_out.dense.tolist()
        sparse_embeddings = embed_out.sparse

        # 3. Delete old data for re-ingestion
        milvus_service.delete_by_source(filename)
        await delete_concepts_by_source(db_session, filename)

        # 4. Insert into Milvus (text stores original, embedding uses heading-augmented)
        count = milvus_service.insert(
            parent_ids=[child.parent_id for child in all_children],
            source_docs=[filename] * len(all_children),
            heading_paths=[child.heading_path for child in all_children],
            chunk_indices=[child.chunk_index for child in all_children],
            texts=texts,
            embeddings=embeddings,
            sparse_embeddings=sparse_embeddings,
        )

        # 5. LLM concept extraction per parent chunk
        concepts_total, concepts_new = await _extract_concepts_for_doc(
            doc.parents, filename, db_session,
        )

        logger.info(
            "Ingested '%s': %d chunks → Milvus, %d concepts found (%d new) → SQLite",
            filename, count, concepts_total, concepts_new,
        )
        return MultiUploadResult(
            filename=filename,
            success=True,
            parents=len(doc.parents),
            chunks_inserted=count,
            concepts_total=concepts_total,
            concepts_new=concepts_new,
        )

    except Exception as e:
        logger.error("Ingestion failed for '%s': %s", filename, e)
        return MultiUploadResult(filename=filename, success=False, error=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/upload", response_model=MultiUploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    session: AsyncSession = Depends(get_db),
):
    """Upload one or more documents: parallel parse → serial embed + write."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # 1. Validate & save all files to temp (fast I/O)
    validated: list[tuple[str, str]] = []
    early_errors: list[MultiUploadResult] = []
    for file in files:
        try:
            filename, tmp_path = await _validate_and_save_temp(file)
            validated.append((filename, tmp_path))
        except ValueError as e:
            early_errors.append(MultiUploadResult(
                filename=file.filename or "unknown", success=False, error=str(e),
            ))

    # 2. Parallel parse
    parse_tasks = [
        asyncio.to_thread(_parse_file, tmp_path, filename)
        for filename, tmp_path in validated
    ]
    parse_results = await asyncio.gather(*parse_tasks, return_exceptions=True)

    # 3. Serial embed + write (one by one to avoid OOM)
    results: list[MultiUploadResult] = list(early_errors)
    for (filename, tmp_path), parsed in zip(validated, parse_results):
        if isinstance(parsed, Exception):
            results.append(MultiUploadResult(
                filename=filename, success=False, error=str(parsed),
            ))
            Path(tmp_path).unlink(missing_ok=True)
            continue

        doc, all_children = parsed

        if not all_children:
            results.append(MultiUploadResult(
                filename=filename, success=False, error="No content extracted.",
            ))
            Path(tmp_path).unlink(missing_ok=True)
            continue

        try:
            texts = [child.text for child in all_children]
            embed_texts = _build_embed_texts(all_children)
            embed_out = embedding_service.encode_documents(embed_texts)
            embeddings = embed_out.dense.tolist()
            sparse_embeddings = embed_out.sparse

            milvus_service.delete_by_source(filename)
            await delete_concepts_by_source(session, filename)

            count = milvus_service.insert(
                parent_ids=[child.parent_id for child in all_children],
                source_docs=[filename] * len(all_children),
                heading_paths=[child.heading_path for child in all_children],
                chunk_indices=[child.chunk_index for child in all_children],
                texts=texts,
                embeddings=embeddings,
                sparse_embeddings=sparse_embeddings,
            )

            concepts_total, concepts_new = await _extract_concepts_for_doc(
                doc.parents, filename, session,
            )

            logger.info(
                "Ingested '%s': %d chunks → Milvus, %d concepts found (%d new) → SQLite",
                filename, count, concepts_total, concepts_new,
            )
            results.append(MultiUploadResult(
                filename=filename,
                success=True,
                parents=len(doc.parents),
                chunks_inserted=count,
                concepts_total=concepts_total,
                concepts_new=concepts_new,
            ))
        except Exception as e:
            logger.error("Ingestion failed for '%s': %s", filename, e)
            results.append(MultiUploadResult(
                filename=filename, success=False, error=str(e),
            ))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return MultiUploadResponse(results=results)


@router.get("/documents", response_model=DocumentListResponse)
async def get_documents(session: AsyncSession = Depends(get_db)):
    """列出所有已上传的文档及其概念数量。"""
    rows = await list_documents(session)
    return DocumentListResponse(
        documents=[DocumentInfo(**row) for row in rows]
    )


@router.delete("/documents/{filename}")
async def delete_document(filename: str, session: AsyncSession = Depends(get_db)):
    """删除一个已上传的文档（Milvus chunks + SQLite concepts）。"""
    sanitized = _sanitize_filename(filename)
    milvus_service.delete_by_source(sanitized)
    deleted = await delete_concepts_by_source(session, sanitized)
    logger.info("Deleted document '%s': %d concepts removed.", sanitized, deleted)
    return {"filename": sanitized, "concepts_deleted": deleted}
