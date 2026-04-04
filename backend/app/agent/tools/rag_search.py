"""RAG Search tool: hybrid retrieval (dense + sparse) + rerank.

Flow:
  1. Encode query → dense vector + sparse vector (BGE-m3)
  2. Milvus hybrid_search → top_k*3 candidates (RRF fusion)
  3. Reranker (bge-reranker-v2-m3) → top_k final results

Used by QA SubAgent as a LangGraph tool.
"""

import logging
import time
from dataclasses import dataclass

from app.services.embedding import embedding_service
from app.services.milvus_client import milvus_service
from app.services.reranker import reranker_service

logger = logging.getLogger(__name__)

# ── Data structures ──────────────────────────────────────────


@dataclass
class RankedChunk:
    """A reranked chunk returned by rag_search."""

    child_id: str
    parent_id: str
    source_doc: str
    heading_path: str
    chunk_index: int
    text: str
    rerank_score: float


# ── Tool implementation ──────────────────────────────────────


def rag_search(
    query: str,
    top_k: int = 5,
    expr: str | None = None,
) -> list[RankedChunk]:
    """Hybrid retrieval + rerank.

    Args:
        query: User query string.
        top_k: Number of final results after reranking.
        expr: Optional Milvus filter expression, e.g. 'source_doc == "xxx.pdf"'.

    Returns:
        List of RankedChunk sorted by rerank_score descending.
    """
    t0 = time.perf_counter()

    # Step 1: Encode query (dense + sparse)
    query_out = embedding_service.encode_query(query)
    dense_vector = query_out.dense.tolist()
    sparse_vector = query_out.sparse[0]  # single query → first element

    t_encode = time.perf_counter()

    # Step 2: Hybrid search — retrieve top_k * 3 candidates
    candidate_k = top_k * 3
    candidates = milvus_service.hybrid_search(
        query_vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=candidate_k,
        expr=expr,
    )

    t_search = time.perf_counter()

    if not candidates:
        logger.warning("rag_search: no candidates found for query: %s", query[:100])
        return []

    # Step 3: Rerank candidates
    passages = [c.text for c in candidates]
    rerank_results = reranker_service.rerank(
        query=query,
        passages=passages,
        top_n=top_k,
    )

    t_rerank = time.perf_counter()

    # Map rerank results back to full chunk metadata
    ranked_chunks = [
        RankedChunk(
            child_id=candidates[r.original_index].child_id,
            parent_id=candidates[r.original_index].parent_id,
            source_doc=candidates[r.original_index].source_doc,
            heading_path=candidates[r.original_index].heading_path,
            chunk_index=candidates[r.original_index].chunk_index,
            text=r.text,
            rerank_score=r.score,
        )
        for r in rerank_results
    ]

    logger.info(
        "rag_search: query=%s | candidates=%d | top_k=%d | "
        "encode=%.0fms search=%.0fms rerank=%.0fms total=%.0fms",
        query[:50],
        len(candidates),
        len(ranked_chunks),
        (t_encode - t0) * 1000,
        (t_search - t_encode) * 1000,
        (t_rerank - t_search) * 1000,
        (t_rerank - t0) * 1000,
    )

    return ranked_chunks
