"""Tests for rag_search tool with mocked services."""

import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Mock heavy third-party libs that are not installed in the test environment
for _mod in ["FlagEmbedding", "pymilvus"]:
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = MagicMock()

from app.services.embedding import EmbeddingOutput  # noqa: E402
from app.services.milvus_client import SearchResult  # noqa: E402
from app.services.reranker import RerankResult  # noqa: E402

# Import the MODULE directly (not the function re-exported via __init__.py)
# so that @patch.object can target module-level service singletons.
import importlib  # noqa: E402
_rag_mod = importlib.import_module("app.agent.tools.rag_search")  # noqa: E402
rag_search = _rag_mod.rag_search
RankedChunk = _rag_mod.RankedChunk


# ── Fixtures ─────────────────────────────────────────────────


def _make_search_results(n: int) -> list[SearchResult]:
    """Create n fake Milvus SearchResult objects."""
    return [
        SearchResult(
            child_id=f"child_{i}",
            parent_id=f"parent_{i}",
            source_doc=f"doc_{i}.pdf",
            heading_path=f"Topic/Sub_{i}",
            chunk_index=i,
            text=f"This is chunk {i} about topic {i}.",
            score=1.0 - i * 0.05,
        )
        for i in range(n)
    ]


def _make_rerank_results(
    passages: list[str], top_n: int
) -> list[RerankResult]:
    """Create fake RerankResult, reversing order to simulate reranker changing ranks."""
    results = [
        RerankResult(text=passages[i], score=0.9 - i * 0.1, original_index=i)
        for i in range(min(top_n, len(passages)))
    ]
    # Reverse to simulate reranker reordering
    results.reverse()
    results.sort(key=lambda r: r.score, reverse=True)
    return results


# ── Tests ────────────────────────────────────────────────────


@patch.object(_rag_mod, "reranker_service")
@patch.object(_rag_mod, "milvus_service")
@patch.object(_rag_mod, "embedding_service")
def test_rag_search_basic(mock_embed, mock_milvus, mock_reranker):
    """Basic flow: encode → hybrid_search → rerank → return RankedChunks."""
    # Mock embedding
    mock_embed.encode_query.return_value = EmbeddingOutput(
        dense=np.random.randn(1024).astype(np.float32),
        sparse=[{1: 0.5, 2: 0.3}],
    )

    # Mock hybrid_search returns 15 candidates (top_k=5, candidate=15)
    candidates = _make_search_results(15)
    mock_milvus.hybrid_search.return_value = candidates

    # Mock reranker
    passages = [c.text for c in candidates]
    mock_reranker.rerank.return_value = _make_rerank_results(passages, top_n=5)

    # Run
    results = rag_search("What is attention?", top_k=5)

    # Assertions
    assert len(results) == 5
    assert all(isinstance(r, RankedChunk) for r in results)

    # Verify rerank scores are descending
    scores = [r.rerank_score for r in results]
    assert scores == sorted(scores, reverse=True)

    # Verify encode_query was called with the query
    mock_embed.encode_query.assert_called_once_with("What is attention?")

    # Verify hybrid_search was called with candidate_k = 5 * 3 = 15
    mock_milvus.hybrid_search.assert_called_once()
    call_kwargs = mock_milvus.hybrid_search.call_args
    assert call_kwargs.kwargs["top_k"] == 15

    # Verify reranker was called with top_n=5
    mock_reranker.rerank.assert_called_once()
    assert mock_reranker.rerank.call_args.kwargs["top_n"] == 5


@patch.object(_rag_mod, "reranker_service")
@patch.object(_rag_mod, "milvus_service")
@patch.object(_rag_mod, "embedding_service")
def test_rag_search_empty_results(mock_embed, mock_milvus, mock_reranker):
    """When Milvus returns no candidates, return empty list without calling reranker."""
    mock_embed.encode_query.return_value = EmbeddingOutput(
        dense=np.random.randn(1024).astype(np.float32),
        sparse=[{1: 0.5}],
    )
    mock_milvus.hybrid_search.return_value = []

    results = rag_search("nonexistent topic", top_k=5)

    assert results == []
    mock_reranker.rerank.assert_not_called()


@patch.object(_rag_mod, "reranker_service")
@patch.object(_rag_mod, "milvus_service")
@patch.object(_rag_mod, "embedding_service")
def test_rag_search_fewer_candidates_than_topk(mock_embed, mock_milvus, mock_reranker):
    """When Milvus returns fewer candidates than top_k, rerank all of them."""
    mock_embed.encode_query.return_value = EmbeddingOutput(
        dense=np.random.randn(1024).astype(np.float32),
        sparse=[{1: 0.5}],
    )

    # Only 3 candidates, but top_k=5
    candidates = _make_search_results(3)
    mock_milvus.hybrid_search.return_value = candidates

    passages = [c.text for c in candidates]
    mock_reranker.rerank.return_value = _make_rerank_results(passages, top_n=5)

    results = rag_search("some query", top_k=5)

    assert len(results) == 3
    assert all(isinstance(r, RankedChunk) for r in results)


@patch.object(_rag_mod, "reranker_service")
@patch.object(_rag_mod, "milvus_service")
@patch.object(_rag_mod, "embedding_service")
def test_rag_search_metadata_preserved(mock_embed, mock_milvus, mock_reranker):
    """Verify that source metadata is correctly mapped from candidates through rerank."""
    mock_embed.encode_query.return_value = EmbeddingOutput(
        dense=np.random.randn(1024).astype(np.float32),
        sparse=[{1: 0.5}],
    )

    candidates = _make_search_results(6)
    mock_milvus.hybrid_search.return_value = candidates

    # Reranker picks candidate at index 3 as top result
    mock_reranker.rerank.return_value = [
        RerankResult(text=candidates[3].text, score=0.95, original_index=3),
        RerankResult(text=candidates[0].text, score=0.80, original_index=0),
    ]

    results = rag_search("test", top_k=2)

    # Top result should have metadata from candidate[3]
    assert results[0].child_id == "child_3"
    assert results[0].parent_id == "parent_3"
    assert results[0].source_doc == "doc_3.pdf"
    assert results[0].heading_path == "Topic/Sub_3"
    assert results[0].rerank_score == 0.95

    # Second result from candidate[0]
    assert results[1].child_id == "child_0"
    assert results[1].rerank_score == 0.80


@patch.object(_rag_mod, "reranker_service")
@patch.object(_rag_mod, "milvus_service")
@patch.object(_rag_mod, "embedding_service")
def test_rag_search_with_filter_expr(mock_embed, mock_milvus, mock_reranker):
    """Verify that expr filter is passed through to hybrid_search."""
    mock_embed.encode_query.return_value = EmbeddingOutput(
        dense=np.random.randn(1024).astype(np.float32),
        sparse=[{1: 0.5}],
    )
    mock_milvus.hybrid_search.return_value = _make_search_results(3)

    passages = [c.text for c in _make_search_results(3)]
    mock_reranker.rerank.return_value = _make_rerank_results(passages, top_n=2)

    rag_search("test", top_k=2, expr='source_doc == "notes.pdf"')

    call_kwargs = mock_milvus.hybrid_search.call_args.kwargs
    assert call_kwargs["expr"] == 'source_doc == "notes.pdf"'
