"""Tests for MilvusService — requires a running Milvus instance.

Run: pytest tests/test_milvus_client.py -v
"""

import uuid

import pytest

from app.config import settings
from app.services.milvus_client import MilvusService, _DIM


@pytest.fixture(scope="module")
def svc():
    """Create a MilvusService connected to a test collection."""
    # Use a throwaway collection name to avoid polluting real data
    original_name = settings.milvus_collection_name
    test_collection = f"test_{uuid.uuid4().hex[:8]}"
    settings.milvus_collection_name = test_collection

    service = MilvusService()
    service.connect()
    service.init_collection()
    yield service

    # Cleanup: drop test collection
    from pymilvus import utility
    utility.drop_collection(test_collection)
    settings.milvus_collection_name = original_name


def _fake_embedding(seed: float = 0.0) -> list[float]:
    """Generate a deterministic normalized 1024-dim vector."""
    import math
    vec = [math.sin(seed + i) for i in range(_DIM)]
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


def _fake_sparse(seed: int = 0) -> dict[int, float]:
    """Generate a fake sparse vector with a few non-zero entries."""
    return {seed * 10 + i: 0.5 + i * 0.1 for i in range(5)}


class TestInsertAndSearch:
    def test_insert(self, svc: MilvusService):
        count = svc.insert(
            parent_ids=["p1", "p1", "p2"],
            source_docs=["test.md", "test.md", "test.md"],
            heading_paths=["A/B", "A/B", "A/C"],
            chunk_indices=[0, 1, 0],
            texts=["hello world", "foo bar baz", "another chunk"],
            embeddings=[_fake_embedding(0), _fake_embedding(1), _fake_embedding(2)],
            sparse_embeddings=[_fake_sparse(0), _fake_sparse(1), _fake_sparse(2)],
        )
        assert count == 3

    def test_count(self, svc: MilvusService):
        assert svc.count() == 3

    def test_search_returns_results(self, svc: MilvusService):
        results = svc.search(query_vector=_fake_embedding(0), top_k=2)
        assert len(results) <= 2
        assert results[0].text == "hello world"
        assert results[0].score > 0

    def test_search_with_filter(self, svc: MilvusService):
        results = svc.search(
            query_vector=_fake_embedding(0),
            top_k=10,
            expr='heading_path == "A/C"',
        )
        assert all(r.heading_path == "A/C" for r in results)

    def test_search_result_fields(self, svc: MilvusService):
        results = svc.search(query_vector=_fake_embedding(0), top_k=1)
        r = results[0]
        assert r.child_id  # non-empty UUID string
        assert r.parent_id in ("p1", "p2")
        assert r.source_doc == "test.md"
        assert isinstance(r.chunk_index, int)

    def test_hybrid_search(self, svc: MilvusService):
        """Hybrid search (dense + sparse) with RRF fusion."""
        results = svc.hybrid_search(
            query_vector=_fake_embedding(0),
            sparse_vector=_fake_sparse(0),
            top_k=2,
        )
        assert len(results) <= 2
        assert len(results) > 0
        # Results should have all metadata fields
        r = results[0]
        assert r.child_id
        assert r.text
        assert r.score > 0

    def test_hybrid_search_with_filter(self, svc: MilvusService):
        results = svc.hybrid_search(
            query_vector=_fake_embedding(0),
            sparse_vector=_fake_sparse(0),
            top_k=10,
            expr='heading_path == "A/C"',
        )
        assert all(r.heading_path == "A/C" for r in results)


class TestDelete:
    def test_delete_by_source(self, svc: MilvusService):
        svc.delete_by_source("test.md")
        assert svc.count() == 0
