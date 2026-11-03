"""Tests for MilvusService — requires a running Milvus instance.

Run: pytest tests/test_milvus_client.py -v
"""

import uuid

import pytest

pymilvus = pytest.importorskip("pymilvus", reason="pymilvus not installed")

from app.config import settings  # noqa: E402
from app.services.milvus_client import MilvusService, _DIM, build_bm25_text  # noqa: E402


@pytest.fixture(scope="module")
def svc():
    """Create a MilvusService connected to a test collection."""
    import app.services.milvus_client as mc

    # Use a throwaway collection name to avoid polluting real data
    original_name = settings.milvus_collection_name
    original_const = mc._COLLECTION
    test_collection = f"test_{uuid.uuid4().hex[:8]}"
    settings.milvus_collection_name = test_collection
    mc._COLLECTION = test_collection

    service = MilvusService()
    service.connect()
    service.init_collection()
    yield service

    # Cleanup: drop test collection
    from pymilvus import utility
    utility.drop_collection(test_collection)
    settings.milvus_collection_name = original_name
    mc._COLLECTION = original_const


def _fake_embedding(seed: float = 0.0) -> list[float]:
    """Generate a deterministic normalized 1024-dim vector."""
    import math
    vec = [math.sin(seed + i) for i in range(_DIM)]
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


def test_build_bm25_text_prepends_heading():
    """BM25 input uses the same heading augmentation as the dense input (no server needed)."""
    assert build_bm25_text("Transformer/FFN", "两层全连接") == "Transformer/FFN\n两层全连接"
    assert build_bm25_text("", "两层全连接") == "两层全连接"


class TestInsertAndSearch:
    def test_insert(self, svc: MilvusService):
        count = svc.insert(
            parent_ids=["p1", "p1", "p2"],
            source_docs=["test.md", "test.md", "test.md"],
            heading_paths=["A/B", "A/B", "A/C"],
            chunk_indices=[0, 1, 0],
            texts=["hello world", "foo bar baz", "another chunk"],
            embeddings=[_fake_embedding(0), _fake_embedding(1), _fake_embedding(2)],
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
        """Hybrid search (dense + BM25) with RRF fusion."""
        results = svc.hybrid_search(
            query_vector=_fake_embedding(0),
            query_text="hello",
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
            query_text="hello",
            top_k=10,
            expr='heading_path == "A/C"',
        )
        assert all(r.heading_path == "A/C" for r in results)


class TestBM25:
    """BM25 path with the jieba analyzer on mixed Chinese/English notes."""

    FFN, POS, RRF = "Transformer/6. Feed Forward Network（FFN）", "Transformer/位置编码", "RAG/融合"

    def test_insert_notes(self, svc: MilvusService):
        count = svc.insert(
            parent_ids=["p3", "p4", "p5"],
            source_docs=["test.md"] * 3,
            heading_paths=[self.FFN, self.POS, self.RRF],
            chunk_indices=[0, 0, 0],
            texts=[
                "两层全连接，中间用激活函数，逐位置独立计算。",
                "原始 Transformer 使用正弦和余弦函数生成位置编码，不参与梯度更新。",
                "RRF 只使用名次，不需要校准两路分数。",
            ],
            embeddings=[_fake_embedding(3), _fake_embedding(4), _fake_embedding(5)],
        )
        assert count == 3

    def test_heading_only_term_matches(self, svc: MilvusService):
        """"FFN" is only in the heading, not the body: heading augmentation must find it."""
        results = svc.bm25_search("FFN", top_k=3)
        assert results[0].heading_path == self.FFN
        assert "FFN" not in results[0].text  # text field keeps the original body

    def test_lowercase_query_matches(self, svc: MilvusService):
        results = svc.bm25_search("ffn", top_k=3)
        assert results[0].heading_path == self.FFN

    def test_chinese_term_matches(self, svc: MilvusService):
        results = svc.bm25_search("位置编码", top_k=3)
        assert results[0].heading_path == self.POS

    def test_mixed_query_matches(self, svc: MilvusService):
        results = svc.bm25_search("RRF 名次", top_k=3)
        assert results[0].heading_path == self.RRF

    def test_bm25_with_filter(self, svc: MilvusService):
        results = svc.bm25_search("位置编码", top_k=10, expr='heading_path == "RAG/融合"')
        assert all(r.heading_path == self.RRF for r in results)

    def test_hybrid_uses_bm25_path(self, svc: MilvusService):
        """Dense vector points elsewhere (seed 0); BM25 still brings the FFN chunk in."""
        results = svc.hybrid_search(query_vector=_fake_embedding(0), query_text="FFN", top_k=3)
        assert self.FFN in [r.heading_path for r in results]


class TestDelete:
    def test_delete_by_source(self, svc: MilvusService):
        svc.delete_by_source("test.md")
        assert svc.count() == 0
