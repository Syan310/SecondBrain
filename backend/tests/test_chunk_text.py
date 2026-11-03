from types import SimpleNamespace

from app.api.routes.ingest import _build_embed_texts
from app.services.chunk_text import reranker_passages, with_heading


def test_with_heading_prepends_heading_path():
    assert with_heading("RAG > 检索", "正文") == "RAG > 检索\n正文"


def test_with_heading_returns_text_when_no_heading():
    assert with_heading("", "正文") == "正文"


def test_embed_texts_and_reranker_passages_use_same_format():
    chunks = [
        SimpleNamespace(heading_path="A > B", text="t1"),
        SimpleNamespace(heading_path="", text="t2"),
    ]

    assert _build_embed_texts(chunks) == reranker_passages(chunks, use_heading=True)
    assert reranker_passages(chunks, use_heading=False) == ["t1", "t2"]
