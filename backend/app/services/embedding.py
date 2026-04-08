from __future__ import annotations

"""Embedding service using BAAI/bge-m3 via FlagEmbedding.

Supports both dense (1024-dim) and sparse (learned lexical weights) outputs.
Uses BGEM3FlagModel instead of SentenceTransformer to unlock sparse retrieval.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Union

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingOutput:
    """Combined dense + sparse embedding output."""

    dense: np.ndarray  # shape (n_texts, 1024)
    sparse: list[dict[int, float]]  # per-text {token_id: weight}


class EmbeddingService:
    """Singleton wrapper around bge-m3 for text embedding (dense + sparse)."""

    DIMENSION = 1024

    def __init__(self) -> None:
        self._model: BGEM3FlagModel | None = None
        self._lock = threading.Lock()

    def _load_model(self) -> BGEM3FlagModel:
        if self._model is None:
            with self._lock:
                if self._model is None:  # double-check after acquiring lock
                    logger.info("Loading BGE-m3 model from %s ...", settings.embedding_model_path)
                    self._model = BGEM3FlagModel(
                        settings.embedding_model_path, use_fp16=True,
                    )
                    logger.info("BGE-m3 model loaded. dense_dim=%d, sparse=enabled", self.DIMENSION)
        return self._model

    def warmup(self) -> None:
        """Pre-load model single-threaded before concurrent access."""
        self._load_model()

    # ── Full output (dense + sparse) ────────────────────────────

    def encode(
        self,
        texts: Union[str, list[str]],
        batch_size: int = 64,
        return_sparse: bool = True,
    ) -> EmbeddingOutput:
        """Encode texts into dense vectors + optional sparse vectors.

        Args:
            texts: Single string or list of strings.
            batch_size: Batch size for encoding.
            return_sparse: Whether to compute sparse (lexical) vectors.

        Returns:
            EmbeddingOutput with dense ndarray and sparse dicts.
        """
        model = self._load_model()
        if isinstance(texts, str):
            texts = [texts]

        output = model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )

        dense = output["dense_vecs"]  # np.ndarray (n, 1024)

        sparse: list[dict[int, float]] = []
        if return_sparse:
            # output["lexical_weights"] is list[dict[str, float]]
            # Milvus SPARSE_FLOAT_VECTOR needs {int: float}
            for token_weights in output["lexical_weights"]:
                sparse.append({int(k): float(v) for k, v in token_weights.items()})
        else:
            sparse = [{} for _ in texts]

        return EmbeddingOutput(dense=dense, sparse=sparse)

    # ── Convenience: dense-only ─────────────────────────────────

    def encode_dense(
        self,
        texts: Union[str, list[str]],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Encode texts into dense vectors only. Returns shape (n_texts, 1024)."""
        return self.encode(texts, batch_size=batch_size, return_sparse=False).dense

    # ── Convenience: single query ───────────────────────────────

    def encode_query(self, query: str) -> EmbeddingOutput:
        """Encode a single query. Dense shape (1024,), sparse is list with one dict."""
        out = self.encode(query, return_sparse=True)
        return EmbeddingOutput(dense=out.dense[0], sparse=out.sparse)

    # ── Convenience: batch documents ────────────────────────────

    def encode_documents(
        self,
        documents: list[str],
        batch_size: int = 64,
    ) -> EmbeddingOutput:
        """Encode a batch of documents. Dense shape (n_docs, 1024)."""
        return self.encode(documents, batch_size=batch_size, return_sparse=True)


# Module-level singleton
embedding_service = EmbeddingService()
