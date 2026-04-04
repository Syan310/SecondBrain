"""Embedding service using BAAI/bge-m3 (768-dim, multilingual Chinese/English)."""

import logging
from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Singleton wrapper around bge-m3 for text embedding."""

    DIMENSION = 1024

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model from %s ...", settings.embedding_model_path)
            self._model = SentenceTransformer(settings.embedding_model_path)
            logger.info("Embedding model loaded. dim=%d", self.DIMENSION)
        return self._model

    def encode(
        self,
        texts: Union[str, list[str]],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts into dense vectors.

        Args:
            texts: Single string or list of strings.
            batch_size: Batch size for encoding.
            normalize: Whether to L2-normalize embeddings (required for cosine similarity).

        Returns:
            np.ndarray of shape (n_texts, DIMENSION).
        """
        model = self._load_model()
        if isinstance(texts, str):
            texts = [texts]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query. Returns shape (DIMENSION,)."""
        return self.encode(query)[0]

    def encode_documents(
        self,
        documents: list[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Encode a batch of documents. Returns shape (n_docs, DIMENSION)."""
        return self.encode(documents, batch_size=batch_size)


# Module-level singleton
embedding_service = EmbeddingService()
