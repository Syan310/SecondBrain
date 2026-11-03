from __future__ import annotations

"""Reranker service using BAAI/bge-reranker-v2-m3 (cross-encoder)."""

import logging
import threading
from dataclasses import dataclass

from FlagEmbedding import FlagReranker

from app.config import settings

logger = logging.getLogger(__name__)

# Max tokens per [query, passage] pair. FlagReranker defaults to 512 and silently
# truncates longer pairs; the model supports up to 8192.
RERANKER_MAX_LENGTH = 1024  # §4.3: qa_l0_v1 has pairs up to 555 tokens (18/1800 over 512)


@dataclass
class RerankResult:
    """A single reranked passage with its score."""

    text: str
    score: float
    original_index: int


class RerankerService:
    """Singleton wrapper around bge-reranker-v2-m3 cross-encoder."""

    def __init__(self, max_length: int = RERANKER_MAX_LENGTH) -> None:
        self._model: FlagReranker | None = None
        self._lock = threading.Lock()
        self.max_length = max_length

    def _load_model(self) -> FlagReranker:
        if self._model is None:
            with self._lock:
                if self._model is None:  # double-check after acquiring lock
                    logger.info("Loading reranker model from %s ...", settings.reranker_model_path)
                    self._model = FlagReranker(
                        settings.reranker_model_path, use_fp16=True, devices="cuda:0",
                    )
                    logger.info("Reranker model loaded.")
        return self._model

    def warmup(self) -> None:
        """Pre-load model single-threaded before concurrent access."""
        self._load_model()

    def rerank(
        self,
        query: str,
        passages: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Rerank passages by relevance to query.

        Args:
            query: The user query.
            passages: List of candidate passages to rerank.
            top_n: Return top N results. None means return all, sorted by score desc.

        Returns:
            List of RerankResult sorted by score descending.
        """
        if not passages:
            return []

        model = self._load_model()

        pairs = [[query, p] for p in passages]
        with self._lock:
            scores = model.compute_score(pairs, normalize=True, max_length=self.max_length)

        # compute_score returns a single float when there's only one pair
        if isinstance(scores, (float, int)):
            scores = [scores]

        results = [
            RerankResult(text=passages[i], score=float(scores[i]), original_index=i)
            for i in range(len(passages))
        ]
        results.sort(key=lambda r: r.score, reverse=True)

        if top_n is not None:
            results = results[:top_n]

        return results


# Module-level singleton
reranker_service = RerankerService()
