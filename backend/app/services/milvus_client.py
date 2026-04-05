from __future__ import annotations

"""Milvus client: child_chunks collection CRUD operations.

Collection schema:
  - child_id        (VARCHAR, PK)           — UUID
  - parent_id       (VARCHAR)               — group identifier for sibling chunks
  - source_doc      (VARCHAR)               — origin filename, for filtering & re-ingestion
  - heading_path    (VARCHAR)               — e.g. "NLP/Transformer/Attention"
  - chunk_index     (INT16)                 — order within parent group
  - text            (VARCHAR)               — raw text for reranker scoring
  - embedding       (FLOAT_VECTOR, 1024)    — BGE-m3 dense vector
  - sparse_embedding (SPARSE_FLOAT_VECTOR)  — BGE-m3 learned sparse vector

Indexes:
  - embedding:        HNSW + IP
  - sparse_embedding: SPARSE_INVERTED_INDEX + IP
  - source_doc:       INVERTED (scalar)
"""

import logging
import uuid
from dataclasses import dataclass

from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    RRFRanker,
    connections,
    utility,
)

from app.config import settings
from app.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)


def _escape_milvus_str(value: str) -> str:
    """Escape a string value for safe use in Milvus filter expressions."""
    return value.replace("\\", "\\\\").replace('"', '\\"')

_COLLECTION = settings.milvus_collection_name
_DIM = EmbeddingService.DIMENSION
_MAX_TEXT_LEN = 8192  # chars, enough for child chunks (≤750 chars typical)
_OUTPUT_FIELDS = ["parent_id", "source_doc", "heading_path", "chunk_index", "text"]


# ── Data structures ──────────────────────────────────────────


@dataclass
class SearchResult:
    """A single search hit returned by MilvusService."""

    child_id: str
    parent_id: str
    source_doc: str
    heading_path: str
    chunk_index: int
    text: str
    score: float


# ── Service ──────────────────────────────────────────────────


class MilvusService:
    """Encapsulates Milvus connection, collection init, insert, search, delete."""

    def __init__(self) -> None:
        self._collection: Collection | None = None

    # ── Connection & collection ──────────────────────────────

    def connect(self) -> None:
        """Establish connection to Milvus server."""
        connections.connect(
            alias="default",
            host=settings.milvus_host,
            port=settings.milvus_port,
        )
        logger.info(
            "Connected to Milvus at %s:%s", settings.milvus_host, settings.milvus_port
        )

    def init_collection(self) -> Collection:
        """Create collection if not exists, create indexes, and load into memory."""
        if self._collection is not None:
            return self._collection

        if utility.has_collection(_COLLECTION):
            existing = Collection(_COLLECTION)
            field_names = {f.name for f in existing.schema.fields}
            if "sparse_embedding" not in field_names:
                logger.warning(
                    "Collection '%s' missing sparse_embedding field, "
                    "dropping and recreating with new schema.",
                    _COLLECTION,
                )
                utility.drop_collection(_COLLECTION)
            else:
                logger.info("Collection '%s' already exists, loading.", _COLLECTION)
                self._collection = existing
                self._collection.load()
                return self._collection

        # Define schema
        fields = [
            FieldSchema(name="child_id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=36),
            FieldSchema(name="source_doc", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="heading_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_index", dtype=DataType.INT16),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=_MAX_TEXT_LEN),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=_DIM),
            FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
        schema = CollectionSchema(fields=fields, description="SecondBrain child chunks")

        self._collection = Collection(name=_COLLECTION, schema=schema)
        logger.info("Created collection '%s'.", _COLLECTION)

        # Dense vector index: HNSW + IP
        self._collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "HNSW",
                "metric_type": "IP",
                "params": {"M": 16, "efConstruction": 256},
            },
        )

        # Sparse vector index: SPARSE_INVERTED_INDEX + IP
        self._collection.create_index(
            field_name="sparse_embedding",
            index_params={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",
            },
        )

        # Scalar index on source_doc for filtered search & delete
        self._collection.create_index(
            field_name="source_doc",
            index_name="idx_source_doc",
            index_params={"index_type": "INVERTED"},
        )

        self._collection.load()
        logger.info("Collection '%s' indexed and loaded.", _COLLECTION)
        return self._collection

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            raise RuntimeError("Call connect() and init_collection() first.")
        return self._collection

    # ── Insert ───────────────────────────────────────────────

    def insert(
        self,
        parent_ids: list[str],
        source_docs: list[str],
        heading_paths: list[str],
        chunk_indices: list[int],
        texts: list[str],
        embeddings: list[list[float]],
        sparse_embeddings: list[dict[int, float]],
    ) -> int:
        """Insert child chunks into Milvus.

        All lists must have the same length. Returns the number of inserted rows.
        """
        n = len(texts)
        child_ids = [str(uuid.uuid4()) for _ in range(n)]

        data = [
            child_ids,
            parent_ids,
            source_docs,
            heading_paths,
            chunk_indices,
            texts,
            embeddings,
            sparse_embeddings,
        ]

        result = self.collection.insert(data)
        self.collection.flush()
        logger.info("Inserted %d chunks into '%s'.", result.insert_count, _COLLECTION)
        return result.insert_count

    # ── Dense-only search ────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        expr: str | None = None,
        ef: int = 128,
    ) -> list[SearchResult]:
        """Dense vector search with optional scalar filter.

        Args:
            query_vector: 1024-dim query embedding.
            top_k: Number of results to return.
            expr: Optional Milvus boolean expression, e.g. 'source_doc == "xxx.pdf"'.
            ef: HNSW search ef parameter (higher = more accurate, slower).

        Returns:
            List of SearchResult ordered by descending score.
        """
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": ef}},
            limit=top_k,
            expr=expr,
            output_fields=_OUTPUT_FIELDS,
        )
        return self._parse_hits(results[0])

    # ── Hybrid search (dense + sparse, RRF fusion) ──────────

    def hybrid_search(
        self,
        query_vector: list[float],
        sparse_vector: dict[int, float],
        top_k: int = 20,
        expr: str | None = None,
        ef: int = 128,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """Dense + sparse hybrid search with RRF fusion.

        Args:
            query_vector: 1024-dim dense query embedding.
            sparse_vector: Sparse query vector {token_id: weight}.
            top_k: Number of final fused results to return.
            expr: Optional Milvus boolean expression filter.
            ef: HNSW search ef parameter.
            rrf_k: RRF smoothing parameter (default 60).

        Returns:
            List of SearchResult ordered by fused RRF score descending.
        """
        # Dense search request
        dense_req = AnnSearchRequest(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": ef}},
            limit=top_k,
            expr=expr,
        )

        # Sparse search request
        sparse_req = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_embedding",
            param={"metric_type": "IP"},
            limit=top_k,
            expr=expr,
        )

        # RRF fusion (pymilvus 2.4+: positional arg `rerank`, not `ranker`)
        results = self.collection.hybrid_search(
            [dense_req, sparse_req],
            RRFRanker(k=rrf_k),
            top_k,
            output_fields=_OUTPUT_FIELDS,
        )
        return self._parse_hits(results[0])

    # ── Delete ───────────────────────────────────────────────

    def delete_by_source(self, source_doc: str) -> None:
        """Delete all chunks from a specific source document (for re-ingestion)."""
        expr = f'source_doc == "{_escape_milvus_str(source_doc)}"'
        self.collection.delete(expr=expr)
        self.collection.flush()
        logger.info("Deleted chunks where %s.", expr)

    # ── Stats ────────────────────────────────────────────────

    def count(self) -> int:
        """Return total number of entities in collection."""
        result = self.collection.query(expr="child_id != ''", output_fields=["count(*)"])
        return result[0]["count(*)"]

    # ── Internal ─────────────────────────────────────────────

    @staticmethod
    def _parse_hits(hits) -> list[SearchResult]:
        """Convert Milvus hit objects to SearchResult list."""
        return [
            SearchResult(
                child_id=hit.id,
                parent_id=hit.entity.get("parent_id"),
                source_doc=hit.entity.get("source_doc"),
                heading_path=hit.entity.get("heading_path"),
                chunk_index=hit.entity.get("chunk_index"),
                text=hit.entity.get("text"),
                score=hit.score,
            )
            for hit in hits
        ]


# ── Memory Collection Service ───────────────────────────────


_MEMORY_COLLECTION = settings.milvus_memory_collection
_MEMORY_MAX_TEXT_LEN = 4096
_MEMORY_OUTPUT_FIELDS = ["user_id", "summary", "topics", "session_id", "created_at"]


@dataclass
class MemorySearchResult:
    """A single long-term memory hit."""

    memory_id: str
    user_id: str
    summary: str
    topics: str          # JSON string of topic list
    session_id: str
    created_at: float    # unix timestamp
    score: float


class MemoryMilvusService:
    """Milvus operations for conversation_memory collection."""

    def __init__(self) -> None:
        self._collection: Collection | None = None

    def init_collection(self) -> Collection:
        """Create conversation_memory collection if not exists."""
        if self._collection is not None:
            return self._collection

        if utility.has_collection(_MEMORY_COLLECTION):
            existing = Collection(_MEMORY_COLLECTION)
            logger.info("Memory collection '%s' already exists, loading.", _MEMORY_COLLECTION)
            self._collection = existing
            self._collection.load()
            return self._collection

        fields = [
            FieldSchema(name="memory_id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=_MEMORY_MAX_TEXT_LEN),
            FieldSchema(name="topics", dtype=DataType.VARCHAR, max_length=1024),     # JSON array string
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="created_at", dtype=DataType.DOUBLE),                   # unix timestamp
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=_DIM),
            FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
        schema = CollectionSchema(fields=fields, description="Long-term conversation memory")

        self._collection = Collection(name=_MEMORY_COLLECTION, schema=schema)
        logger.info("Created memory collection '%s'.", _MEMORY_COLLECTION)

        # Dense index
        self._collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "HNSW",
                "metric_type": "IP",
                "params": {"M": 16, "efConstruction": 256},
            },
        )

        # Sparse index
        self._collection.create_index(
            field_name="sparse_embedding",
            index_params={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",
            },
        )

        # Scalar index on user_id for filtered search
        self._collection.create_index(
            field_name="user_id",
            index_name="idx_memory_user_id",
            index_params={"index_type": "INVERTED"},
        )

        self._collection.load()
        logger.info("Memory collection '%s' indexed and loaded.", _MEMORY_COLLECTION)
        return self._collection

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            raise RuntimeError("Call init_collection() first.")
        return self._collection

    def insert(
        self,
        user_id: str,
        summary: str,
        topics: str,
        session_id: str,
        created_at: float,
        embedding: list[float],
        sparse_embedding: dict[int, float],
    ) -> str:
        """Insert one long-term memory entry. Returns the memory_id."""
        memory_id = str(uuid.uuid4())
        data = [
            [memory_id],
            [user_id],
            [summary],
            [topics],
            [session_id],
            [created_at],
            [embedding],
            [sparse_embedding],
        ]
        self.collection.insert(data)
        self.collection.flush()
        logger.info("Inserted memory '%s' for user '%s'.", memory_id, user_id)
        return memory_id

    def hybrid_search(
        self,
        query_vector: list[float],
        sparse_vector: dict[int, float],
        user_id: str,
        top_k: int = 3,
        rrf_k: int = 60,
    ) -> list[MemorySearchResult]:
        """Search long-term memory for a specific user."""
        expr = f'user_id == "{_escape_milvus_str(user_id)}"'

        dense_req = AnnSearchRequest(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k,
            expr=expr,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_embedding",
            param={"metric_type": "IP"},
            limit=top_k,
            expr=expr,
        )

        results = self.collection.hybrid_search(
            [dense_req, sparse_req],
            RRFRanker(k=rrf_k),
            top_k,
            output_fields=_MEMORY_OUTPUT_FIELDS,
        )
        return [
            MemorySearchResult(
                memory_id=hit.id,
                user_id=hit.entity.get("user_id"),
                summary=hit.entity.get("summary"),
                topics=hit.entity.get("topics"),
                session_id=hit.entity.get("session_id"),
                created_at=hit.entity.get("created_at"),
                score=hit.score,
            )
            for hit in results[0]
        ]


# Module-level singletons
milvus_service = MilvusService()
memory_milvus_service = MemoryMilvusService()
