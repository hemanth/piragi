"""LanceDB vector store implementation."""

import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from ..types import Chunk, Citation


# Common embedding model dimensions
EMBEDDING_DIMENSIONS = {
    "all-mpnet-base-v2": 768,
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "nvidia/llama-embed-nemotron-8b": 4096,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "google/embeddinggemma-2": 768,
    "embeddinggemma-2": 768,
    "google/embeddinggemma-300m": 768,
    "text-embedding-004": 768,
    "google/text-embedding-004": 768,
    "text-multilingual-embedding-002": 768,
    "google/text-multilingual-embedding-002": 768,
}


def get_embedding_dimension(model_name: str) -> int:
    """Get the embedding dimension for a model."""
    if not model_name:
        return 384

    if model_name in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[model_name]

    model_lower = model_name.lower().strip()
    if model_lower in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[model_lower]

    for key, dim in EMBEDDING_DIMENSIONS.items():
        key_lower = key.lower()
        if key_lower in model_lower or model_lower.endswith(key_lower) or key_lower.endswith(model_lower):
            return dim

    # Check for vertex ai publishers/google/models/... resource path pattern
    if "text-embedding-004" in model_lower or "text-multilingual-embedding-002" in model_lower:
        return 768

    try:
        from piragi.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator(model=model_name)
        dim = len(gen.embed_query("dimension probe"))
        EMBEDDING_DIMENSIONS[model_name] = dim  # cache for next time
        logger.info("Auto-detected %d dimensions for model %s", dim, model_name)
        return dim
    except Exception:
        return 384  # Default


class LanceStore:
    """
    LanceDB-based vector store.

    Supports local storage, S3-backed storage, and GCS-backed storage.

    Examples:
        >>> # Local storage
        >>> store = LanceStore(uri=".piragi")
        >>>
        >>> # S3 storage
        >>> store = LanceStore(uri="s3://my-bucket/indices")
        >>>
        >>> # GCS storage
        >>> store = LanceStore(uri="gs://my-bucket/indices")
    """

    def __init__(
        self,
        uri: str = ".piragi",
        embedding_model: str = "all-mpnet-base-v2",
        vector_dimension: Optional[int] = None,
    ) -> None:
        """
        Initialize LanceDB store.

        Args:
            uri: Storage URI (local path, s3://bucket/path, or gs://bucket/path)
            embedding_model: Model name for dimension inference
            vector_dimension: Explicit vector dimension (overrides inference)
        """
        import lancedb

        self.uri = uri
        self.embedding_model = embedding_model

        if vector_dimension is not None:
            self.vector_dimension = vector_dimension
        else:
            self.vector_dimension = get_embedding_dimension(embedding_model)

        # Create local directory if needed (bypass for cloud object storage)
        if not any(uri.lower().startswith(prefix) for prefix in ("s3://", "gs://", "gcs://", "az://", "abfs://")):
            Path(uri).mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(uri)
        self.table_name = "chunks"
        self.table: Optional[Any] = None
        self._chunk_texts: List[str] = []
        self._lock = threading.Lock()

        # Load existing table if present
        if self.table_name in self._table_names():
            self.table = self.db.open_table(self.table_name)
            try:
                results = self.table.to_pandas()
                self._chunk_texts = results["text"].tolist()
            except Exception:
                pass

    def _table_names(self) -> List[str]:
        """Safely list existing table names across LanceDB versions."""
        if hasattr(self.db, "table_names"):
            return list(self.db.table_names())
        if hasattr(self.db, "list_tables"):
            res = self.db.list_tables()
            if hasattr(res, "tables"):
                return list(res.tables)
            return list(res)
        return []

    def create_table(self, data: Any, mode: str = "overwrite") -> Any:
        """Create or overwrite the table with synchronization."""
        with self._lock:
            self.table = self.db.create_table(self.table_name, data=data, mode=mode)
            return self.table

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks with embeddings to the store."""
        if not chunks:
            return

        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError("All chunks must have embeddings")

        data = [
            {
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
                "vector": chunk.embedding,
            }
            for chunk in chunks
        ]

        with self._lock:
            self._chunk_texts.extend([chunk.text for chunk in chunks])

            if self.table is None:
                if self.table_name in self._table_names():
                    self.table = self.db.open_table(self.table_name)
                    self.table.add(data)
                else:
                    self.table = self.db.create_table(self.table_name, data=data, mode="overwrite")
            else:
                self.table.add(data)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_chunk_length: int = 0,
    ) -> List[Citation]:
        """Search for similar chunks."""
        if self.table is None or top_k <= 0:
            return []

        search_limit = top_k * 3
        query = self.table.search(query_embedding).limit(search_limit)

        if filters:
            filter_conditions = []
            for key, value in filters.items():
                safe_key = str(key).replace("'", "''")
                safe_value = str(value).replace("'", "''")
                filter_conditions.append(f"metadata['{safe_key}'] = '{safe_value}'")
            if filter_conditions:
                query = query.where(" AND ".join(filter_conditions))

        results = query.to_list()

        citations = []
        for result in results:
            chunk_text = result["text"]
            if len(chunk_text) < min_chunk_length:
                continue

            citations.append(
                Citation(
                    source=result["source"],
                    chunk=chunk_text,
                    score=max(0.0, min(1.0, 1.0 - result["_distance"])),
                    metadata=result["metadata"],
                )
            )

            if len(citations) >= top_k:
                break

        return citations

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source."""
        with self._lock:
            if self.table is None:
                return 0

            count_before = self.table.count_rows()
            safe_source = source.replace("'", "''")
            self.table.delete(f"source = '{safe_source}'")
            count_after = self.table.count_rows()

            return count_before - count_after

    def count(self) -> int:
        """Return the number of chunks in the store."""
        if self.table is None:
            return 0
        return self.table.count_rows()

    def clear(self) -> None:
        """Clear all data from the store."""
        with self._lock:
            if self.table_name in self._table_names():
                self.db.drop_table(self.table_name)
            self.table = None
            self._chunk_texts = []

    def get_all_chunk_texts(self) -> List[str]:
        """Get all chunk texts for hybrid search."""
        return self._chunk_texts

    def get_all_citations(self, filters: Optional[Dict[str, Any]] = None) -> List[Citation]:
        """Current corpus with source/metadata for independent local BM25 retrieval.

        Read the live table so reopen, deletion and refresh do not leave a stale
        keyword index. Equality filters match the public metadata-filter interface.
        """
        if self.table is None:
            return []
        rows = self.table.to_arrow().select(["text", "source", "metadata"]).to_pylist()
        return [
            Citation(
                source=row["source"],
                chunk=row["text"],
                score=0,
                metadata=row["metadata"] or {},
            )
            for row in rows
            if row.get("text")
            and all((row["metadata"] or {}).get(k) == v for k, v in (filters or {}).items())
        ]
