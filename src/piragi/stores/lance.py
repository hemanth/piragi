"""LanceDB vector store implementation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

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
}


def get_embedding_dimension(model_name: str) -> int:
    """Get the embedding dimension for a model."""
    if model_name in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[model_name]

    for key, dim in EMBEDDING_DIMENSIONS.items():
        if key in model_name or model_name.endswith(key):
            return dim

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

    Supports local storage and S3-backed storage.

    Examples:
        >>> # Local storage
        >>> store = LanceStore(uri=".piragi")
        >>>
        >>> # S3 storage
        >>> store = LanceStore(uri="s3://my-bucket/indices")
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
            uri: Storage URI (local path or s3://bucket/path)
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

        # Create local directory if needed
        if not uri.startswith("s3://"):
            Path(uri).mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(uri)
        self.table_name = "chunks"
        self.table: Optional[Any] = None
        self._chunk_texts: List[str] = []

        # In-memory numpy fast-path (flat L2 scan).
        # Raw matrix + squared norms + parallel row metadata; skips the LanceDB
        # round-trip for small corpora. Lazily built, invalidated on write.
        self._vec_matrix: Optional[np.ndarray] = None
        self._rows: Optional[List[Dict[str, Any]]] = None
        self._sqnorms: Optional[np.ndarray] = None
        self._max_inmem = 50000

        # Load existing table if present
        if self.table_name in self.db.table_names():
            self.table = self.db.open_table(self.table_name)
            try:
                results = self.table.to_pandas()
                self._chunk_texts = results["text"].tolist()
            except Exception:
                pass

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

        self._chunk_texts.extend([chunk.text for chunk in chunks])

        # Invalidate in-memory matrix; rebuilt lazily on next search.
        self._vec_matrix = None
        self._rows = None
        self._sqnorms = None

        if self.table is None:
            self.table = self.db.create_table(self.table_name, data=data, mode="overwrite")
        else:
            self.table.add(data)

    def _ensure_inmem(self) -> bool:
        """Build the in-memory matrix if the corpus is small enough.

        Returns True if the fast-path is available, False to fall back to LanceDB.
        """
        if self._vec_matrix is not None:
            return True
        if self.table is None:
            return False
        try:
            n = self.table.count_rows()
        except Exception:
            n = self.count()
        if n == 0 or n > self._max_inmem:
            return False

        try:
            df = self.table.to_pandas()
        except Exception:
            return False

        vecs = np.asarray(df["vector"].tolist(), dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] == 0:
            return False

        # Keep raw vectors and precompute squared norms once, so query-time L2
        # ranking (matching LanceDB's default metric) is a single matmul.
        self._vec_matrix = vecs
        self._sqnorms = np.einsum("ij,ij->i", vecs, vecs)

        self._rows = [
            {
                "text": row["text"],
                "source": row["source"],
                "metadata": row["metadata"],
            }
            for _, row in df.iterrows()
        ]
        return True

    def _search_inmem(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
        min_chunk_length: int,
    ) -> List[Citation]:
        matrix = self._vec_matrix
        rows = self._rows
        assert matrix is not None and rows is not None

        # Pre-filter candidate rows before scoring.
        if filters:
            cand = np.fromiter(
                (
                    i
                    for i, r in enumerate(rows)
                    if all(
                        (r["metadata"] or {}).get(k) == v for k, v in filters.items()
                    )
                ),
                dtype=np.int64,
            )
            if cand.size == 0:
                return []
            sub = matrix[cand]
        else:
            cand = None
            sub = matrix

        q = np.asarray(query_embedding, dtype=np.float32)
        sqnorms = self._sqnorms if cand is None else self._sqnorms[cand]

        # Squared L2 distance = ||m||^2 - 2 m·q + ||q||^2. The ||q||^2 term is
        # constant across rows, so ranking uses ||m||^2 - 2 m·q. One matmul.
        dist2 = sqnorms - 2.0 * (sub @ q)

        # Smallest distance first; over-fetch to survive the length filter.
        want = min(top_k * 3, dist2.shape[0])
        if dist2.shape[0] <= want:
            order = np.argsort(dist2)
        else:
            part = np.argpartition(dist2, want - 1)[:want]
            order = part[np.argsort(dist2[part])]

        qq = float(q @ q)
        citations: List[Citation] = []
        for local_idx in order:
            global_idx = int(cand[local_idx]) if cand is not None else int(local_idx)
            r = rows[global_idx]
            if len(r["text"]) < min_chunk_length:
                continue
            distance = float(dist2[local_idx]) + qq
            citations.append(
                Citation(
                    source=r["source"],
                    chunk=r["text"],
                    score=float(max(0.0, min(1.0, 1.0 - distance))),
                    metadata=r["metadata"],
                )
            )
            if len(citations) >= top_k:
                break
        return citations

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_chunk_length: int = 100,
    ) -> List[Citation]:
        """Search for similar chunks."""
        if self.table is None:
            return []

        if self._ensure_inmem():
            return self._search_inmem(
                query_embedding, top_k, filters, min_chunk_length
            )

        search_limit = top_k * 3
        query = self.table.search(query_embedding).limit(search_limit)

        if filters:
            filter_conditions = []
            for key, value in filters.items():
                filter_conditions.append(f"metadata['{key}'] = '{value}'")
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
        if self.table is None:
            return 0

        count_before = self.table.count_rows()
        self.table.delete(f"source = '{source}'")
        count_after = self.table.count_rows()

        return count_before - count_after

    def count(self) -> int:
        """Return the number of chunks in the store."""
        if self.table is None:
            return 0
        return self.table.count_rows()

    def clear(self) -> None:
        """Clear all data from the store."""
        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)
            self.table = None
        self._chunk_texts = []

    def get_all_chunk_texts(self) -> List[str]:
        """Get all chunk texts for hybrid search."""
        return self._chunk_texts

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all chunks as records (text/source/metadata) for hybrid indexing."""
        if self.table is None:
            return []
        try:
            rows = self.table.to_list()
        except Exception:
            return [{"text": t, "source": None, "metadata": {}} for t in self._chunk_texts]
        return [
            {"text": r["text"], "source": r.get("source"), "metadata": r.get("metadata") or {}}
            for r in rows
        ]
