"""LanceDB vector store implementation."""

from pathlib import Path
from typing import Any, Dict, List, Optional

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

    return 768  # Default


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

        if self.table is None:
            self.table = self.db.create_table(self.table_name, data=data, mode="overwrite")
        else:
            self.table.add(data)

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
