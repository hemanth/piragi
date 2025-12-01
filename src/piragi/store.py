"""Vector store using LanceDB."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import lancedb
import pyarrow as pa

from .types import Chunk, Citation


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
    """
    Get the embedding dimension for a model.

    Args:
        model_name: Name of the embedding model

    Returns:
        Embedding dimension
    """
    # Check exact match first
    if model_name in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[model_name]

    # Check partial match (for paths like sentence-transformers/all-mpnet-base-v2)
    for key, dim in EMBEDDING_DIMENSIONS.items():
        if key in model_name or model_name.endswith(key):
            return dim

    # Default to 768 (common dimension)
    return 768


class VectorStore:
    """Vector store using LanceDB with dynamic vector dimensions."""

    def __init__(
        self,
        persist_dir: str = ".piragi",
        embedding_model: str = "all-mpnet-base-v2",
        vector_dimension: Optional[int] = None,
    ) -> None:
        """
        Initialize the vector store.

        Args:
            persist_dir: Directory to persist the vector database
            embedding_model: Name of embedding model (used to infer dimension)
            vector_dimension: Explicit vector dimension (overrides model inference)
        """
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model

        # Determine vector dimension
        if vector_dimension is not None:
            self.vector_dimension = vector_dimension
        else:
            self.vector_dimension = get_embedding_dimension(embedding_model)

        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(persist_dir)
        self.table_name = "chunks"
        self.metadata_table_name = "source_metadata"
        self.table: Optional[Any] = None
        self.metadata_table: Optional[Any] = None

        # Track all chunk texts for hybrid search
        self._chunk_texts: List[str] = []

        # Initialize tables if they exist
        if self.table_name in self.db.table_names():
            self.table = self.db.open_table(self.table_name)
            # Load existing chunk texts
            try:
                results = self.table.to_pandas()
                self._chunk_texts = results["text"].tolist()
            except Exception:
                pass

        if self.metadata_table_name in self.db.table_names():
            self.metadata_table = self.db.open_table(self.metadata_table_name)

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to the vector store.

        Args:
            chunks: List of chunks with embeddings
        """
        if not chunks:
            return

        # Validate embeddings
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError("All chunks must have embeddings before adding to store")

        # Convert chunks to LanceDB format
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

        # Track chunk texts for hybrid search
        self._chunk_texts.extend([chunk.text for chunk in chunks])

        # Create or update table
        if self.table is None:
            self.table = self.db.create_table(self.table_name, data=data, mode="overwrite")
        else:
            self.table.add(data)

    def get_all_chunk_texts(self) -> List[str]:
        """
        Get all chunk texts for hybrid search indexing.

        Returns:
            List of all chunk texts
        """
        return self._chunk_texts

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_chunk_length: int = 100,
    ) -> List[Citation]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional metadata filters
            min_chunk_length: Minimum chunk length to return (filters out headers/navigation)

        Returns:
            List of citations
        """
        if self.table is None:
            return []

        # Request more results to account for filtering
        search_limit = top_k * 3  # Get 3x to ensure we have enough after filtering

        # Build query
        query = self.table.search(query_embedding).limit(search_limit)

        # Apply filters if provided
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                # Handle nested metadata filters
                filter_conditions.append(f"metadata['{key}'] = '{value}'")

            if filter_conditions:
                query = query.where(" AND ".join(filter_conditions))

        # Execute search
        results = query.to_list()

        # Convert to citations and filter out short chunks
        citations = []
        for result in results:
            chunk_text = result["text"]

            # Skip chunks that are too short (usually headers/navigation)
            if len(chunk_text) < min_chunk_length:
                continue

            citation = Citation(
                source=result["source"],
                chunk=chunk_text,
                score=1.0 - result["_distance"],  # Convert distance to similarity score
                metadata=result["metadata"],
            )
            citations.append(citation)

            # Stop once we have enough results
            if len(citations) >= top_k:
                break

        return citations

    def count(self) -> int:
        """Return the number of chunks in the store."""
        if self.table is None:
            return 0
        return self.table.count_rows()

    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks from a specific source.

        Args:
            source: Source file path or URL to delete

        Returns:
            Number of chunks deleted
        """
        if self.table is None:
            return 0

        # Count chunks before deletion
        count_before = self.table.count_rows()

        # Delete chunks matching the source
        self.table.delete(f"source = '{source}'")

        # Count chunks after deletion
        count_after = self.table.count_rows()

        return count_before - count_after

    def clear(self) -> None:
        """Clear all data from the store."""
        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)
            self.table = None
