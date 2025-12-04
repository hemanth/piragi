"""Vector store protocol definition."""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ..types import Chunk, Citation


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """
    Protocol for vector store implementations.

    Any class implementing these methods can be used as a store backend.

    Example:
        >>> class MyCustomStore:
        ...     def add_chunks(self, chunks: List[Chunk]) -> None:
        ...         # Store chunks with embeddings
        ...         pass
        ...
        ...     def search(self, query_embedding: List[float], top_k: int = 5,
        ...                filters: Optional[Dict[str, Any]] = None) -> List[Citation]:
        ...         # Return relevant citations
        ...         pass
        ...
        ...     def delete_by_source(self, source: str) -> int:
        ...         # Delete chunks from source, return count
        ...         pass
        ...
        ...     def count(self) -> int:
        ...         # Return total chunk count
        ...         pass
        ...
        ...     def clear(self) -> None:
        ...         # Clear all data
        ...         pass
        ...
        ...     def get_all_chunk_texts(self) -> List[str]:
        ...         # Return all chunk texts for hybrid search
        ...         pass
        >>>
        >>> kb = Ragi("./docs", store=MyCustomStore())
    """

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks with embeddings to the store.

        Args:
            chunks: List of Chunk objects with embeddings set
        """
        ...

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Citation]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of Citation objects with source, chunk text, score, and metadata
        """
        ...

    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks from a specific source.

        Args:
            source: Source identifier (file path or URL)

        Returns:
            Number of chunks deleted
        """
        ...

    def count(self) -> int:
        """
        Return the total number of chunks in the store.

        Returns:
            Chunk count
        """
        ...

    def clear(self) -> None:
        """Clear all data from the store."""
        ...

    def get_all_chunk_texts(self) -> List[str]:
        """
        Get all chunk texts for hybrid search indexing.

        Returns:
            List of all chunk texts
        """
        ...
