"""Pinecone vector store implementation."""

from typing import Any, Dict, List, Optional
import uuid

from ..types import Chunk, Citation


class PineconeStore:
    """
    Pinecone vector store.

    Requires:
        pip install pinecone-client

    Examples:
        >>> store = PineconeStore(
        ...     api_key="your-api-key",
        ...     index_name="my-index",
        ...     environment="us-east-1"  # or your region
        ... )
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        environment: Optional[str] = None,
        namespace: str = "default",
        vector_dimension: int = 768,
    ) -> None:
        """
        Initialize Pinecone store.

        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            environment: Pinecone environment (for legacy, not needed for serverless)
            namespace: Namespace within the index
            vector_dimension: Dimension of embedding vectors
        """
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError(
                "PineconeStore requires pinecone-client. "
                "Install with: pip install piragi[pinecone]"
            )

        self.api_key = api_key
        self.index_name = index_name
        self.namespace = namespace
        self.vector_dimension = vector_dimension
        self._chunk_texts: List[str] = []
        self._chunk_map: Dict[str, str] = {}  # id -> text mapping

        # Initialize Pinecone
        self.pc = Pinecone(api_key=api_key)

        # Get or create index
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=vector_dimension,
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": environment or "us-east-1",
                    }
                },
            )

        self.index = self.pc.Index(index_name)

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks with embeddings to the store."""
        if not chunks:
            return

        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError("All chunks must have embeddings")

        vectors = []
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())

            vectors.append({
                "id": chunk_id,
                "values": chunk.embedding,
                "metadata": {
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                },
            })

            self._chunk_texts.append(chunk.text)
            self._chunk_map[chunk_id] = chunk.text

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_chunk_length: int = 100,
    ) -> List[Citation]:
        """Search for similar chunks."""
        # Build filter
        pinecone_filter = None
        if filters:
            pinecone_filter = {}
            for key, value in filters.items():
                pinecone_filter[key] = {"$eq": value}

        # Query with extra results to account for length filtering
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k * 3,
            include_metadata=True,
            namespace=self.namespace,
            filter=pinecone_filter,
        )

        citations = []
        for match in results.matches:
            metadata = match.metadata or {}
            chunk_text = metadata.get("text", "")

            if len(chunk_text) < min_chunk_length:
                continue

            # Extract source and clean metadata
            source = metadata.pop("text", "")
            source = metadata.pop("source", "unknown")
            chunk_index = metadata.pop("chunk_index", 0)

            citations.append(
                Citation(
                    source=source,
                    chunk=chunk_text,
                    score=match.score,
                    metadata=metadata,
                )
            )

            if len(citations) >= top_k:
                break

        return citations

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source."""
        # Pinecone requires fetching IDs first, then deleting
        # This is a limitation - we need to query to find matching IDs

        # Query for all vectors with this source
        # Note: This is inefficient for large datasets
        results = self.index.query(
            vector=[0.0] * self.vector_dimension,  # Dummy vector
            top_k=10000,
            include_metadata=True,
            namespace=self.namespace,
            filter={"source": {"$eq": source}},
        )

        if not results.matches:
            return 0

        ids_to_delete = [match.id for match in results.matches]
        self.index.delete(ids=ids_to_delete, namespace=self.namespace)

        # Update local cache
        for id in ids_to_delete:
            if id in self._chunk_map:
                text = self._chunk_map.pop(id)
                if text in self._chunk_texts:
                    self._chunk_texts.remove(text)

        return len(ids_to_delete)

    def count(self) -> int:
        """Return the number of chunks in the store."""
        stats = self.index.describe_index_stats()
        namespace_stats = stats.namespaces.get(self.namespace, {})
        return namespace_stats.get("vector_count", 0)

    def clear(self) -> None:
        """Clear all data from the store."""
        self.index.delete(delete_all=True, namespace=self.namespace)
        self._chunk_texts = []
        self._chunk_map = {}

    def get_all_chunk_texts(self) -> List[str]:
        """Get all chunk texts for hybrid search."""
        return self._chunk_texts
