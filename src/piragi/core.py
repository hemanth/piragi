"""Core Ragi class - the main interface for piragi."""

from typing import Any, Dict, List, Optional, Union

from .chunking import Chunker
from .embeddings import EmbeddingGenerator
from .loader import DocumentLoader
from .retrieval import Retriever
from .store import VectorStore
from .types import Answer, Document
from .async_updater import AsyncUpdater
from .change_detection import ChangeDetector


class Ragi:
    """
    Zero-setup RAG library with auto-chunking, embeddings, and smart citations.

    Examples:
        >>> from piragi import Ragi
        >>>
        >>> # Simple - uses free local models
        >>> kb = Ragi("./docs")
        >>>
        >>> # Custom config
        >>> kb = Ragi("./docs", config={
        ...     "llm": {"model": "gpt-4o-mini"},
        ...     "embedding": {"device": "cuda"}
        ... })
        >>>
        >>> # Ask questions
        >>> answer = kb.ask("How do I install this?")
        >>> print(answer.text)
        >>>
        >>> # Callable shorthand
        >>> answer = kb("What's the API?")
    """

    def __init__(
        self,
        sources: Union[str, List[str], None] = None,
        persist_dir: str = ".piragi",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize Ragi with optional document sources.

        Args:
            sources: File paths, URLs, or glob patterns to load
            persist_dir: Directory to persist vector database
            config: Configuration dict with optional sections:
                - llm: LLM configuration
                    - model: Model name (default: "llama3.2")
                    - base_url: API base URL (default: "http://localhost:11434/v1")
                    - api_key: API key (default: "not-needed")
                - embedding: Embedding configuration
                    - model: Model name (default: "all-mpnet-base-v2")
                    - device: Device to use for local models (default: None for auto-detect)
                    - base_url: API base URL for remote embeddings (optional)
                    - api_key: API key for remote embeddings (optional)
                - chunk: Chunking configuration
                    - size: Chunk size in tokens (default: 512)
                    - overlap: Overlap in tokens (default: 50)
                - auto_update: Auto-update configuration (enabled by default)
                    - enabled: Enable background updates (default: True)
                    - interval: Check interval in seconds (default: 300)
                    - workers: Number of background workers (default: 2)

        Examples:
            >>> # Use defaults
            >>> kb = Ragi("./docs")
            >>>
            >>> # Custom LLM
            >>> kb = Ragi("./docs", config={
            ...     "llm": {"model": "gpt-4o-mini", "api_key": "sk-..."}
            ... })
            >>>
            >>> # Full config
            >>> kb = Ragi("./docs", config={
            ...     "llm": {"model": "llama3.2"},
            ...     "embedding": {"device": "cuda"},
            ...     "chunk": {"size": 1024, "overlap": 200}
            ... })
        """
        # Initialize config
        cfg = config or {}

        # Initialize components
        self.loader = DocumentLoader()

        # Chunking
        chunk_cfg = cfg.get("chunk", {})
        self.chunker = Chunker(
            chunk_size=chunk_cfg.get("size", 512),
            chunk_overlap=chunk_cfg.get("overlap", 50),
        )

        # Embeddings
        embed_cfg = cfg.get("embedding", {})
        self.embedder = EmbeddingGenerator(
            model=embed_cfg.get("model", "all-mpnet-base-v2"),
            device=embed_cfg.get("device"),
            base_url=embed_cfg.get("base_url"),
            api_key=embed_cfg.get("api_key"),
        )

        # Vector store
        self.store = VectorStore(persist_dir=persist_dir)

        # LLM
        llm_cfg = cfg.get("llm", {})
        self.retriever = Retriever(
            model=llm_cfg.get("model", "llama3.2"),
            api_key=llm_cfg.get("api_key"),
            base_url=llm_cfg.get("base_url"),
            temperature=llm_cfg.get("temperature", 0.1),
            enable_reranking=llm_cfg.get("enable_reranking", True),
            enable_query_expansion=llm_cfg.get("enable_query_expansion", True),
        )

        # State for filtering
        self._filters: Optional[Dict[str, Any]] = None

        # Auto-update setup
        auto_update_cfg = cfg.get("auto_update", {})
        self._auto_update_enabled = auto_update_cfg.get("enabled", True)
        self._updater: Optional[AsyncUpdater] = None
        self._tracked_sources: Dict[str, Document] = {}

        if self._auto_update_enabled:
            interval = auto_update_cfg.get("interval", 300.0)
            workers = auto_update_cfg.get("workers", 2)

            self._updater = AsyncUpdater(
                refresh_callback=self._background_refresh,
                check_interval=interval,
                max_workers=workers,
            )
            self._updater.start()

        # Load initial sources if provided
        if sources:
            self.add(sources)

    def add(self, sources: Union[str, List[str]]) -> "Ragi":
        """
        Add documents to the knowledge base.

        Args:
            sources: File paths, URLs, or glob patterns

        Returns:
            Self for chaining
        """
        # Load documents
        documents = self.loader.load(sources)

        # Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        # Generate embeddings
        chunks_with_embeddings = self.embedder.embed_chunks(all_chunks)

        # Store in vector database
        self.store.add_chunks(chunks_with_embeddings)

        # Register sources for auto-update
        if self._auto_update_enabled and self._updater:
            for doc in documents:
                self._tracked_sources[doc.source] = doc
                # Register with updater
                if ChangeDetector.is_url(doc.source):
                    metadata = ChangeDetector.get_url_metadata(doc.source, doc.content)
                else:
                    metadata = ChangeDetector.get_file_metadata(doc.source, doc.content)

                self._updater.register_source(
                    doc.source, doc.content, check_interval=None
                )

        return self

    def _background_refresh(self, source: Union[str, List[str]]) -> None:
        """
        Internal method called by background updater.
        Refreshes sources without user interaction.

        Args:
            source: Source(s) to refresh
        """
        # This is called from background thread, so be careful with state
        self.refresh(source)

    def ask(
        self,
        query: str,
        top_k: int = 5,
        system_prompt: Optional[str] = None,
    ) -> Answer:
        """
        Ask a question and get an answer with citations.

        Args:
            query: Question to ask
            top_k: Number of relevant chunks to retrieve
            system_prompt: Optional custom system prompt for answer generation

        Returns:
            Answer with citations
        """
        # Expand query if enabled
        query_variations = self.retriever.expand_query(query)

        # Search with all query variations and merge results
        all_citations = []
        seen_chunks = set()

        for query_var in query_variations:
            # Generate query embedding
            query_embedding = self.embedder.embed_query(query_var)

            # Search for relevant chunks
            citations = self.store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=self._filters,
            )

            # Add unique citations
            for citation in citations:
                chunk_id = (citation.source, citation.chunk[:100])  # Use source + chunk preview as ID
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    all_citations.append(citation)

        # Sort by score and take top_k
        all_citations.sort(key=lambda c: c.score, reverse=True)
        final_citations = all_citations[:top_k]

        # Generate answer
        answer = self.retriever.generate_answer(
            query=query,
            citations=final_citations,
            system_prompt=system_prompt,
        )

        # Reset filters after use
        self._filters = None

        return answer

    def filter(self, **kwargs: Any) -> "Ragi":
        """
        Filter documents by metadata for the next query.

        Args:
            **kwargs: Metadata key-value pairs to filter by

        Returns:
            Self for chaining

        Examples:
            >>> kb.filter(type="api").ask("How does auth work?")
            >>> kb.filter(source="docs/guide.pdf").ask("What's in the guide?")
        """
        self._filters = kwargs
        return self

    def __call__(self, query: str, top_k: int = 5) -> Answer:
        """
        Callable shorthand for ask().

        Args:
            query: Question to ask
            top_k: Number of relevant chunks to retrieve

        Returns:
            Answer with citations
        """
        return self.ask(query, top_k=top_k)

    def count(self) -> int:
        """Return the number of chunks in the knowledge base."""
        return self.store.count()

    def refresh(self, sources: Union[str, List[str]]) -> "Ragi":
        """
        Refresh specific sources by deleting old chunks and re-adding.
        Useful when documents have been updated.

        Args:
            sources: File paths, URLs, or glob patterns to refresh

        Returns:
            Self for chaining

        Examples:
            >>> # Refresh a single file
            >>> kb.refresh("./docs/api.md")
            >>>
            >>> # Refresh multiple files
            >>> kb.refresh(["./docs/*.pdf", "./README.md"])
        """
        # Load documents to get their actual source paths
        documents = self.loader.load(sources)

        # Delete old chunks for each source
        for doc in documents:
            deleted = self.store.delete_by_source(doc.source)

        # Re-add the documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        # Generate embeddings
        chunks_with_embeddings = self.embedder.embed_chunks(all_chunks)

        # Store in vector database
        self.store.add_chunks(chunks_with_embeddings)

        return self

    def clear(self) -> None:
        """Clear all data from the knowledge base."""
        # Stop auto-updater if running
        if self._updater:
            self._updater.stop()
            self._tracked_sources.clear()

        self.store.clear()

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "_updater") and self._updater:
            self._updater.stop()
