"""Core Ragi class - the main interface for piragi."""

import logging
from typing import Any, Dict, List, Optional, Union

from .chunking import Chunker
from .embeddings import EmbeddingGenerator
from .loader import DocumentLoader
from .retrieval import Retriever
from .stores import VectorStoreProtocol, create_store
from .types import Answer, Chunk, Document, ChunkHook, DocumentHook
from .async_updater import AsyncUpdater
from .change_detection import ChangeDetector

logger = logging.getLogger(__name__)


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
        store: Union[str, Dict[str, Any], VectorStoreProtocol, None] = None,
        hooks: Optional[Dict[str, Any]] = None,
        graph: bool = False,
    ) -> None:
        """
        Initialize Ragi with optional document sources.

        Args:
            sources: File paths, URLs, or glob patterns to load
            persist_dir: Directory to persist vector database (used if store is None)
            config: Configuration dict with optional sections:
                - llm: LLM configuration
                    - model: Model name (default: "llama3.2")
                    - base_url: API base URL (default: "http://localhost:11434/v1")
                    - api_key: API key (default: "not-needed")
                - embedding: Embedding configuration
                    - model: Model name (default: "all-mpnet-base-v2")
                    - device: Device to use for local embedding models (default: None for auto-detect)
                    - base_url: API base URL for remote embeddings (optional)
                    - api_key: API key for remote embeddings (optional)
                    - batch_size: Number of chunks that is progressively processed when generating embeddings (default: 32)
                - chunk: Chunking configuration
                    - size: Chunk size in tokens (default: 512)
                    - overlap: Overlap in tokens (default: 50)
                    - strategy: Chunking strategy (default: "fixed")
                        Options: "fixed", "semantic", "contextual", "hierarchical"
                - retrieval: Retrieval configuration
                    - use_hyde: Enable HyDE (default: False)
                    - use_hybrid_search: Enable BM25 + vector hybrid (default: False)
                    - use_cross_encoder: Enable cross-encoder reranking (default: False)
                    - cross_encoder_device: Device to use for local cross-encoder reranking models (default: embedding's "device")
                    - cross_encoder_model: Model for cross-encoder (default: "cross-encoder/ms-marco-MiniLM-L-6-v2")
                    - trust_remote_code: Trust remote code for custom reranker models (default: False)
                        Required for models like "Alibaba-NLP/gte-multilingual-reranker-base"
                    - vector_weight: Weight for vector similarity in hybrid (default: 0.5)
                    - bm25_weight: Weight for BM25 in hybrid (default: 0.5)
                - auto_update: Auto-update configuration (enabled by default)
                    - enabled: Enable background updates (default: True)
                    - interval: Check interval in seconds (default: 300)
                    - workers: Number of background workers (default: 2)
            store: Vector store backend. Can be:
                - None: Use default LanceDB with persist_dir
                - str: URI (e.g., "s3://bucket/path", "postgres://...", "pinecone://...")
                - dict: Store config {"type": "pinecone", "api_key": "...", ...}
                - VectorStoreProtocol: Custom store implementation
            hooks: Processing hooks for custom transformations at each stage:
                - post_load: Called after loading documents, before chunking
                    Signature: (docs: List[Document]) -> List[Document]
                - post_chunk: Called after chunking, before embedding
                    Signature: (chunks: List[Chunk]) -> List[Chunk]
                - post_embed: Called after embedding, before storage
                    Signature: (chunks: List[Chunk]) -> List[Chunk]
            graph: Enable knowledge graph for entity/relationship extraction (default: False)
                Requires: pip install piragi[graph]

        Examples:
            >>> # Use defaults
            >>> kb = Ragi("./docs")
            >>>
            >>> # Custom LLM
            >>> kb = Ragi("./docs", config={
            ...     "llm": {"model": "gpt-4o-mini", "api_key": "sk-..."}
            ... })
            >>>
            >>> # Use S3-backed storage
            >>> kb = Ragi("./docs", store="s3://my-bucket/indices")
            >>>
            >>> # Use PostgreSQL with pgvector
            >>> kb = Ragi("./docs", store="postgres://user:pass@localhost/db")
            >>>
            >>> # Use Pinecone
            >>> from piragi.stores import PineconeStore
            >>> kb = Ragi("./docs", store=PineconeStore(api_key="...", index_name="my-index"))
            >>>
            >>> # Full advanced config
            >>> kb = Ragi("./docs", config={
            ...     "llm": {"model": "llama3.2"},
            ...     "embedding": {"device": "cuda"},
            ...     "chunk": {"size": 1024, "strategy": "semantic"},
            ...     "retrieval": {
            ...         "use_hyde": True,
            ...         "use_hybrid_search": True,
            ...         "use_cross_encoder": True,
            ...     }
            ... })
        """
        # Initialize config
        cfg = config or {}

        # Store config for later use
        self._config = cfg

        # Initialize components
        self.loader = DocumentLoader()

        # Chunking configuration
        chunk_cfg = cfg.get("chunk", {})
        chunk_strategy = chunk_cfg.get("strategy", "fixed")

        if chunk_strategy == "semantic":
            from .semantic_chunking import SemanticChunker
            self.chunker = SemanticChunker(
                similarity_threshold=chunk_cfg.get("similarity_threshold", 0.5),
                min_chunk_size=chunk_cfg.get("min_size", 100),
                max_chunk_size=chunk_cfg.get("max_size", 2000),
            )
        elif chunk_strategy == "contextual":
            from .semantic_chunking import ContextualChunker
            llm_cfg = cfg.get("llm", {})
            self.chunker = ContextualChunker(
                model=llm_cfg.get("model", "llama3.2"),
                api_key=llm_cfg.get("api_key"),
                base_url=llm_cfg.get("base_url"),
            )
        elif chunk_strategy == "hierarchical":
            from .semantic_chunking import HierarchicalChunker
            self.chunker = HierarchicalChunker(
                parent_chunk_size=chunk_cfg.get("parent_size", 2000),
                child_chunk_size=chunk_cfg.get("child_size", 400),
            )
            self._use_hierarchical = True
        else:
            self.chunker = Chunker(
                chunk_size=chunk_cfg.get("size", 512),
                chunk_overlap=chunk_cfg.get("overlap", 50),
            )

        self._use_hierarchical = chunk_strategy == "hierarchical"

        # Embeddings
        embed_cfg = cfg.get("embedding", {})
        embed_model = embed_cfg.get("model", "all-mpnet-base-v2")
        self.embedder = EmbeddingGenerator(
            model=embed_model,
            device=embed_cfg.get("device"),
            base_url=embed_cfg.get("base_url"),
            api_key=embed_cfg.get("api_key"),
            batch_size=embed_cfg.get("batch_size", 32),
        )

        # Vector store - supports multiple backends
        self.store = create_store(
            store=store,
            persist_dir=persist_dir,
            embedding_model=embed_model,
        )

        # Retrieval configuration
        retrieval_cfg = cfg.get("retrieval", {})
        self._use_hyde = retrieval_cfg.get("use_hyde", False)
        self._use_hybrid_search = retrieval_cfg.get("use_hybrid_search", False)
        self._use_cross_encoder = retrieval_cfg.get("use_cross_encoder", False)

        # Initialize advanced retrieval components
        self._hyde = None
        self._hybrid_searcher = None
        self._cross_encoder = None

        llm_cfg = cfg.get("llm", {})

        if self._use_hyde:
            from .query_transform import HyDE
            self._hyde = HyDE(
                model=llm_cfg.get("model", "llama3.2"),
                api_key=llm_cfg.get("api_key"),
                base_url=llm_cfg.get("base_url"),
            )

        if self._use_hybrid_search:
            from .hybrid_search import HybridSearcher
            self._hybrid_searcher = HybridSearcher(
                vector_weight=retrieval_cfg.get("vector_weight", 0.5),
                bm25_weight=retrieval_cfg.get("bm25_weight", 0.5),
                use_rrf=retrieval_cfg.get("use_rrf", True),
            )

        if self._use_cross_encoder:
            from .reranker import CrossEncoderReranker
            self._cross_encoder = CrossEncoderReranker(
                model_name=retrieval_cfg.get(
                    "cross_encoder_model",
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                ),
                device=retrieval_cfg.get("cross_encoder_device", embed_cfg.get("device")),
                trust_remote_code=retrieval_cfg.get("trust_remote_code", False),
            )

        # LLM / Basic retriever
        self.retriever = Retriever(
            model=llm_cfg.get("model", "llama3.2"),
            api_key=llm_cfg.get("api_key"),
            base_url=llm_cfg.get("base_url"),
            temperature=llm_cfg.get("temperature", 0.1),
            enable_reranking=llm_cfg.get("enable_reranking", True) and not self._use_cross_encoder,
            enable_query_expansion=llm_cfg.get("enable_query_expansion", True) and not self._use_hyde,
        )

        # State for filtering
        self._filters: Optional[Dict[str, Any]] = None

        # Processing hooks
        hooks_cfg = hooks or {}
        self._post_load_hook: Optional[DocumentHook] = hooks_cfg.get("post_load")
        self._post_chunk_hook: Optional[ChunkHook] = hooks_cfg.get("post_chunk")
        self._post_embed_hook: Optional[ChunkHook] = hooks_cfg.get("post_embed")

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

        # Knowledge graph setup
        self._use_graph = graph
        self._graph = None

        if graph:
            from .knowledge_graph import KnowledgeGraph
            import os
            graph_path = os.path.join(persist_dir, "graph.json")
            self._graph = KnowledgeGraph(persist_path=graph_path)

        # Load initial sources if provided
        if sources:
            self.add(sources)

    def add(
        self,
        sources: Union[str, List[str]],
        on_progress: Optional[callable] = None,
    ) -> "Ragi":
        """
        Add documents to the knowledge base.

        Args:
            sources: File paths, URLs, or glob patterns
            on_progress: Optional callback for progress updates.
                Called with a string message at each stage.

        Returns:
            Self for chaining
        """
        def _progress(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        # Load documents
        _progress("Discovering files...")
        documents = self.loader.load(sources)
        _progress(f"Found {len(documents)} documents")

        # Hook: post_load - transform documents before chunking
        if self._post_load_hook:
            documents = self._post_load_hook(documents)

        # Chunk documents
        all_chunks = []
        for i, doc in enumerate(documents, 1):
            _progress(f"Chunking {i}/{len(documents)}: {doc.source}")
            if self._use_hierarchical:
                # Hierarchical chunking returns (parents, children)
                # We store children for retrieval but keep parent context
                parent_chunks, child_chunks = self.chunker.chunk_document(doc)
                all_chunks.extend(child_chunks)
            else:
                chunks = self.chunker.chunk_document(doc)
                all_chunks.extend(chunks)

        _progress(f"Created {len(all_chunks)} chunks")

        # Hook: post_chunk - transform chunks before embedding
        if self._post_chunk_hook:
            all_chunks = self._post_chunk_hook(all_chunks)

        # Generate embeddings with per-batch progress
        _progress(f"Generating embeddings for {len(all_chunks)} chunks...")
        chunks_with_embeddings = self.embedder.embed_chunks(
            all_chunks,
            on_progress=_progress,
        )
        _progress("Embeddings complete")

        # Hook: post_embed - transform chunks before storage (e.g., entity extraction)
        if self._post_embed_hook:
            chunks_with_embeddings = self._post_embed_hook(chunks_with_embeddings)

        # Store in vector database
        _progress("Storing chunks...")
        self.store.add_chunks(chunks_with_embeddings)

        # Extract entities and relationships for knowledge graph
        if self._use_graph and self._graph:
            _progress("Extracting knowledge graph...")
            llm_cfg = self._config.get("llm", {})
            for chunk in chunks_with_embeddings:
                self._graph.extract_and_add(
                    text=chunk.text,
                    llm_client=self.retriever._client,
                    model=llm_cfg.get("model", "llama3.2"),
                )
            self._graph.save()

        # Index for hybrid search if enabled
        if self._use_hybrid_search and self._hybrid_searcher:
            chunk_texts = self.store.get_all_chunk_texts()
            self._hybrid_searcher.index_chunks(chunk_texts)

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

        _progress("Done")
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
        # Validate query
        if not query or not query.strip():
            return Answer(
                text="Please provide a valid question.",
                citations=[],
                query=query,
            )

        # Determine queries to use for retrieval
        if self._use_hyde and self._hyde:
            # HyDE: generate hypothetical document and use that for retrieval
            try:
                hypothetical_doc = self._hyde.transform_query(query)
                query_variations = [hypothetical_doc]
                logger.debug(f"HyDE generated: {hypothetical_doc[:100]}...")
            except Exception as e:
                logger.warning(f"HyDE failed: {e}, falling back to regular query")
                query_variations = self.retriever.expand_query(query)
        else:
            # Standard query expansion
            query_variations = self.retriever.expand_query(query)

        # Search with all query variations and merge results
        all_citations = []
        seen_chunks = set()

        # Get more candidates if we're using cross-encoder reranking
        search_top_k = top_k * 4 if self._use_cross_encoder else top_k

        for query_var in query_variations:
            # Generate query embedding
            query_embedding = self.embedder.embed_query(query_var)

            # Search for relevant chunks
            citations = self.store.search(
                query_embedding=query_embedding,
                top_k=search_top_k,
                filters=self._filters,
            )

            # Add unique citations
            for citation in citations:
                chunk_id = (citation.source, citation.chunk[:100])
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    all_citations.append(citation)

        # Apply hybrid search if enabled
        if self._use_hybrid_search and self._hybrid_searcher:
            try:
                all_citations = self._hybrid_searcher.search(
                    query=query,  # Use original query for BM25
                    vector_citations=all_citations,
                    top_k=search_top_k,
                )
            except Exception as e:
                logger.warning(f"Hybrid search failed: {e}")
                # Continue with vector-only results

        # Apply cross-encoder reranking if enabled
        if self._use_cross_encoder and self._cross_encoder:
            try:
                all_citations = self._cross_encoder.rerank(
                    query=query,  # Use original query for reranking
                    citations=all_citations,
                    top_k=top_k,
                )
            except Exception as e:
                logger.warning(f"Cross-encoder reranking failed: {e}")
                # Fall back to score-based sorting
                all_citations.sort(key=lambda c: c.score, reverse=True)
                all_citations = all_citations[:top_k]
        else:
            # Sort by score and take top_k
            all_citations.sort(key=lambda c: c.score, reverse=True)
            all_citations = all_citations[:top_k]

        final_citations = all_citations

        # For hierarchical chunks, expand to parent context
        if self._use_hierarchical:
            final_citations = self._expand_to_parent_context(final_citations)

        # Add graph context if enabled
        graph_context = ""
        if self._use_graph and self._graph:
            graph_context = self._graph.to_context(query, max_triples=10)

        # Build system prompt with graph context
        final_system_prompt = system_prompt
        if graph_context:
            if final_system_prompt:
                final_system_prompt = f"{final_system_prompt}\n\n{graph_context}"
            else:
                final_system_prompt = graph_context

        # Generate answer
        answer = self.retriever.generate_answer(
            query=query,
            citations=final_citations,
            system_prompt=final_system_prompt,
        )

        # Reset filters after use
        self._filters = None

        return answer

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List:
        """
        Retrieve relevant chunks without LLM generation.

        Use this when you want to handle LLM generation yourself or integrate
        with other frameworks (LangChain, LlamaIndex, etc.).

        Args:
            query: Search query
            top_k: Number of relevant chunks to retrieve

        Returns:
            List of Citation objects with text, source, score, and metadata

        Examples:
            >>> chunks = kb.retrieve("How does authentication work?")
            >>> for chunk in chunks:
            ...     print(chunk.text, chunk.source, chunk.score)
            >>>
            >>> # Use with your own LLM
            >>> context = "\\n".join(c.chunk for c in chunks)
            >>> response = your_llm(f"Based on: {context}\\n\\nQ: {query}")
        """
        from .types import Citation

        # Validate query
        if not query or not query.strip():
            return []

        # Determine queries to use for retrieval
        if self._use_hyde and self._hyde:
            try:
                hypothetical_doc = self._hyde.transform_query(query)
                query_variations = [hypothetical_doc]
                logger.debug(f"HyDE generated: {hypothetical_doc[:100]}...")
            except Exception as e:
                logger.warning(f"HyDE failed: {e}, falling back to regular query")
                query_variations = [query]
        else:
            # Use original query (skip expansion for pure retrieval)
            query_variations = [query]

        # Search with all query variations and merge results
        all_citations = []
        seen_chunks = set()

        # Get more candidates if we're using cross-encoder reranking
        search_top_k = top_k * 4 if self._use_cross_encoder else top_k

        for query_var in query_variations:
            # Generate query embedding
            query_embedding = self.embedder.embed_query(query_var)

            # Search for relevant chunks
            citations = self.store.search(
                query_embedding=query_embedding,
                top_k=search_top_k,
                filters=self._filters,
            )

            # Add unique citations
            for citation in citations:
                chunk_id = (citation.source, citation.chunk[:100])
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    all_citations.append(citation)

        # Apply hybrid search if enabled
        if self._use_hybrid_search and self._hybrid_searcher:
            try:
                all_citations = self._hybrid_searcher.search(
                    query=query,
                    vector_citations=all_citations,
                    top_k=search_top_k,
                )
            except Exception as e:
                logger.warning(f"Hybrid search failed: {e}")

        # Apply cross-encoder reranking if enabled
        if self._use_cross_encoder and self._cross_encoder:
            try:
                all_citations = self._cross_encoder.rerank(
                    query=query,
                    citations=all_citations,
                    top_k=top_k,
                )
            except Exception as e:
                logger.warning(f"Cross-encoder reranking failed: {e}")
                all_citations.sort(key=lambda c: c.score, reverse=True)
                all_citations = all_citations[:top_k]
        else:
            all_citations.sort(key=lambda c: c.score, reverse=True)
            all_citations = all_citations[:top_k]

        # For hierarchical chunks, expand to parent context
        if self._use_hierarchical:
            all_citations = self._expand_to_parent_context(all_citations)

        # Reset filters after use
        self._filters = None

        return all_citations

    def _expand_to_parent_context(self, citations: List) -> List:
        """
        Expand child chunks to include parent context.

        For hierarchical chunking, we retrieve using precise child chunks
        but include the larger parent context for answer generation.
        """
        from .types import Citation

        expanded = []
        for citation in citations:
            if "parent_text" in citation.metadata:
                # Replace chunk with parent context
                expanded.append(
                    Citation(
                        source=citation.source,
                        chunk=citation.metadata["parent_text"],
                        score=citation.score,
                        metadata={
                            k: v for k, v in citation.metadata.items()
                            if k != "parent_text"
                        },
                    )
                )
            else:
                expanded.append(citation)

        return expanded

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

    @property
    def graph(self):
        """
        Access the knowledge graph for direct queries.

        Returns:
            KnowledgeGraph instance if graph=True was set, None otherwise

        Examples:
            >>> kb = Ragi("./docs", graph=True)
            >>> kb.graph.entities()  # List all entities
            >>> kb.graph.neighbors("Alice")  # Get related entities
            >>> kb.graph.triples()  # Get all (subject, predicate, object) triples
        """
        return self._graph

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

        # Clear knowledge graph if enabled
        if self._graph:
            self._graph.clear()

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "_updater") and self._updater:
            self._updater.stop()
