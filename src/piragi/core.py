"""Core Ragi class - the main interface for piragi."""

import logging
from typing import Any, Callable, Dict, List, Optional, Union

from .chunking import Chunker
from .embeddings import EmbeddingGenerator
from .loader import DocumentLoader
from .retrieval import Retriever
from .stores import VectorStoreProtocol, create_store
from .types import Answer, Chunk, Citation, Document, ChunkHook, DocumentHook
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
        >>> # Custom embedder (shared across instances)
        >>> from piragi import EmbeddingGenerator
        >>> embedder = EmbeddingGenerator(model="custom-model")
        >>> kb1 = Ragi("./docs1", embedder=embedder)
        >>> kb2 = Ragi("./docs2", embedder=embedder)
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
        embedder: Optional[EmbeddingGenerator] = None,
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
                    - min_chunk_length: Minimum chunk length in characters (default: 0)
                        Chunks shorter than this are filtered out. Useful for removing
                        garbage chunks like navigation elements, short headers, etc.
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
            embedder: Optional custom EmbeddingGenerator instance. If provided, this
                embedder will be used instead of creating a new one from config.
                Useful for sharing a single embedder across multiple Ragi instances
                to reduce memory usage and loading time.

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
        from .config import RagiConfig
        if isinstance(config, RagiConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = RagiConfig(**config)
        else:
            self.config = RagiConfig()

        cfg = config or {}

        # Store config for later use
        self._config = cfg

        from .llm_client import LLMClient
        self.llm_client = LLMClient(
            model=self.config.llm.model,
            api_key=self.config.llm.api_key,
            base_url=self.config.llm.base_url,
            temperature=self.config.llm.temperature,
        )

        # Initialize components
        self.loader = DocumentLoader()

        # Chunking configuration
        chunk_strategy = self.config.chunk.strategy

        if chunk_strategy == "semantic":
            from .semantic_chunking import SemanticChunker
            self.chunker = SemanticChunker(
                similarity_threshold=self.config.chunk.similarity_threshold,
                min_chunk_size=self.config.chunk.min_size,
                max_chunk_size=self.config.chunk.max_size,
            )
        elif chunk_strategy == "contextual":
            from .semantic_chunking import ContextualChunker
            self.chunker = ContextualChunker(
                llm_client=self.llm_client,
            )
        elif chunk_strategy == "hierarchical":
            from .semantic_chunking import HierarchicalChunker
            self.chunker = HierarchicalChunker(
                parent_chunk_size=self.config.chunk.parent_size,
                child_chunk_size=self.config.chunk.child_size,
            )
            self._use_hierarchical = True
        else:
            self.chunker = Chunker(
                chunk_size=self.config.chunk.size,
                chunk_overlap=self.config.chunk.overlap,
                min_chunk_length=self.config.chunk.min_chunk_length,
            )

        self._use_hierarchical = chunk_strategy == "hierarchical"

        # Embeddings - use provided embedder or create new one
        embed_model = self.config.embedding.model
        if embedder is not None:
            self.embedder = embedder
            # Try to get model name from the provided embedder for store configuration
            if hasattr(embedder, "model_name"):
                embed_model = embedder.model_name
        else:
            self.embedder = EmbeddingGenerator(
                model=embed_model,
                device=self.config.embedding.device,
                backend=self.config.embedding.backend,
                base_url=self.config.embedding.base_url,
                api_key=self.config.embedding.api_key,
                batch_size=self.config.embedding.batch_size,
            )

        # Vector store - supports multiple backends
        self.store = create_store(
            store=store,
            persist_dir=persist_dir,
            embedding_model=embed_model,
        )

        # Retrieval configuration
        self._use_hyde = self.config.retrieval.use_hyde
        self._use_hybrid_search = self.config.retrieval.use_hybrid_search
        self._use_cross_encoder = self.config.retrieval.use_cross_encoder

        # Initialize advanced retrieval components
        self._hyde = None
        self._hybrid_searcher = None
        self._cross_encoder = None

        if self._use_hyde:
            from .query_transform import HyDE
            self._hyde = HyDE(
                llm_client=self.llm_client,
            )

        if self._use_hybrid_search:
            from .hybrid_search import HybridSearcher
            self._hybrid_searcher = HybridSearcher(
                vector_weight=self.config.retrieval.vector_weight,
                bm25_weight=self.config.retrieval.bm25_weight,
                use_rrf=self.config.retrieval.use_rrf,
            )

        if self._use_cross_encoder:
            from .reranker import CrossEncoderReranker
            self._cross_encoder = CrossEncoderReranker(
                model_name=self.config.retrieval.cross_encoder_model,
                device=self.config.retrieval.cross_encoder_device or self.config.embedding.device,
                trust_remote_code=self.config.retrieval.trust_remote_code,
            )

        # LLM / Basic retriever
        self.retriever = Retriever(
            llm_client=self.llm_client,
            enable_reranking=self.config.llm.enable_reranking and not self._use_cross_encoder,
            enable_query_expansion=self.config.llm.enable_query_expansion and not self._use_hyde,
        )

        # State for filtering
        self._filters: Optional[Dict[str, Any]] = None

        # Processing hooks
        hooks_cfg = hooks or {}
        self._post_load_hook: Optional[DocumentHook] = hooks_cfg.get("post_load")
        self._post_chunk_hook: Optional[ChunkHook] = hooks_cfg.get("post_chunk")
        self._post_embed_hook: Optional[ChunkHook] = hooks_cfg.get("post_embed")

        # Auto-update setup
        self._auto_update_enabled = self.config.auto_update.enabled
        self._updater: Optional[AsyncUpdater] = None
        self._tracked_sources: Dict[str, Document] = {}

        if self._auto_update_enabled:
            interval = self.config.auto_update.interval
            workers = self.config.auto_update.workers

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

        from .pipelines.retrieval import RetrievalPipeline
        self._retrieval_pipeline = RetrievalPipeline(
            embedder=self.embedder,
            store=self.store,
            retriever=self.retriever,
            hyde=self._hyde,
            hybrid_searcher=self._hybrid_searcher,
            cross_encoder=self._cross_encoder,
            graph=self._graph,
            use_hierarchical=self._use_hierarchical,
        )

        # Load initial sources if provided
        if sources:
            self.add(sources)

    def add(
        self,
        sources: Union[str, List[str]],
        on_progress: Optional[Callable[[str], None]] = None,
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
        # We need to wrap post_load_hook to do auto update
        user_hook = self._post_load_hook
        def auto_update_hook(docs):
            if user_hook:
                docs = user_hook(docs)
                if not docs:
                    return docs
            
            if self._auto_update_enabled and self._updater:
                for doc in docs:
                    self._tracked_sources[doc.source] = doc
                    # Register with updater
                    if ChangeDetector.is_url(doc.source):
                        metadata = ChangeDetector.get_url_metadata(doc.source, doc.content)
                    else:
                        metadata = ChangeDetector.get_file_metadata(doc.source, doc.content)
                    self._updater.register_source(
                        doc.source, doc.content, check_interval=None
                    )
            return docs

        from .pipelines.ingestion import IngestionPipeline
        pipeline = IngestionPipeline(
            loader=self.loader,
            chunker=self.chunker,
            embedder=self.embedder,
            store=self.store,
            graph=self._graph if self._use_graph else None,
            hybrid_searcher=self._hybrid_searcher if self._use_hybrid_search else None,
            post_load_hook=auto_update_hook,
            post_chunk_hook=self._post_chunk_hook,
            post_embed_hook=self._post_embed_hook,
            use_hierarchical=self._use_hierarchical,
        )
        
        # Determine the llm_client to pass for graph extraction
        llm_client = None
        if self._use_graph and self._graph:
            if hasattr(self, "retriever") and self.retriever:
                llm_client = self.retriever.client
            elif hasattr(self, "llm_client"):
                llm_client = self.llm_client
                
        # Also, attach model to the client object temporarily for the pipeline
        if llm_client and not hasattr(llm_client, "model"):
            llm_client.model = self.config.llm.model
            
        def _progress(msg: str) -> None:
            if on_progress:
                on_progress(msg)
            else:
                logger.info(msg)

        pipeline.ingest(
            sources, 
            on_progress=_progress,
            llm_client=llm_client
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
        return self._ask_with_filters(query, top_k, system_prompt, None)

    def _ask_with_filters(
        self,
        query: str,
        top_k: int = 5,
        system_prompt: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Answer:
        """Internal implementation of ask with explicit filters."""
        return self._retrieval_pipeline.ask(
            query=query,
            top_k=top_k,
            system_prompt=system_prompt,
            filters=filters,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Citation]:
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
        return self._retrieve_with_filters(query, top_k, None)

    def _retrieve_with_filters(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Citation]:
        """Internal implementation of retrieve with explicit filters."""
        return self._retrieval_pipeline.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
        )

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
        return FilteredRagi(self, kwargs)

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

class FilteredRagi:
    """A wrapper for Ragi that binds specific metadata filters."""

    def __init__(self, ragi: Ragi, filters: Dict[str, Any]):
        self._ragi = ragi
        self._filters = filters

    def ask(self, query: str, top_k: int = 5, system_prompt: Optional[str] = None) -> Answer:
        return self._ragi._ask_with_filters(query, top_k, system_prompt, self._filters)

    def retrieve(self, query: str, top_k: int = 5) -> List[Citation]:
        return self._ragi._retrieve_with_filters(query, top_k, self._filters)

    def filter(self, **kwargs: Any) -> "FilteredRagi":
        new_filters = {**self._filters, **kwargs}
        return FilteredRagi(self._ragi, new_filters)

    def __call__(self, query: str, top_k: int = 5) -> Answer:
        return self.ask(query, top_k=top_k)
