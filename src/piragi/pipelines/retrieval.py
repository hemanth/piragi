import logging
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class RetrievalPipeline:
    """Composable retrieval pipeline: Transform -> Search -> Rerank -> Generate."""
    
    def __init__(self, embedder, store, retriever,
                 hyde=None, hybrid_searcher=None, cross_encoder=None,
                 graph=None, use_hierarchical=False, max_parallel_queries=4,
                 mode="dense", query_rewriter=None, bm25_index=None, on_the_fly_index=None):
        self.embedder = embedder
        self.store = store
        self.retriever = retriever
        self.hyde = hyde
        self.hybrid_searcher = hybrid_searcher
        self.cross_encoder = cross_encoder
        self.graph = graph
        self.use_hierarchical = use_hierarchical
        self.max_parallel_queries = max_parallel_queries
        self.mode = mode
        self.query_rewriter = query_rewriter
        self.bm25_index = bm25_index
        self.on_the_fly_index = on_the_fly_index
        
    def _expand_to_parent_context(self, citations: List) -> List:
        """
        Expand child chunks to include parent context.
        """
        from ..types import Citation
        
        expanded = []
        for citation in citations:
            if "parent_text" in citation.metadata:
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
    
    def _search_single_query(self, query_var, search_top_k, filters):
        """Search for a single query variation. Thread-safe."""
        query_embedding = self.embedder.embed_query(query_var)
        return self.store.search(
            query_embedding=query_embedding,
            top_k=search_top_k,
            filters=filters,
        )

    def _rewrite_if_enabled(self, query: str) -> str:
        """Apply agentic query rewriting for lexical search, if configured."""
        if not self.query_rewriter:
            return query
        try:
            return self.query_rewriter.rewrite(query)
        except Exception as e:
            logger.warning("Query rewrite failed: {}, using original query".format(e))
            return query

    def _non_dense_search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List:
        """Retrieval for bm25_only / on_the_fly / hot_cold modes (no pre-built vector index)."""
        search_query = self._rewrite_if_enabled(query)

        if self.mode == "bm25_only":
            citations = self.bm25_index.search(search_query, top_k=top_k)
        elif self.mode == "on_the_fly":
            citations = self.on_the_fly_index.search(self.embedder, search_query, top_k=top_k)
        elif self.mode == "hot_cold":
            hot_citations = self.store.search(
                query_embedding=self.embedder.embed_query(search_query),
                top_k=top_k,
                filters=filters,
            )
            cold_citations = (
                self.on_the_fly_index.search(self.embedder, search_query, top_k=top_k)
                if self.on_the_fly_index and self.on_the_fly_index.count()
                else []
            )
            citations = sorted(hot_citations + cold_citations, key=lambda c: c.score, reverse=True)[:top_k]
        else:
            citations = []

        if self.cross_encoder:
            try:
                citations = self.cross_encoder.rerank(query=query, citations=citations, top_k=top_k)
            except Exception as e:
                logger.warning("Cross-encoder reranking failed: {}".format(e))
                citations = citations[:top_k]

        if self.use_hierarchical:
            citations = self._expand_to_parent_context(citations)

        return citations

    def retrieve(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None):
        """Retrieve relevant chunks for a query.

        Steps:
        1. Transform query (HyDE / multi-query expansion)
        2. Vector search (with optional filters)
        3. Hybrid search fusion (if enabled)
        4. Cross-encoder reranking (if enabled)
        5. Hierarchical context expansion (if enabled)

        Returns:
            List of Citation objects
        """
        from ..types import Citation

        # Validate query
        if not query or not query.strip():
            return []

        if self.mode in ("bm25_only", "on_the_fly", "hot_cold"):
            return self._non_dense_search(query, top_k, filters)

        # Determine queries to use for retrieval
        if self.hyde:
            try:
                hypothetical_doc = self.hyde.transform_query(query)
                query_variations = [hypothetical_doc]
                logger.debug("HyDE generated: {}...".format(hypothetical_doc[:100]))
            except Exception as e:
                logger.warning("HyDE failed: {}, falling back to regular query".format(e))
                query_variations = [query]
        else:
            # Use original query (skip expansion for pure retrieval)
            query_variations = [query]

        # Search with all query variations and merge results
        all_citations = []
        seen_chunks = set()

        # Get more candidates if we're using cross-encoder reranking
        search_top_k = top_k * 4 if self.cross_encoder else top_k

        # Calculate max workers based on variations and configured limit
        workers = min(len(query_variations), max(1, self.max_parallel_queries))
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._search_single_query, qv, search_top_k, filters): qv
                for qv in query_variations
            }
            for future in as_completed(futures):
                try:
                    citations = future.result()
                    for citation in citations:
                        chunk_id = (citation.source, citation.chunk[:100])
                        if chunk_id not in seen_chunks:
                            seen_chunks.add(chunk_id)
                            all_citations.append(citation)
                except Exception as e:
                    logger.warning("Query variation search failed: {}".format(str(e)))

        # Apply hybrid search if enabled
        if self.hybrid_searcher:
            try:
                all_citations = self.hybrid_searcher.search(
                    query=self._rewrite_if_enabled(query),
                    vector_citations=all_citations,
                    top_k=search_top_k,
                )
            except Exception as e:
                logger.warning("Hybrid search failed: {}".format(e))

        # Apply cross-encoder reranking if enabled
        if self.cross_encoder:
            try:
                all_citations = self.cross_encoder.rerank(
                    query=query,
                    citations=all_citations,
                    top_k=top_k,
                )
            except Exception as e:
                logger.warning("Cross-encoder reranking failed: {}".format(e))
                all_citations.sort(key=lambda c: c.score, reverse=True)
                all_citations = all_citations[:top_k]
        else:
            all_citations.sort(key=lambda c: c.score, reverse=True)
            all_citations = all_citations[:top_k]

        # For hierarchical chunks, expand to parent context
        if self.use_hierarchical:
            all_citations = self._expand_to_parent_context(all_citations)

        return all_citations
    
    def _generate_answer_from_citations(self, query: str, citations: List, system_prompt: Optional[str] = None):
        """Shared tail of ask(): graph context + LLM generation from already-retrieved citations."""
        graph_context = ""
        if self.graph:
            graph_context = self.graph.to_context(query, max_triples=10)

        final_system_prompt = system_prompt
        if graph_context:
            if final_system_prompt:
                final_system_prompt = "{}\n\n{}".format(final_system_prompt, graph_context)
            else:
                final_system_prompt = graph_context

        return self.retriever.generate_answer(
            query=query,
            citations=citations,
            system_prompt=final_system_prompt,
        )

    def ask(self, query: str, top_k: int = 5, system_prompt: Optional[str] = None, filters: Optional[Dict[str, Any]] = None):
        """Retrieve and generate an answer.

        Steps:
        1. Retrieve relevant chunks
        2. Build context from chunks
        3. Generate answer via LLM

        Returns:
            Answer object with citations
        """
        from ..types import Answer

        # Validate query
        if not query or not query.strip():
            return Answer(
                text="Please provide a valid question.",
                citations=[],
                query=query,
            )

        if self.mode in ("bm25_only", "on_the_fly", "hot_cold"):
            citations = self._non_dense_search(query, top_k, filters)
            return self._generate_answer_from_citations(query, citations, system_prompt)

        # Determine queries to use for retrieval
        if self.hyde:
            # HyDE: generate hypothetical document and use that for retrieval
            try:
                hypothetical_doc = self.hyde.transform_query(query)
                query_variations = [hypothetical_doc]
                logger.debug("HyDE generated: {}...".format(hypothetical_doc[:100]))
            except Exception as e:
                logger.warning("HyDE failed: {}, falling back to regular query".format(e))
                query_variations = self.retriever.expand_query(query)
        else:
            # Standard query expansion
            query_variations = self.retriever.expand_query(query)

        # Search with all query variations and merge results
        all_citations = []
        seen_chunks = set()

        # Get more candidates if we're using cross-encoder reranking
        search_top_k = top_k * 4 if self.cross_encoder else top_k

        # Calculate max workers based on variations and configured limit
        workers = min(len(query_variations), max(1, self.max_parallel_queries))
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._search_single_query, qv, search_top_k, filters): qv
                for qv in query_variations
            }
            for future in as_completed(futures):
                try:
                    citations = future.result()
                    for citation in citations:
                        chunk_id = (citation.source, citation.chunk[:100])
                        if chunk_id not in seen_chunks:
                            seen_chunks.add(chunk_id)
                            all_citations.append(citation)
                except Exception as e:
                    logger.warning("Query variation search failed: {}".format(str(e)))

        # Apply hybrid search if enabled
        if self.hybrid_searcher:
            try:
                all_citations = self.hybrid_searcher.search(
                    query=self._rewrite_if_enabled(query),  # Use rewritten query for BM25
                    vector_citations=all_citations,
                    top_k=search_top_k,
                )
            except Exception as e:
                logger.warning("Hybrid search failed: {}".format(e))
                # Continue with vector-only results

        # Apply cross-encoder reranking if enabled
        if self.cross_encoder:
            try:
                all_citations = self.cross_encoder.rerank(
                    query=query,  # Use original query for reranking
                    citations=all_citations,
                    top_k=top_k,
                )
            except Exception as e:
                logger.warning("Cross-encoder reranking failed: {}".format(e))
                # Fall back to score-based sorting
                all_citations.sort(key=lambda c: c.score, reverse=True)
                all_citations = all_citations[:top_k]
        else:
            # Sort by score and take top_k
            all_citations.sort(key=lambda c: c.score, reverse=True)
            all_citations = all_citations[:top_k]

        final_citations = all_citations

        # For hierarchical chunks, expand to parent context
        if self.use_hierarchical:
            final_citations = self._expand_to_parent_context(final_citations)

        return self._generate_answer_from_citations(query, final_citations, system_prompt)
