import logging
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..telemetry import classify_store_provider, get_telemetry, wrap_with_context

logger = logging.getLogger(__name__)

class RetrievalPipeline:
    """Composable retrieval pipeline: Transform -> Search -> Rerank -> Generate."""
    
    def __init__(self, embedder, store, retriever,
                 hyde=None, hybrid_searcher=None, cross_encoder=None,
                 graph=None, use_hierarchical=False, max_parallel_queries=4,
                 mode="dense", query_rewriter=None, bm25_index=None, on_the_fly_index=None,
                 evidence_judge=None, candidate_k=20,
                 auto_merge_parents=True,
                 step_back=None,
                 context_ordering: str = "u_shaped",
                 prune_unreferenced_citations: bool = True,
                 rerank_top_n: Optional[int] = None,
                 rerank_fusion_weight: float = 0.0,
                 telemetry=None):
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
        self.evidence_judge = evidence_judge
        self.candidate_k = candidate_k
        self.auto_merge_parents = auto_merge_parents
        self.step_back = step_back
        self.context_ordering = context_ordering
        self.prune_unreferenced_citations = prune_unreferenced_citations
        self.rerank_top_n = rerank_top_n
        self.rerank_fusion_weight = rerank_fusion_weight
        self._telemetry = telemetry

    def _get_telemetry(self):
        return self._telemetry if self._telemetry is not None else get_telemetry()

    def _extract_parent_key(self, citation) -> Optional[Any]:
        """Extract a unique parent identifier tuple (source, parent_id/index/hash)."""
        meta = citation.metadata or {}
        parent_text = meta.get("parent_text")
        if not parent_text or not isinstance(parent_text, str) or not parent_text.strip():
            return None

        # 1. Explicit parent_id takes precedence
        if "parent_id" in meta and meta["parent_id"] is not None:
            return (citation.source, str(meta["parent_id"]))

        # 2. Document source + parent_index
        if "parent_index" in meta and meta["parent_index"] is not None:
            return (citation.source, meta["parent_index"])

        # 3. Fallback to source + parent_text hash
        return (citation.source, hash(parent_text.strip()))
    def _expand_to_parent_context(self, citations: List) -> List:
        """
        Expand child chunks to parent context with auto-merging deduplication.

        Groups sibling child citations by parent, eliminates duplicate parent text,
        merges child scores (max score) and metadata into a single unified Citation.
        """
        from ..types import Citation

        if not citations:
            return []

        # Legacy fallback if auto_merge_parents is disabled
        if not getattr(self, "auto_merge_parents", True):
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

        parent_groups = {}
        item_order = []

        for citation in citations:
            parent_key = self._extract_parent_key(citation)
            if parent_key is None:
                item_order.append(("single", citation))
            else:
                if parent_key not in parent_groups:
                    parent_groups[parent_key] = []
                    item_order.append(("parent", parent_key))
                parent_groups[parent_key].append(citation)

        merged = []
        for item_type, item in item_order:
            if item_type == "single":
                merged.append(item)
            else:
                children = parent_groups[item]
                best_child = max(children, key=lambda c: c.score)
                parent_text = best_child.metadata["parent_text"]

                # Start from best child's metadata, removing parent_text
                merged_meta = dict(best_child.metadata)
                merged_meta.pop("parent_text", None)

                # Consolidate attributes from other sibling children
                for c in children:
                    for k, v in c.metadata.items():
                        if k != "parent_text" and k not in merged_meta:
                            merged_meta[k] = v

                merged_meta["chunk_type"] = "parent"
                merged_meta["merged_children_count"] = len(children)

                # Extract child chunk indices and scores
                child_indices = [
                    c.metadata.get("chunk_index")
                    for c in children
                    if c.metadata.get("chunk_index") is not None
                ]
                if child_indices:
                    merged_meta["child_indices"] = sorted(child_indices)

                merged_meta["child_scores"] = [c.score for c in children]

                # Composite score rewarding multi-child convergence
                max_score = best_child.score
                if len(children) > 1:
                    merged_meta["composite_score"] = min(1.0, max_score + 0.05 * (min(len(children), 4) - 1))
                else:
                    merged_meta["composite_score"] = max_score

                merged.append(
                    Citation(
                        source=best_child.source,
                        chunk=parent_text,
                        score=max_score,
                        metadata=merged_meta,
                    )
                )

        merged.sort(key=lambda c: c.score, reverse=True)
        return merged
    
    def _search_single_query(self, query_var, search_top_k, filters):
        """Search for a single query variation. Thread-safe with span propagation."""
        tel = self._get_telemetry()
        store_prov = classify_store_provider(self.store)
        with tel.span("embed.query", operation="embed", provider="local"):
            query_embedding = self.embedder.embed_query(query_var)
        with tel.span(
            "store.vector_search",
            operation="vector_search",
            provider=store_prov,
            topK=search_top_k,
            hasFilters=bool(filters),
        ):
            return self.store.search(
                query_embedding=query_embedding,
                top_k=search_top_k,
                filters=filters,
            )

    def _search_and_fuse_variations(self, query_variations: List[str], search_top_k: int, filters: Optional[Dict[str, Any]] = None, rrf_k: int = 60) -> List:
        """Execute parallel search across query variations and fuse results deterministically using RRF."""
        if not query_variations:
            return []

        tel = self._get_telemetry()
        store_prov = classify_store_provider(self.store)
        with tel.span(
            "search.variations",
            provider=store_prov,
            variationCount=len(query_variations),
            topK=search_top_k,
        ):
            if len(query_variations) == 1:
                return self._search_single_query(query_variations[0], search_top_k, filters)

            workers = min(len(query_variations), max(1, self.max_parallel_queries))
            results_by_query = {}
            worker_fn = wrap_with_context(self._search_single_query)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(worker_fn, qv, search_top_k, filters): qv
                    for qv in query_variations
                }
                for future in as_completed(futures):
                    qv = futures[future]
                    try:
                        results_by_query[qv] = future.result()
                    except Exception as e:
                        logger.warning("Query variation search failed for '{}': {}".format(qv, e))
                        tel.log(
                            "warn",
                            "variation_search_failed",
                            operation="vector_search",
                            provider=store_prov,
                            errorType=type(e).__name__,
                        )
                        results_by_query[qv] = []

            # Reciprocal Rank Fusion across query variations
            rrf_scores = {}
            citation_map = {}
            for qv in query_variations:
                citations = results_by_query.get(qv, [])
                for rank, citation in enumerate(citations, 1):
                    key = (citation.source, citation.chunk)
                    if key not in citation_map:
                        citation_map[key] = citation
                        rrf_scores[key] = 0.0
                    rrf_scores[key] += 1.0 / (rrf_k + rank)

            # Sort descending by fused score
            sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
            return [citation_map[k].model_copy(update={"score": rrf_scores[k]}) for k in sorted_keys]

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
        1. Transform query (HyDE / step-back / multi-query expansion)
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

        tel = self._get_telemetry()
        store_prov = classify_store_provider(self.store)

        with tel.span(
            "ragi.retrieve",
            operation="retrieve",
            provider=store_prov,
            topK=top_k,
            hasFilters=bool(filters),
        ) as span_attrs:
            if self.mode in ("bm25_only", "on_the_fly", "hot_cold"):
                non_dense = self._non_dense_search(query, top_k, filters)
                span_attrs["citationCount"] = len(non_dense)
                tel.record_retrieval_quality(len(non_dense))
                return non_dense

            # Determine queries to use for retrieval
            query_variations = [query]
            if self.step_back or self.hyde:
                with tel.span("query.transform", operation="query_transform", provider="local"):
                    if self.step_back:
                        try:
                            sb_queries = self.step_back.get_queries(query)
                            query_variations = sb_queries
                        except Exception as e:
                            logger.warning("Step-back prompting failed: {}, using original query".format(e))
                            query_variations = [query]

                    if self.hyde:
                        try:
                            hypothetical_doc = self.hyde.transform_query(query)
                            if len(query_variations) > 1:
                                query_variations = [hypothetical_doc, query_variations[1]]
                            else:
                                query_variations = [hypothetical_doc]
                            logger.debug("HyDE generated: {}...".format(hypothetical_doc[:100]))
                        except Exception as e:
                            logger.warning("HyDE failed: {}, falling back to regular query".format(e))

            # Get more candidates if we're using cross-encoder reranking
            if self.cross_encoder:
                search_top_k = max(top_k * 4, self.rerank_top_n) if self.rerank_top_n else top_k * 4
            else:
                search_top_k = top_k

            all_citations = self._search_and_fuse_variations(query_variations, search_top_k, filters)

            # Apply hybrid search if enabled
            if self.hybrid_searcher:
                try:
                    with tel.span("search.hybrid", operation="hybrid_fusion", provider="bm25", topK=search_top_k):
                        all_citations = self._hybrid_search(query, all_citations, search_top_k, filters)
                except Exception as e:
                    logger.warning("Hybrid search failed: {}".format(e))

            # Apply cross-encoder reranking if enabled
            if self.cross_encoder:
                try:
                    try:
                        all_citations = self.cross_encoder.rerank(
                            query=query,
                            citations=all_citations,
                            top_k=top_k,
                            fusion_weight=self.rerank_fusion_weight if self.rerank_fusion_weight > 0 else None,
                        )
                    except TypeError:
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

            span_attrs["citationCount"] = len(all_citations)
            tel.record_retrieval_quality(len(all_citations))
            if not all_citations:
                tel.log(
                    "warn",
                    "empty_retrieval",
                    operation="retrieve",
                    provider=store_prov,
                    topK=top_k,
                    hasFilters=bool(filters),
                )
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

    def _hybrid_search(self, query, citations, top_k, filters):
        # Full citations let keyword retrieval recover evidence outside the vector shortlist.
        # Stores without this optional capability retain the existing hybrid behavior.
        lexical_query = self._rewrite_if_enabled(query)
        if callable(getattr(type(self.store), "get_all_citations", None)):
            corpus = self.store.get_all_citations(filters=filters)
            return self.hybrid_searcher.search_corpus(lexical_query, citations, corpus, top_k)
        return self.hybrid_searcher.search(query=lexical_query, vector_citations=citations, top_k=top_k)

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

        if top_k <= 0:
            return Answer(text="Please request at least one passage.", citations=[], query=query)

        tel = self._get_telemetry()
        store_prov = classify_store_provider(self.store)

        with tel.span(
            "ragi.ask",
            operation="ask",
            provider=store_prov,
            topK=top_k,
            hasFilters=bool(filters),
        ) as span_attrs:
            if self.mode in ("bm25_only", "on_the_fly", "hot_cold"):
                citations = self._non_dense_search(query, top_k, filters)
                answer = self._generate_answer_from_citations(query, citations, system_prompt)
                span_attrs["citationCount"] = len(answer.citations)
                tel.record_retrieval_quality(len(answer.citations))
                return answer

            stepback_query = None
            with tel.span("query.transform", operation="query_transform", provider="local"):
                if self.step_back:
                    try:
                        sb = self.step_back.generate_stepback_query(query)
                        if sb and sb.strip() and sb.strip() != query.strip():
                            stepback_query = sb.strip()
                    except Exception as e:
                        logger.warning("Step-back generation failed: {}".format(e))

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

                if stepback_query and stepback_query not in query_variations:
                    query_variations.append(stepback_query)

            # Get more candidates if we're using cross-encoder reranking
            context_k = max(top_k, self.candidate_k) if self.evidence_judge else top_k
            if self.cross_encoder:
                search_top_k = max(context_k * 4, self.rerank_top_n) if self.rerank_top_n else context_k * 4
            else:
                search_top_k = context_k

            all_citations = self._search_and_fuse_variations(query_variations, search_top_k, filters)

            # Apply hybrid search if enabled
            if self.hybrid_searcher:
                try:
                    with tel.span("search.hybrid", operation="hybrid_fusion", provider="bm25", topK=search_top_k):
                        all_citations = self._hybrid_search(query, all_citations, search_top_k, filters)
                except Exception as e:
                    logger.warning("Hybrid search failed: {}".format(e))
                    # Continue with vector-only results

            # Apply cross-encoder reranking if enabled
            if self.cross_encoder:
                try:
                    try:
                        all_citations = self.cross_encoder.rerank(
                            query=query,  # Use original query for reranking
                            citations=all_citations,
                            top_k=context_k,
                            fusion_weight=self.rerank_fusion_weight if self.rerank_fusion_weight > 0 else None,
                        )
                    except TypeError:
                        all_citations = self.cross_encoder.rerank(
                            query=query,
                            citations=all_citations,
                            top_k=context_k,
                        )
                except Exception as e:
                    logger.warning("Cross-encoder reranking failed: {}".format(e))
                    # Fall back to score-based sorting
                    all_citations.sort(key=lambda c: c.score, reverse=True)
                    all_citations = all_citations[:context_k]
            else:
                # Sort by score and take top_k
                all_citations.sort(key=lambda c: c.score, reverse=True)
                all_citations = all_citations[:context_k]

            final_citations = all_citations

            # For hierarchical chunks, expand to parent context
            if self.use_hierarchical:
                final_citations = self._expand_to_parent_context(final_citations)

            # Add graph context if enabled
            graph_context = ""
            if self.graph:
                graph_context = self.graph.to_context(query, max_triples=10)

            decision_report = None
            if self.evidence_judge is not None:
                # Graph evidence must pass the same judge, rather than bypass it in a prompt.
                if graph_context:
                    from ..types import Citation
                    final_citations.append(Citation(source="knowledge_graph", chunk=graph_context, score=0))
                    graph_context = ""
                selection = self.evidence_judge.select(query, final_citations, top_k)
                decision_report = selection.report
                if selection.abstained:
                    from ..evidence import NO_EVIDENCE
                    span_attrs["abstained"] = True
                    span_attrs["citationCount"] = 0
                    tel.record_retrieval_quality(0)
                    tel.log(
                        "warn",
                        "empty_retrieval",
                        operation="ask",
                        provider=store_prov,
                        abstained=True,
                    )
                    return Answer(text=NO_EVIDENCE, citations=[], query=query, decision=decision_report)
                final_citations = selection.citations

            # Build system prompt with graph context
            final_system_prompt = system_prompt
            if graph_context:
                if final_system_prompt:
                    final_system_prompt = "{}\n\n{}".format(final_system_prompt, graph_context)
                else:
                    final_system_prompt = graph_context

            # Generate answer with context ordering applied post-reranking in retriever
            should_reorder = True if self.context_ordering == "u_shaped" else False
            try:
                answer = self.retriever.generate_answer(
                    query=query,
                    citations=final_citations,
                    system_prompt=final_system_prompt,
                    reorder=should_reorder,
                )
            except TypeError:
                answer = self.retriever.generate_answer(
                    query=query,
                    citations=final_citations,
                    system_prompt=final_system_prompt,
                )

            if decision_report is not None:
                answer.decision = decision_report

            span_attrs["citationCount"] = len(answer.citations)
            tel.record_retrieval_quality(len(answer.citations))
            if not answer.citations:
                tel.log(
                    "warn",
                    "empty_retrieval",
                    operation="ask",
                    provider=store_prov,
                    topK=top_k,
                    hasFilters=bool(filters),
                )
            return answer
