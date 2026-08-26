import logging
from typing import Optional, Callable, List, Any

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """Composable document ingestion pipeline: Load -> Chunk -> Embed -> Store."""
    
    def __init__(self, loader, chunker, embedder, store,
                 graph=None, hybrid_searcher=None,
                 post_load_hook=None, post_chunk_hook=None, post_embed_hook=None,
                 use_hierarchical=False, mode="dense",
                 bm25_index=None, on_the_fly_index=None):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.graph = graph
        self.hybrid_searcher = hybrid_searcher
        self.post_load_hook = post_load_hook
        self.post_chunk_hook = post_chunk_hook
        self.post_embed_hook = post_embed_hook
        self.use_hierarchical = use_hierarchical
        self.mode = mode
        self.bm25_index = bm25_index
        self.on_the_fly_index = on_the_fly_index
    
    def ingest(self, sources, on_progress=None, llm_client=None):
        """Run the full ingestion pipeline.
        
        Args:
            sources: File paths, directories, URLs to ingest
            on_progress: Optional progress callback
            llm_client: Optional LLMClient for graph extraction
            
        Returns:
            Number of chunks ingested
        """
        _progress = on_progress or (lambda msg: logger.info(msg))
        
        # 1. Stream and chunk
        all_chunks = []
        doc_count = 0
        for doc in self.loader.stream(sources):
            if self.post_load_hook:
                docs = self.post_load_hook([doc])
                if not docs:
                    continue
                doc = docs[0]
            
            doc_count += 1
            _progress("Chunking document {}: {}".format(doc_count, doc.source))
            
            if self.use_hierarchical:
                parents, children = self.chunker.chunk_document(doc)
                all_chunks.extend(children)
            else:
                chunks = self.chunker.chunk_document(doc)
                all_chunks.extend(chunks)
        
        _progress("Processed {} documents, created {} chunks".format(doc_count, len(all_chunks)))
        
        # 2. Post-chunk hook
        if self.post_chunk_hook:
            all_chunks = self.post_chunk_hook(all_chunks)

        # 3-5. Embed + store, branching on retrieval mode
        if self.mode == "bm25_only":
            from ..types import Citation
            _progress("Indexing {} chunks for BM25 (no embeddings)...".format(len(all_chunks)))
            self.bm25_index.index_chunks([
                Citation(source=c.source, chunk=c.text, score=0.0, metadata=c.metadata)
                for c in all_chunks
            ])
            chunks_with_embeddings = all_chunks
        elif self.mode == "on_the_fly":
            _progress("Adding {} chunks for on-the-fly embedding...".format(len(all_chunks)))
            self.on_the_fly_index.add_chunks(all_chunks)
            chunks_with_embeddings = all_chunks
        elif self.mode == "hot_cold":
            hot_chunks = [c for c in all_chunks if c.metadata.get("tier") == "hot"]
            cold_chunks = [c for c in all_chunks if c.metadata.get("tier") != "hot"]

            _progress("Pre-embedding {} hot chunks...".format(len(hot_chunks)))
            hot_embedded = self.embedder.embed_chunks(hot_chunks, on_progress=_progress) if hot_chunks else []
            if self.post_embed_hook and hot_embedded:
                hot_embedded = self.post_embed_hook(hot_embedded)
            if hot_embedded:
                (self.store.add_chunks if hasattr(self.store, "add_chunks") else self.store.add)(hot_embedded)

            _progress("Adding {} cold chunks for on-the-fly embedding...".format(len(cold_chunks)))
            if cold_chunks:
                self.on_the_fly_index.add_chunks(cold_chunks)

            chunks_with_embeddings = hot_embedded + cold_chunks
        else:
            # Dense (default): embed everything up front and store in the vector store
            _progress("Generating embeddings for {} chunks...".format(len(all_chunks)))
            chunks_with_embeddings = self.embedder.embed_chunks(all_chunks, on_progress=_progress)

            if self.post_embed_hook:
                chunks_with_embeddings = self.post_embed_hook(chunks_with_embeddings)

            _progress("Storing {} chunks...".format(len(chunks_with_embeddings)))
            # Store might use add() or add_chunks()
            if hasattr(self.store, "add_chunks"):
                self.store.add_chunks(chunks_with_embeddings)
            else:
                self.store.add(chunks_with_embeddings)

        # 6. Graph extraction
        if self.graph and llm_client:
            _progress("Extracting knowledge graph...")
            for chunk in chunks_with_embeddings:
                self.graph.extract_and_add(
                    text=chunk.text,
                    llm_client=llm_client.client if hasattr(llm_client, "client") else llm_client,
                    model=getattr(llm_client, "model", "default")
                )
            self.graph.save()

        # 7. Hybrid search indexing (dense mode only - other modes build their own index above)
        if self.hybrid_searcher and self.mode == "dense":
            chunk_texts = self.store.get_all_chunk_texts()
            self.hybrid_searcher.index_chunks(chunk_texts)

        _progress("Done")
        return len(chunks_with_embeddings)
