import logging
from typing import Optional, Callable, List, Any

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """Composable document ingestion pipeline: Load -> Chunk -> Embed -> Store."""
    
    def __init__(self, loader, chunker, embedder, store,
                 graph=None, hybrid_searcher=None,
                 post_load_hook=None, post_chunk_hook=None, post_embed_hook=None,
                 use_hierarchical=False):
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
        
        # 3. Embed
        _progress("Generating embeddings for {} chunks...".format(len(all_chunks)))
        chunks_with_embeddings = self.embedder.embed_chunks(all_chunks, on_progress=_progress)
        
        # 4. Post-embed hook
        if self.post_embed_hook:
            chunks_with_embeddings = self.post_embed_hook(chunks_with_embeddings)
        
        # 5. Store
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
        
        # 7. Hybrid search indexing
        if self.hybrid_searcher:
            if hasattr(self.store, "get_all_chunks"):
                self.hybrid_searcher.index_chunks(self.store.get_all_chunks())
            else:
                self.hybrid_searcher.index_chunks(self.store.get_all_chunk_texts())
        
        _progress("Done")
        return len(chunks_with_embeddings)
