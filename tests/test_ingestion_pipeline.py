import pytest
from unittest.mock import Mock, call, ANY
from piragi.pipelines.ingestion import IngestionPipeline
from piragi import Ragi

def test_basic_ingestion():
    loader = Mock()
    chunker = Mock()
    embedder = Mock()
    store = Mock()
    
    doc = Mock()
    doc.source = "test.txt"
    loader.stream.return_value = [doc]
    
    chunk1 = Mock()
    chunk2 = Mock()
    chunker.chunk_document.return_value = [chunk1, chunk2]
    
    embedder.embed_chunks.return_value = ["embedded1", "embedded2"]
    
    pipeline = IngestionPipeline(loader, chunker, embedder, store)
    
    count = pipeline.ingest(["test.txt"])
    
    assert count == 2
    loader.stream.assert_called_once_with(["test.txt"])
    chunker.chunk_document.assert_called_once_with(doc)
    embedder.embed_chunks.assert_called_once_with([chunk1, chunk2], on_progress=ANY)
    store.add_chunks.assert_called_once_with(["embedded1", "embedded2"])

def test_ingestion_with_hooks():
    loader = Mock()
    chunker = Mock()
    embedder = Mock()
    store = Mock()
    
    doc = Mock()
    doc.source = "test.txt"
    loader.stream.return_value = [doc]
    
    post_load_doc = Mock()
    post_load_doc.source = "test2.txt"
    post_load_hook = Mock(return_value=[post_load_doc])
    
    chunk = Mock()
    chunker.chunk_document.return_value = [chunk]
    
    post_chunk_hook = Mock(return_value=["hook_chunk"])
    
    embedder.embed_chunks.return_value = ["embedded"]
    
    post_embed_hook = Mock(return_value=["hook_embedded"])
    
    pipeline = IngestionPipeline(
        loader, chunker, embedder, store,
        post_load_hook=post_load_hook,
        post_chunk_hook=post_chunk_hook,
        post_embed_hook=post_embed_hook
    )
    
    pipeline.ingest(["test.txt"])
    
    post_load_hook.assert_called_once_with([doc])
    chunker.chunk_document.assert_called_once_with(post_load_doc)
    post_chunk_hook.assert_called_once_with([chunk])
    post_embed_hook.assert_called_once_with(["embedded"])
    store.add_chunks.assert_called_once_with(["hook_embedded"])

def test_ingestion_with_graph():
    loader = Mock()
    chunker = Mock()
    embedder = Mock()
    store = Mock()
    graph = Mock()
    
    doc = Mock()
    doc.source = "test.txt"
    loader.stream.return_value = [doc]
    
    chunk = Mock()
    chunker.chunk_document.return_value = [chunk]
    
    embedded = Mock()
    embedded.text = "test chunk"
    embedder.embed_chunks.return_value = [embedded]
    
    pipeline = IngestionPipeline(loader, chunker, embedder, store, graph=graph)
    
    llm_client = Mock()
    llm_client.client = "fake_client"
    llm_client.model = "fake_model"
    
    pipeline.ingest(["test.txt"], llm_client=llm_client)
    
    graph.extract_and_add.assert_called_once_with(
        text="test chunk",
        llm_client="fake_client",
        model="fake_model"
    )
    graph.save.assert_called_once()

def test_ingestion_with_hybrid():
    loader = Mock()
    chunker = Mock()
    embedder = Mock()
    store = Mock()
    hybrid_searcher = Mock()
    
    doc = Mock()
    doc.source = "test.txt"
    loader.stream.return_value = [doc]
    chunker.chunk_document.return_value = [Mock()]
    embedder.embed_chunks.return_value = [Mock()]
    store.get_all_chunk_texts.return_value = ["text1", "text2"]
    
    pipeline = IngestionPipeline(loader, chunker, embedder, store, hybrid_searcher=hybrid_searcher)
    pipeline.ingest(["test.txt"])
    
    hybrid_searcher.index_chunks.assert_called_once_with(["text1", "text2"])

def test_ingestion_returns_count():
    loader = Mock()
    chunker = Mock()
    embedder = Mock()
    store = Mock()
    
    doc = Mock()
    doc.source = "test.txt"
    loader.stream.return_value = [doc]
    
    chunker.chunk_document.return_value = [Mock(), Mock(), Mock()]
    embedder.embed_chunks.return_value = [Mock(), Mock(), Mock()]
    
    pipeline = IngestionPipeline(loader, chunker, embedder, store)
    count = pipeline.ingest(["test.txt"])
    
    assert count == 3

def test_ragi_add_uses_pipeline(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document.")

    kb = Ragi(persist_dir=str(tmp_path / ".piragi"))

    # Store initial count
    initial_count = kb.count()

    kb.add(str(test_file))

    # Verify count increased
    assert kb.count() > initial_count

    # Verify we can query
    ans = kb.ask("test document")
    assert ans is not None


def _text_chunk(source="test.txt", text="chunk text"):
    """A chunk-like Mock with real str/dict attrs (Citation validates types)."""
    chunk = Mock()
    chunk.source = source
    chunk.text = text
    chunk.metadata = {}
    return chunk


def test_ingestion_bm25_only_mode_skips_embedding():
    loader = Mock()
    chunker = Mock()
    embedder = Mock()
    store = Mock()
    bm25_index = Mock()

    doc = Mock()
    doc.source = "test.txt"
    loader.stream.return_value = [doc]

    chunk = _text_chunk()
    chunker.chunk_document.return_value = [chunk]

    pipeline = IngestionPipeline(
        loader, chunker, embedder, store, mode="bm25_only", bm25_index=bm25_index
    )
    count = pipeline.ingest(["test.txt"])

    embedder.embed_chunks.assert_not_called()
    store.add_chunks.assert_not_called()
    bm25_index.index_chunks.assert_called_once()
    indexed = bm25_index.index_chunks.call_args[0][0]
    assert indexed[0].source == "test.txt"
    assert indexed[0].chunk == "chunk text"
    assert count == 1


def test_ingestion_on_the_fly_mode_skips_embedding():
    loader = Mock()
    chunker = Mock()
    embedder = Mock()
    store = Mock()
    on_the_fly_index = Mock()

    doc = Mock()
    doc.source = "test.txt"
    loader.stream.return_value = [doc]

    chunk = _text_chunk()
    chunker.chunk_document.return_value = [chunk]

    pipeline = IngestionPipeline(
        loader, chunker, embedder, store,
        mode="on_the_fly", on_the_fly_index=on_the_fly_index,
    )
    count = pipeline.ingest(["test.txt"])

    embedder.embed_chunks.assert_not_called()
    store.add_chunks.assert_not_called()
    on_the_fly_index.add_chunks.assert_called_once_with([chunk])
    assert count == 1


def test_ingestion_hot_cold_mode_routes_by_tier():
    loader = Mock()
    chunker = Mock()
    embedder = Mock()
    store = Mock()
    on_the_fly_index = Mock()

    doc = Mock()
    doc.source = "test.txt"
    loader.stream.return_value = [doc]

    hot_chunk = _text_chunk(text="hot")
    hot_chunk.metadata = {"tier": "hot"}
    cold_chunk = _text_chunk(text="cold")
    cold_chunk.metadata = {"tier": "cold"}
    chunker.chunk_document.return_value = [hot_chunk, cold_chunk]

    embedder.embed_chunks.return_value = ["hot_embedded"]

    pipeline = IngestionPipeline(
        loader, chunker, embedder, store,
        mode="hot_cold", on_the_fly_index=on_the_fly_index,
    )
    count = pipeline.ingest(["test.txt"])

    embedder.embed_chunks.assert_called_once_with([hot_chunk], on_progress=ANY)
    store.add_chunks.assert_called_once_with(["hot_embedded"])
    on_the_fly_index.add_chunks.assert_called_once_with([cold_chunk])
    assert count == 2  # 1 hot embedded + 1 cold


def test_ingestion_hot_cold_mode_skips_store_when_no_hot_chunks():
    loader = Mock()
    chunker = Mock()
    embedder = Mock()
    store = Mock()
    on_the_fly_index = Mock()

    doc = Mock()
    doc.source = "test.txt"
    loader.stream.return_value = [doc]

    cold_chunk = _text_chunk(text="cold")
    cold_chunk.metadata = {"tier": "cold"}
    chunker.chunk_document.return_value = [cold_chunk]

    pipeline = IngestionPipeline(
        loader, chunker, embedder, store,
        mode="hot_cold", on_the_fly_index=on_the_fly_index,
    )
    pipeline.ingest(["test.txt"])

    embedder.embed_chunks.assert_not_called()
    store.add_chunks.assert_not_called()
    on_the_fly_index.add_chunks.assert_called_once_with([cold_chunk])


def test_ingestion_hybrid_indexing_skipped_for_non_dense_modes():
    """The dense-only hybrid_searcher.index_chunks() re-indexing step (step 7) must
    not fire for bm25_only/on_the_fly/hot_cold - they build their own index instead."""
    loader = Mock()
    chunker = Mock()
    embedder = Mock()
    store = Mock()
    bm25_index = Mock()
    hybrid_searcher = Mock()

    doc = Mock()
    doc.source = "test.txt"
    loader.stream.return_value = [doc]
    chunker.chunk_document.return_value = [_text_chunk()]

    pipeline = IngestionPipeline(
        loader, chunker, embedder, store,
        mode="bm25_only", bm25_index=bm25_index, hybrid_searcher=hybrid_searcher,
    )
    pipeline.ingest(["test.txt"])

    hybrid_searcher.index_chunks.assert_not_called()
