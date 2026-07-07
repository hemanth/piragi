"""Tests for AsyncRagi wrapper."""

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def mock_ragi():
    """Mock the sync Ragi class."""
    with patch("piragi.async_ragi.Ragi") as mock:
        mock_instance = MagicMock()

        def mock_add(sources, on_progress=None):
            if on_progress:
                on_progress("Discovering files...")
                on_progress("Found 2 documents")
                on_progress("Chunking 1/2: doc1.md")
                on_progress("Chunking 2/2: doc2.md")
                on_progress("Generating embeddings for 64 chunks...")
                on_progress("Embedded 32/64 chunks")
                on_progress("Embedded 64/64 chunks")
                on_progress("Embeddings complete")
                on_progress("Done")
            return mock_instance

        mock_instance.add.side_effect = mock_add
        mock_instance.ask.return_value = MagicMock(text="Test answer", citations=[])
        mock_instance.retrieve.return_value = []
        mock_instance.refresh.return_value = mock_instance
        mock_instance.filter.return_value = mock_instance
        mock_instance.count.return_value = 42
        mock_instance.clear.return_value = None
        mock_instance.graph = MagicMock()
        mock.return_value = mock_instance
        yield mock, mock_instance


@pytest.mark.asyncio
async def test_async_ragi_init(mock_ragi):
    """Test AsyncRagi initialization."""
    mock_class, mock_instance = mock_ragi

    from piragi.async_ragi import AsyncRagi

    kb = AsyncRagi("./docs", persist_dir=".test", graph=True)

    mock_class.assert_called_once_with(
        sources="./docs",
        persist_dir=".test",
        config=None,
        store=None,
        graph=True,
        embedder=None,
    )


@pytest.mark.asyncio
async def test_async_ragi_init_with_embedder(mock_ragi):
    """Test AsyncRagi initialization with custom embedder."""
    mock_class, mock_instance = mock_ragi

    from piragi.async_ragi import AsyncRagi

    mock_embedder = MagicMock()
    mock_embedder.model_name = "custom-model"

    kb = AsyncRagi("./docs", embedder=mock_embedder)

    mock_class.assert_called_once_with(
        sources="./docs",
        persist_dir=".piragi",
        config=None,
        store=None,
        graph=False,
        embedder=mock_embedder,
    )


@pytest.mark.asyncio
async def test_async_ragi_shared_embedder(mock_ragi):
    """Test that embedder can be shared across multiple AsyncRagi instances."""
    mock_class, mock_instance = mock_ragi

    from piragi.async_ragi import AsyncRagi

    mock_embedder = MagicMock()

    kb1 = AsyncRagi("./docs1", embedder=mock_embedder)
    kb2 = AsyncRagi("./docs2", embedder=mock_embedder)

    # Both should have passed the same embedder
    calls = mock_class.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs.get("embedder") is mock_embedder
    assert calls[1].kwargs.get("embedder") is mock_embedder


@pytest.mark.asyncio
async def test_async_ragi_add(mock_ragi):
    """Test async add method."""
    mock_class, mock_instance = mock_ragi

    from piragi.async_ragi import AsyncRagi

    kb = AsyncRagi()
    result = await kb.add("./docs")

    mock_instance.add.assert_called_once_with("./docs")
    assert result is kb  # Returns self for chaining


@pytest.mark.asyncio
async def test_async_ragi_ask(mock_ragi):
    """Test async ask method."""
    mock_class, mock_instance = mock_ragi

    from piragi.async_ragi import AsyncRagi

    kb = AsyncRagi()
    answer = await kb.ask("What is X?", top_k=3)

    mock_instance.ask.assert_called_once_with("What is X?", 3, None)
    assert answer.text == "Test answer"


@pytest.mark.asyncio
async def test_async_ragi_ask_with_system_prompt(mock_ragi):
    """Test async ask with custom system prompt."""
    mock_class, mock_instance = mock_ragi

    from piragi.async_ragi import AsyncRagi

    kb = AsyncRagi()
    await kb.ask("What is X?", system_prompt="Be concise")

    mock_instance.ask.assert_called_once_with("What is X?", 5, "Be concise")


@pytest.mark.asyncio
async def test_async_ragi_retrieve(mock_ragi):
    """Test async retrieve method."""
    mock_class, mock_instance = mock_ragi
    mock_instance.retrieve.return_value = [MagicMock(chunk="test chunk")]

    from piragi.async_ragi import AsyncRagi

    kb = AsyncRagi()
    chunks = await kb.retrieve("query", top_k=10)

    mock_instance.retrieve.assert_called_once_with("query", 10)
    assert len(chunks) == 1


@pytest.mark.asyncio
async def test_async_ragi_refresh(mock_ragi):
    """Test async refresh method."""
    mock_class, mock_instance = mock_ragi

    from piragi.async_ragi import AsyncRagi

    kb = AsyncRagi()
    result = await kb.refresh("./docs")

    mock_instance.refresh.assert_called_once_with("./docs")
    assert result is kb


@pytest.mark.asyncio
async def test_async_ragi_filter(mock_ragi):
    """Test filter method (sync, returns self)."""
    mock_class, mock_instance = mock_ragi

    from piragi.async_ragi import AsyncRagi

    kb = AsyncRagi()
    result = kb.filter(type="docs", source="api.md")

    mock_instance.filter.assert_called_once_with(type="docs", source="api.md")
    assert result is kb


@pytest.mark.asyncio
async def test_async_ragi_count(mock_ragi):
    """Test async count method."""
    mock_class, mock_instance = mock_ragi

    from piragi.async_ragi import AsyncRagi

    kb = AsyncRagi()
    count = await kb.count()

    mock_instance.count.assert_called_once()
    assert count == 42


@pytest.mark.asyncio
async def test_async_ragi_clear(mock_ragi):
    """Test async clear method."""
    mock_class, mock_instance = mock_ragi

    from piragi.async_ragi import AsyncRagi

    kb = AsyncRagi()
    await kb.clear()

    mock_instance.clear.assert_called_once()


@pytest.mark.asyncio
async def test_async_ragi_graph_property(mock_ragi):
    """Test graph property access."""
    mock_class, mock_instance = mock_ragi
    mock_instance.graph = MagicMock(triples=lambda: [("a", "b", "c")])

    from piragi.async_ragi import AsyncRagi

    kb = AsyncRagi()
    graph = kb.graph

    assert graph is mock_instance.graph


@pytest.mark.asyncio
async def test_async_ragi_callable(mock_ragi):
    """Test callable shorthand."""
    mock_class, mock_instance = mock_ragi

    from piragi.async_ragi import AsyncRagi

    kb = AsyncRagi()
    answer = await kb("What is X?", top_k=3)

    mock_instance.ask.assert_called_once_with("What is X?", 3, None)


@pytest.mark.asyncio
async def test_async_ragi_concurrent_calls(mock_ragi):
    """Test that multiple async calls can run concurrently."""
    mock_class, mock_instance = mock_ragi

    call_count = 0

    def slow_ask(*args):
        nonlocal call_count
        call_count += 1
        return MagicMock(text=f"Answer {call_count}", citations=[])

    mock_instance.ask.side_effect = slow_ask

    from piragi.async_ragi import AsyncRagi

    kb = AsyncRagi()

    # Run multiple asks concurrently
    results = await asyncio.gather(
        kb.ask("Q1"),
        kb.ask("Q2"),
        kb.ask("Q3"),
    )

    assert len(results) == 3
    assert mock_instance.ask.call_count == 3


@pytest.mark.asyncio
async def test_async_ragi_add_without_progress(mock_ragi):
    """Test add() without progress - simple await."""
    mock_class, mock_instance = mock_ragi

    from piragi.async_ragi import AsyncRagi

    kb = AsyncRagi()
    result = await kb.add("./docs")

    mock_instance.add.assert_called_once()
    assert result is kb


@pytest.mark.asyncio
async def test_async_ragi_add_with_progress(mock_ragi):
    """Test add() with progress=True - async iterator."""
    mock_class, mock_instance = mock_ragi

    from piragi.async_ragi import AsyncRagi

    kb = AsyncRagi()
    messages = []

    async for msg in kb.add("./docs", progress=True):
        messages.append(msg)

    assert len(messages) == 9
    assert messages[0] == "Discovering files..."
    assert messages[1] == "Found 2 documents"
    assert "Embedded 32/64 chunks" in messages
    assert "Embedded 64/64 chunks" in messages
    assert messages[-1] == "Done"
    mock_instance.add.assert_called_once()


@pytest.mark.asyncio
async def test_async_ragi_add_progress_iterator_type(mock_ragi):
    """Test that add(progress=True) returns an async iterator."""
    mock_class, mock_instance = mock_ragi

    from piragi.async_ragi import AsyncRagi, _AddIterator

    kb = AsyncRagi()
    result = kb.add("./docs", progress=True)

    assert isinstance(result, _AddIterator)
    assert hasattr(result, "__aiter__")
    assert hasattr(result, "__anext__")
