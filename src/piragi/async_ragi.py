"""Async wrapper for Ragi - non-blocking API for web frameworks."""

import asyncio
import queue
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from .core import Ragi
from .embeddings import EmbeddingGenerator
from .stores import VectorStoreProtocol
from .types import Answer


class AsyncRagi:
    """
    Async wrapper around Ragi for use with async web frameworks.

    Provides non-blocking versions of all Ragi methods by running them
    in a thread executor. Use this with FastAPI, Starlette, aiohttp, etc.

    Examples:
        >>> from piragi import AsyncRagi
        >>>
        >>> kb = AsyncRagi("./docs")
        >>> answer = await kb.ask("How do I deploy?")
        >>>
        >>> # With progress tracking
        >>> async for progress in kb.add("./large-docs", progress=True):
        >>>     print(progress)
        >>>
        >>> # With shared embedder
        >>> from piragi import EmbeddingGenerator
        >>> embedder = EmbeddingGenerator(model="custom-model")
        >>> kb1 = AsyncRagi("./docs1", embedder=embedder)
        >>> kb2 = AsyncRagi("./docs2", embedder=embedder)
        >>>
        >>> # With FastAPI
        >>> @app.post("/ingest")
        >>> async def ingest(files: list[str]):
        >>>     await kb.add(files)
        >>>     return {"status": "done"}
    """

    def __init__(
        self,
        sources: Union[str, List[str], None] = None,
        persist_dir: str = ".piragi",
        config: Optional[Dict[str, Any]] = None,
        store: Union[str, Dict[str, Any], VectorStoreProtocol, None] = None,
        graph: bool = False,
        embedder: Optional[EmbeddingGenerator] = None,
    ) -> None:
        """
        Initialize AsyncRagi with optional document sources.

        Args:
            sources: File paths, URLs, or glob patterns to load
            persist_dir: Directory to persist vector database
            config: Configuration dict (see Ragi for options)
            store: Vector store backend
            graph: Enable knowledge graph
            embedder: Optional custom EmbeddingGenerator instance. If provided, this
                embedder will be used instead of creating a new one from config.
                Useful for sharing a single embedder across multiple AsyncRagi instances.
        """
        self._sync = Ragi(
            sources=sources,
            persist_dir=persist_dir,
            config=config,
            store=store,
            graph=graph,
            embedder=embedder,
        )

    def add(
        self,
        sources: Union[str, List[str]],
        progress: bool = False,
    ) -> Union["_AddAwaitable", "_AddIterator"]:
        """
        Add documents to the knowledge base (non-blocking).

        Args:
            sources: File paths, URLs, or glob patterns
            progress: If True, returns an async iterator yielding progress messages

        Returns:
            Awaitable that resolves to self, or async iterator if progress=True

        Examples:
            >>> # Simple - just await
            >>> await kb.add("./docs")
            >>>
            >>> # With progress
            >>> async for msg in kb.add("./docs", progress=True):
            >>>     print(msg)
        """
        if progress:
            return _AddIterator(self, sources)
        return _AddAwaitable(self, sources)

    async def ask(
        self,
        query: str,
        top_k: int = 5,
        system_prompt: Optional[str] = None,
    ) -> Answer:
        """
        Ask a question and get an answer with citations (non-blocking).

        Args:
            query: Question to ask
            top_k: Number of relevant chunks to retrieve
            system_prompt: Optional custom system prompt

        Returns:
            Answer with citations
        """
        return await asyncio.to_thread(
            self._sync.ask, query, top_k, system_prompt
        )

    async def retrieve(self, query: str, top_k: int = 5) -> List:
        """
        Retrieve relevant chunks without LLM generation (non-blocking).

        Args:
            query: Search query
            top_k: Number of chunks to retrieve

        Returns:
            List of Citation objects
        """
        return await asyncio.to_thread(self._sync.retrieve, query, top_k)

    async def refresh(self, sources: Union[str, List[str]]) -> "AsyncRagi":
        """
        Refresh specific sources (non-blocking).

        Args:
            sources: File paths, URLs, or glob patterns to refresh

        Returns:
            Self for chaining
        """
        await asyncio.to_thread(self._sync.refresh, sources)
        return self

    def filter(self, **kwargs: Any) -> "AsyncRagi":
        """
        Filter documents by metadata for the next query.

        Args:
            **kwargs: Metadata key-value pairs to filter by

        Returns:
            Self for chaining
        """
        self._sync.filter(**kwargs)
        return self

    async def count(self) -> int:
        """Return the number of chunks in the knowledge base."""
        return await asyncio.to_thread(self._sync.count)

    async def clear(self) -> None:
        """Clear all data from the knowledge base."""
        await asyncio.to_thread(self._sync.clear)

    @property
    def graph(self):
        """Access the knowledge graph for direct queries."""
        return self._sync.graph

    async def __call__(self, query: str, top_k: int = 5) -> Answer:
        """Callable shorthand for ask()."""
        return await self.ask(query, top_k=top_k)


class _AddAwaitable:
    """Awaitable wrapper for add() without progress."""

    def __init__(self, ragi: AsyncRagi, sources: Union[str, List[str]]) -> None:
        self._ragi = ragi
        self._sources = sources

    def __await__(self):
        return self._run().__await__()

    async def _run(self) -> AsyncRagi:
        await asyncio.to_thread(self._ragi._sync.add, self._sources)
        return self._ragi


class _AddIterator:
    """Async iterator wrapper for add() with progress."""

    def __init__(self, ragi: AsyncRagi, sources: Union[str, List[str]]) -> None:
        self._ragi = ragi
        self._sources = sources
        self._queue: queue.Queue = queue.Queue()
        self._done = False
        self._task: Optional[asyncio.Task] = None

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        # Start the background task on first iteration
        if self._task is None:
            self._task = asyncio.create_task(self._run_in_thread())

        # Poll for progress messages
        while True:
            try:
                msg = self._queue.get_nowait()
                return msg
            except queue.Empty:
                if self._done:
                    raise StopAsyncIteration
                # Wait a bit before polling again
                await asyncio.sleep(0.05)

    async def _run_in_thread(self) -> None:
        def on_progress(msg: str) -> None:
            self._queue.put(msg)

        await asyncio.to_thread(
            self._ragi._sync.add, self._sources, on_progress
        )
        self._done = True
