"""Type definitions for Ragi."""

from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A single citation with source information."""

    source: str = Field(description="Source file path or URL")
    chunk: str = Field(description="The actual text chunk")
    score: float = Field(description="Relevance score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @property
    def preview(self) -> str:
        """Return a preview of the chunk (first 100 chars)."""
        return self.chunk[:100] + "..." if len(self.chunk) > 100 else self.chunk


class Answer(BaseModel):
    """Answer with citations from the RAG system."""

    text: str = Field(description="The generated answer")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    query: str = Field(description="Original query")

    def __str__(self) -> str:
        """Return the answer text."""
        return self.text

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"Answer(text='{self.text[:50]}...', citations={len(self.citations)})"


class Document(BaseModel):
    """Internal document representation."""

    content: str = Field(description="Document content in markdown")
    source: str = Field(description="Source file path or URL")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    content_hash: Optional[str] = Field(default=None, description="Hash of content for change detection")
    last_modified: Optional[float] = Field(default=None, description="Last modification timestamp")


class Chunk(BaseModel):
    """A chunk of a document with metadata."""

    text: str = Field(description="Chunk text")
    source: str = Field(description="Source document")
    chunk_index: int = Field(description="Index of chunk in document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding")


# Type aliases for hooks
DocumentHook = Callable[[List["Document"]], List["Document"]]
ChunkHook = Callable[[List["Chunk"]], List["Chunk"]]
