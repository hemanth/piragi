"""
piragi - Zero-setup RAG library with auto-chunking, embeddings, and smart citations.

Example:
    >>> from piragi import Ragi
    >>>
    >>> # One-liner setup and query
    >>> kb = Ragi("./docs")
    >>> answer = kb.ask("How do I install this?")
    >>>
    >>> # Access answer and citations
    >>> print(answer.text)
    >>> for citation in answer.citations:
    ...     print(f"Source: {citation.source}")
    ...     print(f"Relevance: {citation.score:.2f}")
    >>>
    >>> # Callable shorthand
    >>> answer = kb("What's the API?")
    >>>
    >>> # Filter by metadata
    >>> answer = kb.filter(type="documentation").ask("How to configure?")
"""

from .core import Ragi
from .types import Answer, Citation

__version__ = "0.1.5"
__all__ = ["Ragi", "Answer", "Citation"]
