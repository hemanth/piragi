"""Semantic chunking strategies for improved document splitting."""

import logging
import re
from typing import List, Optional, Tuple

from .types import Chunk, Document

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Semantic chunking that splits documents at natural semantic boundaries.

    Unlike fixed-size chunking, semantic chunking analyzes the content to find
    natural break points where the topic or meaning shifts. This preserves
    coherent units of information.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize semantic chunker.

        Args:
            embedding_model: Model for computing sentence embeddings
            similarity_threshold: Threshold below which to split (0-1)
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
            device: Device for embeddings ('cuda', 'cpu', or None for auto)
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.device = device

        self._model = None

    def _load_model(self):
        """Lazy load embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.embedding_model,
                device=self.device,
            )
        return self._model

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)

        # Also split on double newlines (paragraphs)
        result = []
        for sent in sentences:
            parts = sent.split('\n\n')
            result.extend([p.strip() for p in parts if p.strip()])

        return result

    def _compute_similarities(
        self,
        sentences: List[str],
    ) -> List[float]:
        """
        Compute cosine similarity between adjacent sentences.

        Args:
            sentences: List of sentences

        Returns:
            List of similarities (length = len(sentences) - 1)
        """
        if len(sentences) < 2:
            return []

        model = self._load_model()

        # Embed all sentences
        embeddings = model.encode(sentences, show_progress_bar=False)

        # Compute cosine similarity between adjacent pairs
        import numpy as np

        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            similarities.append(float(sim))

        return similarities

    def _find_split_points(
        self,
        sentences: List[str],
        similarities: List[float],
    ) -> List[int]:
        """
        Find indices where chunks should be split.

        Args:
            sentences: List of sentences
            similarities: Similarities between adjacent sentences

        Returns:
            List of split indices
        """
        split_points = []

        current_chunk_size = 0
        for i, (sent, sim) in enumerate(zip(sentences[:-1], similarities)):
            current_chunk_size += len(sent)

            # Split if:
            # 1. Similarity is below threshold (semantic shift)
            # 2. AND we've accumulated enough content
            # OR
            # 3. Chunk is getting too large
            if (
                sim < self.similarity_threshold
                and current_chunk_size >= self.min_chunk_size
            ) or current_chunk_size >= self.max_chunk_size:
                split_points.append(i + 1)
                current_chunk_size = 0

        return split_points

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Chunk document using semantic analysis.

        Args:
            document: Document to chunk

        Returns:
            List of semantically coherent chunks
        """
        sentences = self._split_sentences(document.content)

        if not sentences:
            return []

        if len(sentences) == 1:
            return [
                Chunk(
                    text=sentences[0],
                    source=document.source,
                    chunk_index=0,
                    metadata=document.metadata.copy(),
                )
            ]

        # Compute similarities
        similarities = self._compute_similarities(sentences)

        # Find split points
        split_points = self._find_split_points(sentences, similarities)

        # Create chunks
        chunks = []
        start_idx = 0

        for chunk_idx, end_idx in enumerate(split_points + [len(sentences)]):
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)

            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        source=document.source,
                        chunk_index=chunk_idx,
                        metadata=document.metadata.copy(),
                    )
                )

            start_idx = end_idx

        return chunks


class ContextualChunker:
    """
    Contextual chunking that prepends document context to each chunk.

    Based on Anthropic's Contextual Retrieval approach, this prepends
    a short context summary to each chunk before embedding, improving
    retrieval accuracy by ~49%.

    Reference: "Introducing Contextual Retrieval" (Anthropic, 2024)
    """

    def __init__(
        self,
        base_chunker: Optional["Chunker"] = None,
        model: str = "llama3.2",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        context_template: Optional[str] = None,
    ) -> None:
        """
        Initialize contextual chunker.

        Args:
            base_chunker: Underlying chunker to use (uses default if None)
            model: LLM model for generating context
            api_key: API key
            base_url: API base URL
            context_template: Template for context prefix
        """
        import os

        from openai import OpenAI

        self.base_chunker = base_chunker
        self.model = model

        if base_url is None:
            base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        if api_key is None:
            api_key = os.getenv("LLM_API_KEY", "not-needed")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.context_template = context_template or (
            "<document_context>\n{context}\n</document_context>\n\n{chunk}"
        )

    def _generate_context(
        self,
        document: Document,
        chunk: Chunk,
    ) -> str:
        """
        Generate contextual summary for a chunk.

        Args:
            document: Full document
            chunk: The chunk to contextualize

        Returns:
            Context string to prepend
        """
        prompt = f"""Here is a document:
<document>
{document.content[:4000]}
</document>

Here is a chunk from that document:
<chunk>
{chunk.text}
</chunk>

Write a very brief (1-2 sentence) context that situates this chunk within the overall document. Focus on:
- What section/topic this is from
- Key entities or concepts mentioned
- How it relates to the document's main subject

Context:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You write brief, informative context summaries.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=100,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            logger.warning(f"Context generation failed: {e}")
            # Fallback: use document source and metadata
            return f"From {document.source}"

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Chunk document with contextual prefixes.

        Args:
            document: Document to chunk

        Returns:
            List of contextualized chunks
        """
        # Use base chunker
        if self.base_chunker is None:
            from .chunking import Chunker

            self.base_chunker = Chunker()

        base_chunks = self.base_chunker.chunk_document(document)

        # Add context to each chunk
        contextualized_chunks = []
        for chunk in base_chunks:
            context = self._generate_context(document, chunk)

            # Create contextualized text
            contextualized_text = self.context_template.format(
                context=context,
                chunk=chunk.text,
            )

            contextualized_chunks.append(
                Chunk(
                    text=contextualized_text,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    metadata={
                        **chunk.metadata,
                        "original_text": chunk.text,
                        "context": context,
                    },
                )
            )

        return contextualized_chunks


class PropositionChunker:
    """
    Proposition-based chunking that extracts atomic factual statements.

    Breaks documents into discrete propositions (facts, claims, statements)
    that can be independently retrieved and verified.

    Reference: "Dense X Retrieval: What Retrieval Granularity Should We Use?"
    (Chen et al., 2023)
    """

    def __init__(
        self,
        model: str = "llama3.2",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_propositions_per_call: int = 20,
    ) -> None:
        """
        Initialize proposition chunker.

        Args:
            model: LLM model for extracting propositions
            api_key: API key
            base_url: API base URL
            max_propositions_per_call: Max propositions to extract per LLM call
        """
        import os

        from openai import OpenAI

        self.model = model
        self.max_propositions_per_call = max_propositions_per_call

        if base_url is None:
            base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        if api_key is None:
            api_key = os.getenv("LLM_API_KEY", "not-needed")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _extract_propositions(self, text: str) -> List[str]:
        """
        Extract atomic propositions from text.

        Args:
            text: Text to extract from

        Returns:
            List of propositions
        """
        prompt = f"""Extract atomic factual propositions from this text. Each proposition should:
- Be a single, self-contained fact or claim
- Be understandable without context
- Include relevant entity names (not just pronouns)
- Be concise but complete

Text:
{text}

Extract up to {self.max_propositions_per_call} propositions, one per line.
Do not number them or add any prefixes.

Propositions:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You extract atomic factual propositions from text.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1000,
            )

            content = response.choices[0].message.content or ""
            propositions = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and len(line.strip()) > 10
            ]

            return propositions

        except Exception as e:
            logger.warning(f"Proposition extraction failed: {e}")
            return [text]  # Fallback to original text

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Chunk document into propositions.

        Args:
            document: Document to chunk

        Returns:
            List of proposition chunks
        """
        # Split document into manageable sections first
        sections = document.content.split("\n\n")

        all_propositions = []
        for section in sections:
            if len(section.strip()) > 50:
                propositions = self._extract_propositions(section)
                all_propositions.extend(propositions)

        # Create chunks from propositions
        chunks = []
        for i, prop in enumerate(all_propositions):
            chunks.append(
                Chunk(
                    text=prop,
                    source=document.source,
                    chunk_index=i,
                    metadata={
                        **document.metadata,
                        "chunk_type": "proposition",
                    },
                )
            )

        return chunks


class HierarchicalChunker:
    """
    Hierarchical chunking with parent-child relationships.

    Creates both large parent chunks (for context) and small child chunks
    (for precise retrieval). Retrieval uses children, but the parent
    context is included in the final answer generation.
    """

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 400,
        overlap: int = 50,
    ) -> None:
        """
        Initialize hierarchical chunker.

        Args:
            parent_chunk_size: Size of parent chunks in characters
            child_chunk_size: Size of child chunks in characters
            overlap: Overlap between child chunks in characters
        """
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.overlap = overlap

    def _create_parent_chunks(
        self,
        text: str,
        source: str,
    ) -> List[Tuple[str, int]]:
        """Create parent-level chunks."""
        parents = []
        start = 0
        parent_idx = 0

        while start < len(text):
            end = start + self.parent_chunk_size

            # Try to end at sentence boundary
            if end < len(text):
                for boundary in [". ", ".\n", "! ", "? "]:
                    last_boundary = text[start:end].rfind(boundary)
                    if last_boundary > self.parent_chunk_size * 0.5:
                        end = start + last_boundary + len(boundary)
                        break

            parent_text = text[start:end].strip()
            if parent_text:
                parents.append((parent_text, parent_idx))
                parent_idx += 1

            start = end

        return parents

    def _create_child_chunks(
        self,
        parent_text: str,
        parent_idx: int,
        source: str,
        metadata: dict,
    ) -> List[Chunk]:
        """Create child chunks from a parent."""
        children = []
        start = 0
        child_idx = 0

        while start < len(parent_text):
            end = min(start + self.child_chunk_size, len(parent_text))

            # Try to end at word boundary
            if end < len(parent_text):
                last_space = parent_text[start:end].rfind(" ")
                if last_space > self.child_chunk_size * 0.5:
                    end = start + last_space

            child_text = parent_text[start:end].strip()
            if child_text:
                children.append(
                    Chunk(
                        text=child_text,
                        source=source,
                        chunk_index=child_idx,
                        metadata={
                            **metadata,
                            "parent_index": parent_idx,
                            "parent_text": parent_text,
                            "chunk_type": "child",
                        },
                    )
                )
                child_idx += 1

            # Ensure we make progress (avoid infinite loop)
            new_start = end - self.overlap
            start = max(new_start, start + 1)

        return children

    def chunk_document(
        self,
        document: Document,
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Chunk document into parent and child chunks.

        Args:
            document: Document to chunk

        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        parents = self._create_parent_chunks(document.content, document.source)

        parent_chunks = []
        child_chunks = []

        for parent_text, parent_idx in parents:
            # Create parent chunk
            parent_chunk = Chunk(
                text=parent_text,
                source=document.source,
                chunk_index=parent_idx,
                metadata={
                    **document.metadata,
                    "chunk_type": "parent",
                },
            )
            parent_chunks.append(parent_chunk)

            # Create child chunks
            children = self._create_child_chunks(
                parent_text,
                parent_idx,
                document.source,
                document.metadata,
            )
            child_chunks.extend(children)

        return parent_chunks, child_chunks
