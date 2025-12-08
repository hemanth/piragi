"""Smart chunking strategies for documents."""

import re
from typing import List

import pysbd
from transformers import AutoTokenizer

from .types import Chunk, Document


class Chunker:
    """Smart document chunker with markdown awareness."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        tokenizer_name: str = "nvidia/llama-embed-nemotron-8b",
    ) -> None:
        """
        Initialize the chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            tokenizer_name: Tokenizer to use (default: nvidia/llama-embed-nemotron-8b)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._segmenter = pysbd.Segmenter(language="en", clean=False)

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Chunk a document into smaller pieces.

        Args:
            document: Document to chunk

        Returns:
            List of chunks
        """
        # Split by markdown headers first to respect document structure
        sections = self._split_by_headers(document.content)

        chunks = []
        chunk_index = 0

        for section in sections:
            section_chunks = self._chunk_text(section, document.source, chunk_index)
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        # Add document metadata to all chunks
        for chunk in chunks:
            chunk.metadata.update(document.metadata)

        return chunks

    def _split_by_headers(self, text: str) -> List[str]:
        """Split text by markdown headers while preserving structure."""
        # Pattern to match markdown headers (# Header)
        header_pattern = r"^(#{1,6}\s+.+)$"

        lines = text.split("\n")
        sections = []
        current_section = []

        for line in lines:
            if re.match(header_pattern, line.strip()):
                # Save previous section if it exists
                if current_section:
                    sections.append("\n".join(current_section))
                    current_section = []

            # Always add the line to current section (including headers)
            current_section.append(line)

        # Add the last section
        if current_section:
            sections.append("\n".join(current_section))

        return sections if sections else [text]

    def _chunk_text(self, text: str, source: str, start_index: int) -> List[Chunk]:
        """
        Chunk text into token-sized pieces with overlap.

        Args:
            text: Text to chunk
            source: Source identifier
            start_index: Starting chunk index

        Returns:
            List of chunks
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= self.chunk_size:
            return [
                Chunk(
                    text=text,
                    source=source,
                    chunk_index=start_index,
                    metadata={},
                )
            ]

        chunks = []
        start = 0
        chunk_idx = start_index

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            # Try to break at sentence boundary if possible
            if end < len(tokens):
                chunk_text = self._break_at_sentence(chunk_text)

            # Re-encode to get actual token length used
            actual_tokens = self.tokenizer.encode(chunk_text, add_special_tokens=False)
            actual_len = len(actual_tokens)

            chunks.append(
                Chunk(
                    text=chunk_text.strip(),
                    source=source,
                    chunk_index=chunk_idx,
                    metadata={},
                )
            )

            # Calculate the step for the next chunk's start, ensuring the desired overlap
            # while always making forward progress.
            step = max(1, actual_len - self.chunk_overlap)
            start += step
            chunk_idx += 1

        return chunks

    def _break_at_sentence(self, text: str) -> str:
        """Try to break text at a sentence boundary using pysbd.

        Uses pysbd for accurate sentence boundary detection that handles:
        - Numbered lists (1. 2. 3.)
        - Abbreviations (Dr., Mr., etc.)
        - Acronyms (U.S., Ph.D., B.A.)
        - Initials (J.K. Rowling)
        """
        sentences = self._segmenter.segment(text)

        if len(sentences) <= 1:
            return text

        # Find a break point in the latter half of the text
        half_len = len(text) * 0.5
        accumulated = ""

        for sentence in sentences:
            accumulated += sentence
            # Break after a sentence that ends past the halfway point
            if len(accumulated) >= half_len:
                return accumulated

        # If no good break point, return as is
        return text
