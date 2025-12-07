"""Embedding generation using local or remote models."""

import os
from typing import List, Optional

from .types import Chunk


class EmbeddingGenerator:
    """Generate embeddings using local sentence-transformers or remote API."""

    def __init__(
        self,
        model: str = "nvidia/llama-embed-nemotron-8b",
        device: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        Initialize the embedding generator.

        Args:
            model: Embedding model to use (default: nvidia/llama-embed-nemotron-8b)
            device: Device to run on ('cuda', 'cpu', or None for auto-detect) - only for local models
            base_url: Optional API base URL for remote embeddings (e.g., https://api.openai.com/v1)
            api_key: Optional API key for remote embeddings
        """
        self.model_name = model
        self.base_url = base_url
        self.api_key = api_key
        self.use_remote = base_url is not None

        if self.use_remote:
            # Use OpenAI-compatible API client
            from openai import OpenAI

            if self.api_key is None:
                self.api_key = os.getenv("EMBEDDING_API_KEY", "not-needed")

            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.model = None
        else:
            # Use local sentence-transformers
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model,
                trust_remote_code=True,
                device=device,
            )
            self.client = None

    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Generate embeddings for a list of chunks.

        Args:
            chunks: List of chunks to embed

        Returns:
            Chunks with embeddings added
        """
        if not chunks:
            return chunks

        # Extract texts
        texts = [chunk.text for chunk in chunks]

        # Generate embeddings in batch (documents don't need instruction prefix)
        embeddings = self._generate_embeddings(texts)

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            # Handle both numpy arrays (local) and lists (remote/Ollama)
            if hasattr(embedding, 'tolist'):
                chunk.embedding = embedding.tolist()
            else:
                chunk.embedding = embedding

        return chunks

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        try:
            if self.use_remote:
                # Use OpenAI-compatible API
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model_name,
                )
                return [item.embedding for item in response.data]
            else:
                # Use local sentence-transformers
                # Use encode_document for document chunks if available
                if hasattr(self.model, "encode_document"):
                    embeddings = self.model.encode_document(texts)
                else:
                    embeddings = self.model.encode(texts)
                return embeddings

        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {e}")

    def embed_query(self, query: str, task_instruction: str | None = None) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            query: Query text
            task_instruction: Optional task instruction for query
                (e.g., "Retrieve relevant documents for this question")

        Returns:
            Embedding vector
        """
        try:
            if self.use_remote:
                # Use OpenAI-compatible API
                query_text = query
                if task_instruction:
                    query_text = f"{task_instruction}\n{query}"

                response = self.client.embeddings.create(
                    input=query_text,
                    model=self.model_name,
                )
                return response.data[0].embedding
            else:
                # Use local sentence-transformers
                # Use encode_query for search queries if available
                if hasattr(self.model, "encode_query"):
                    if task_instruction:
                        query_with_instruction = f"{task_instruction}\n{query}"
                        embedding = self.model.encode_query(query_with_instruction)
                    else:
                        embedding = self.model.encode_query(query)
                else:
                    query_text = query
                    if task_instruction:
                        query_text = f"{task_instruction}\n{query}"
                    embedding = self.model.encode(query_text)

                # Handle both numpy arrays and lists
                if hasattr(embedding, 'tolist'):
                    return embedding.tolist()
                return embedding

        except Exception as e:
            raise RuntimeError(f"Failed to generate query embedding: {e}")
