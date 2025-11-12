"""Retrieval and answer generation using OpenAI-compatible APIs."""

import os
from typing import List, Optional

from openai import OpenAI

from .types import Answer, Citation


class Retriever:
    """Generate answers from retrieved chunks using OpenAI-compatible LLM APIs."""

    def __init__(
        self,
        model: str = "llama3.2",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
        enable_reranking: bool = True,
        enable_query_expansion: bool = True,
    ) -> None:
        """
        Initialize the retriever.

        Args:
            model: Model name to use (default: llama3.2 for Ollama)
            api_key: API key (optional for local models like Ollama)
            base_url: Base URL for OpenAI-compatible API (e.g., http://localhost:11434/v1 for Ollama)
            temperature: Sampling temperature (0.0-1.0, default: 0.1)
            enable_reranking: Enable reranking of results (default: True)
            enable_query_expansion: Enable query expansion for better retrieval (default: True)
        """
        self.model = model
        self.temperature = temperature
        self.enable_reranking = enable_reranking
        self.enable_query_expansion = enable_query_expansion

        # Default to Ollama if no base_url provided
        if base_url is None:
            base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")

        # API key is optional for local models
        if api_key is None:
            api_key = os.getenv("LLM_API_KEY", "not-needed")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def expand_query(self, query: str) -> List[str]:
        """
        Expand query into multiple variations for better retrieval.

        Args:
            query: Original query

        Returns:
            List of query variations including original
        """
        if not self.enable_query_expansion:
            return [query]

        try:
            expansion_prompt = f"""Given this question: "{query}"

Generate 2 alternative phrasings that preserve the same meaning but use different words.
Return only the alternatives, one per line, without numbering or explanation."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that rephrases questions."},
                    {"role": "user", "content": expansion_prompt},
                ],
                temperature=0.7,
                max_tokens=100,
            )

            alternatives = response.choices[0].message.content or ""
            variations = [query] + [line.strip() for line in alternatives.split('\n') if line.strip()]
            return variations[:3]  # Original + 2 alternatives

        except Exception:
            # Fallback to original query on error
            return [query]

    def rerank_citations(self, query: str, citations: List[Citation]) -> List[Citation]:
        """
        Rerank citations by relevance to query using simple scoring.

        Args:
            query: User's question
            citations: List of citations to rerank

        Returns:
            Reranked citations
        """
        if not self.enable_reranking or len(citations) <= 1:
            return citations

        # Simple keyword-based reranking
        query_terms = set(query.lower().split())

        def relevance_score(citation: Citation) -> float:
            chunk_terms = set(citation.chunk.lower().split())
            # Combine original vector score with keyword overlap
            keyword_overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            # Weight: 70% vector similarity, 30% keyword overlap
            return 0.7 * citation.score + 0.3 * keyword_overlap

        # Rerank by combined score
        reranked = sorted(citations, key=relevance_score, reverse=True)
        return reranked

    def generate_answer(
        self,
        query: str,
        citations: List[Citation],
        system_prompt: Optional[str] = None,
    ) -> Answer:
        """
        Generate an answer from retrieved citations.

        Args:
            query: User's question
            citations: Retrieved citations
            system_prompt: Optional custom system prompt

        Returns:
            Answer with citations
        """
        if not citations:
            return Answer(
                text="I couldn't find any relevant information to answer your question.",
                citations=[],
                query=query,
            )

        # Rerank citations if enabled
        if self.enable_reranking:
            citations = self.rerank_citations(query, citations)

        # Build context from citations
        context = self._build_context(citations)

        # Generate answer
        answer_text = self._generate_with_llm(query, context, system_prompt)

        return Answer(
            text=answer_text,
            citations=citations,
            query=query,
        )

    def _build_context(self, citations: List[Citation]) -> str:
        """Build context string from citations."""
        context_parts = []

        for i, citation in enumerate(citations, 1):
            source_info = f"Source {i} ({citation.source}):"
            context_parts.append(f"{source_info}\n{citation.chunk}\n")

        return "\n".join(context_parts)

    def _generate_with_llm(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate answer using OpenAI-compatible API."""
        if system_prompt is None:
            system_prompt = (
                "You are an expert assistant. Answer questions based ONLY on the provided context sources. "
                "Your answer must be grounded in the given sources - do not add information from outside knowledge. "
                "Always cite sources using 'According to Source X' or 'Source X states'. "
                "If the context lacks information to fully answer the question, explain what you found and what's missing. "
                "Be specific, detailed, and accurate in your responses."
            )

        user_prompt = f"""Context from documents:

{context}

Question: {query}

Please answer the question based on the context provided above. Cite your sources."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            raise RuntimeError(f"Failed to generate answer: {e}")
