"""Query transformation techniques for improved retrieval."""

import logging
from typing import List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class HyDE:
    """
    Hypothetical Document Embeddings (HyDE) implementation.

    HyDE improves retrieval by generating a hypothetical answer to the query,
    then using that answer for embedding-based retrieval. This bridges the
    vocabulary gap between short queries and longer documents.

    Reference: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    (Gao et al., 2022)
    """

    def __init__(
        self,
        model: str = "llama3.2",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        num_hypothetical: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> None:
        """
        Initialize HyDE.

        Args:
            model: LLM model for generating hypothetical documents
            api_key: API key for LLM
            base_url: Base URL for OpenAI-compatible API
            num_hypothetical: Number of hypothetical documents to generate
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens for hypothetical document
        """
        import os

        self.model = model
        self.num_hypothetical = num_hypothetical
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Default to Ollama
        if base_url is None:
            base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        if api_key is None:
            api_key = os.getenv("LLM_API_KEY", "not-needed")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query.

        Args:
            query: The user's query

        Returns:
            Hypothetical document/answer
        """
        prompt = f"""Write a detailed passage that would be found in a document that answers this question:

Question: {query}

Write the passage as if it's from an authoritative source document. Include specific details, facts, and technical information that would help answer the question. Do not include phrases like "This document explains" - just write the content directly.

Passage:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical writer creating documentation passages. Write factual, detailed content.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return response.choices[0].message.content or query

        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}, using original query")
            return query

    def generate_multiple(self, query: str) -> List[str]:
        """
        Generate multiple hypothetical documents for the query.

        Args:
            query: The user's query

        Returns:
            List of hypothetical documents
        """
        hypotheticals = []
        for i in range(self.num_hypothetical):
            # Vary temperature slightly for diversity
            self.temperature = 0.7 + (i * 0.1)
            doc = self.generate_hypothetical_document(query)
            hypotheticals.append(doc)

        return hypotheticals

    def transform_query(self, query: str) -> str:
        """
        Transform query using HyDE.

        For single hypothetical mode, returns the hypothetical document.
        For multiple, returns the original query (embeddings should be
        computed separately for each hypothetical).

        Args:
            query: Original query

        Returns:
            Transformed query (hypothetical document)
        """
        return self.generate_hypothetical_document(query)


class QueryExpander:
    """
    Query expansion through LLM-generated variations.

    Generates multiple phrasings of the same query to improve recall.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        num_expansions: int = 2,
    ) -> None:
        """
        Initialize query expander.

        Args:
            model: LLM model for expansion
            api_key: API key
            base_url: API base URL
            num_expansions: Number of alternative queries to generate
        """
        import os

        self.model = model
        self.num_expansions = num_expansions

        if base_url is None:
            base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        if api_key is None:
            api_key = os.getenv("LLM_API_KEY", "not-needed")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def expand(self, query: str) -> List[str]:
        """
        Expand query into multiple variations.

        Args:
            query: Original query

        Returns:
            List of query variations including original
        """
        prompt = f"""Given this question: "{query}"

Generate {self.num_expansions} alternative phrasings that preserve the same meaning but use different words or structure.
Return only the alternatives, one per line, without numbering or explanation."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that rephrases questions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=150,
            )

            alternatives = response.choices[0].message.content or ""
            variations = [query]
            for line in alternatives.split("\n"):
                line = line.strip()
                if line and len(line) > 5:  # Filter empty/too short
                    # Remove numbering if present
                    if line[0].isdigit() and line[1] in ".):":
                        line = line[2:].strip()
                    variations.append(line)

            return variations[: self.num_expansions + 1]

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return [query]


class MultiQueryRetriever:
    """
    Combines HyDE and query expansion for comprehensive retrieval.

    Generates both hypothetical documents and query variations,
    then merges results from all retrieval passes.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_hyde: bool = True,
        use_expansion: bool = True,
        num_expansions: int = 2,
    ) -> None:
        """
        Initialize multi-query retriever.

        Args:
            model: LLM model
            api_key: API key
            base_url: API base URL
            use_hyde: Enable HyDE
            use_expansion: Enable query expansion
            num_expansions: Number of query variations
        """
        self.use_hyde = use_hyde
        self.use_expansion = use_expansion

        if use_hyde:
            self.hyde = HyDE(model=model, api_key=api_key, base_url=base_url)
        else:
            self.hyde = None

        if use_expansion:
            self.expander = QueryExpander(
                model=model,
                api_key=api_key,
                base_url=base_url,
                num_expansions=num_expansions,
            )
        else:
            self.expander = None

    def get_queries(self, query: str) -> List[str]:
        """
        Get all query variations for retrieval.

        Args:
            query: Original query

        Returns:
            List of queries to use for retrieval
        """
        queries = [query]

        # Add HyDE hypothetical document
        if self.hyde:
            try:
                hypothetical = self.hyde.transform_query(query)
                queries.append(hypothetical)
            except Exception as e:
                logger.warning(f"HyDE failed: {e}")

        # Add expanded variations
        if self.expander:
            try:
                expansions = self.expander.expand(query)
                # Skip first (original) since we already have it
                queries.extend(expansions[1:])
            except Exception as e:
                logger.warning(f"Expansion failed: {e}")

        return queries


class StepBackPrompting:
    """
    Step-back prompting for complex queries.

    Generates a more general/abstract version of the query to retrieve
    broader context, then uses that to answer the specific question.

    Reference: "Take a Step Back: Evoking Reasoning via Abstraction"
    (Zheng et al., 2023)
    """

    def __init__(
        self,
        model: str = "llama3.2",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """Initialize step-back prompting."""
        import os

        self.model = model

        if base_url is None:
            base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        if api_key is None:
            api_key = os.getenv("LLM_API_KEY", "not-needed")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_stepback_query(self, query: str) -> str:
        """
        Generate a more general step-back query.

        Args:
            query: Specific query

        Returns:
            More general/abstract query
        """
        prompt = f"""Given this specific question: "{query}"

Generate a more general, higher-level question that would help provide context for answering the specific question.

For example:
- Specific: "What is the error code for invalid API key in the auth module?"
- General: "How does the authentication system handle errors?"

- Specific: "How do I configure the Redis cache timeout?"
- General: "How does the caching system work and what are its configuration options?"

Now generate a general question for the given specific question. Return only the general question, nothing else.

General question:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You generate higher-level questions to provide context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=100,
            )

            return response.choices[0].message.content or query

        except Exception as e:
            logger.warning(f"Step-back generation failed: {e}")
            return query

    def get_queries(self, query: str) -> List[str]:
        """
        Get both original and step-back queries.

        Args:
            query: Original specific query

        Returns:
            List containing [original_query, stepback_query]
        """
        stepback = self.generate_stepback_query(query)
        return [query, stepback]
