"""Knowledge graph for entity and relationship extraction."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

# Lazy import networkx
_nx = None


def _get_networkx():
    """Lazy load networkx, raising helpful error if not installed."""
    global _nx
    if _nx is None:
        try:
            import networkx as nx
            _nx = nx
        except ImportError:
            raise ImportError(
                "networkx is required for knowledge graph support. "
                "Install it with: pip install piragi[graph] or pip install networkx"
            )
    return _nx


class KnowledgeGraph:
    """
    In-memory knowledge graph for entity-relationship extraction.

    Uses LLM to extract (subject, predicate, object) triples from text,
    stores them in a NetworkX graph for traversal and hybrid retrieval.
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        """
        Initialize knowledge graph.

        Args:
            persist_path: Optional path to persist/load graph as JSON
        """
        nx = _get_networkx()
        self._graph = nx.DiGraph()
        self._persist_path = persist_path
        self._triples: List[Tuple[str, str, str]] = []

        # Load existing graph if available
        if persist_path and os.path.exists(persist_path):
            self._load()

    def extract_and_add(self, text: str, llm_client: Any, model: str) -> List[Tuple[str, str, str]]:
        """
        Extract entities and relationships from text using LLM.

        Args:
            text: Text to extract from
            llm_client: OpenAI-compatible client
            model: Model name to use

        Returns:
            List of extracted (subject, predicate, object) triples
        """
        prompt = """Extract entities and relationships from the following text.
Return a JSON array of triples in the format: [["subject", "predicate", "object"], ...]

Rules:
- Extract only explicit relationships stated in the text
- Keep entity names concise but specific
- Use lowercase for predicates (e.g., "manages", "works_on", "is_part_of")
- Return empty array [] if no clear relationships found

Text:
{text}

Return ONLY the JSON array, no other text."""

        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt.format(text=text[:2000])}],
                temperature=0.0,
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            triples = json.loads(content)

            # Add to graph
            for triple in triples:
                if len(triple) == 3:
                    self.add_triple(triple[0], triple[1], triple[2])

            return [(t[0], t[1], t[2]) for t in triples if len(t) == 3]

        except Exception:
            # Silently fail extraction - don't break ingestion
            return []

    def add_triple(self, subject: str, predicate: str, obj: str) -> None:
        """Add a (subject, predicate, object) triple to the graph."""
        subject = subject.strip().lower()
        obj = obj.strip().lower()
        predicate = predicate.strip().lower()

        self._graph.add_edge(subject, obj, relation=predicate)
        self._triples.append((subject, predicate, obj))

    def neighbors(self, entity: str) -> List[str]:
        """Get all entities connected to the given entity."""
        entity = entity.strip().lower()
        if entity not in self._graph:
            return []

        # Get both predecessors and successors
        preds = list(self._graph.predecessors(entity))
        succs = list(self._graph.successors(entity))
        return list(set(preds + succs))

    def get_relations(self, entity: str) -> List[Tuple[str, str, str]]:
        """Get all triples involving the given entity."""
        entity = entity.strip().lower()
        results = []

        for s, p, o in self._triples:
            if s == entity or o == entity:
                results.append((s, p, o))

        return results

    def triples(self) -> List[Tuple[str, str, str]]:
        """Get all triples in the graph."""
        return self._triples.copy()

    def entities(self) -> List[str]:
        """Get all entities in the graph."""
        return list(self._graph.nodes())

    def search(self, query: str) -> List[Tuple[str, str, str]]:
        """
        Search for triples matching the query string.

        Simple substring matching on entities and predicates.
        """
        query = query.strip().lower()
        results = []

        for s, p, o in self._triples:
            if query in s or query in p or query in o:
                results.append((s, p, o))

        return results

    def to_context(self, query: str, max_triples: int = 10) -> str:
        """
        Convert relevant triples to context string for LLM.

        Args:
            query: Query to find relevant triples for
            max_triples: Maximum triples to include

        Returns:
            Formatted string of relevant relationships
        """
        # Find matching triples
        matches = self.search(query)[:max_triples]

        if not matches:
            return ""

        lines = ["Known relationships:"]
        for s, p, o in matches:
            lines.append(f"- {s} {p} {o}")

        return "\n".join(lines)

    def count(self) -> Dict[str, int]:
        """Return counts of entities and relationships."""
        return {
            "entities": len(self._graph.nodes()),
            "relationships": len(self._triples),
        }

    def clear(self) -> None:
        """Clear all data from the graph."""
        nx = _get_networkx()
        self._graph = nx.DiGraph()
        self._triples = []

        if self._persist_path and os.path.exists(self._persist_path):
            os.remove(self._persist_path)

    def save(self) -> None:
        """Save graph to disk."""
        if not self._persist_path:
            return

        data = {
            "triples": self._triples,
        }

        os.makedirs(os.path.dirname(self._persist_path) or ".", exist_ok=True)
        with open(self._persist_path, "w") as f:
            json.dump(data, f)

    def _load(self) -> None:
        """Load graph from disk."""
        if not self._persist_path or not os.path.exists(self._persist_path):
            return

        try:
            with open(self._persist_path, "r") as f:
                data = json.load(f)

            for s, p, o in data.get("triples", []):
                self.add_triple(s, p, o)
        except Exception:
            # Ignore load errors
            pass
