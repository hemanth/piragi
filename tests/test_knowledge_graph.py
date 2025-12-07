"""Tests for knowledge graph functionality."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Direct import of just the knowledge_graph module to avoid full dependency chain
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_knowledge_graph_import_error():
    """Test helpful error when networkx is not installed."""
    from piragi import knowledge_graph as kg_module

    # Reset the cached import
    original_nx = kg_module._nx
    kg_module._nx = None

    with patch.dict("sys.modules", {"networkx": None}):
        with pytest.raises(ImportError, match="pip install piragi"):
            kg_module._get_networkx()

    # Restore
    kg_module._nx = original_nx


@pytest.fixture
def kg():
    """Create a KnowledgeGraph instance for testing."""
    from piragi.knowledge_graph import KnowledgeGraph

    return KnowledgeGraph()


@pytest.fixture
def kg_with_persist(tmp_path):
    """Create a KnowledgeGraph instance with persistence."""
    from piragi.knowledge_graph import KnowledgeGraph

    persist_path = str(tmp_path / "graph.json")
    return KnowledgeGraph(persist_path=persist_path), persist_path


def test_knowledge_graph_add_triple(kg):
    """Test adding triples to the graph."""
    kg.add_triple("Alice", "manages", "Bob")

    # Should store in triples list
    assert ("alice", "manages", "bob") in kg.triples()


def test_knowledge_graph_triples(kg):
    """Test getting all triples."""
    kg.add_triple("Alice", "manages", "Bob")
    kg.add_triple("Bob", "works_on", "Project X")

    triples = kg.triples()
    assert len(triples) == 2
    assert ("alice", "manages", "bob") in triples
    assert ("bob", "works_on", "project x") in triples


def test_knowledge_graph_search(kg):
    """Test searching triples."""
    kg.add_triple("Alice", "manages", "Bob")
    kg.add_triple("Bob", "works_on", "Project X")
    kg.add_triple("Charlie", "reports_to", "Alice")

    # Search for Alice
    results = kg.search("alice")
    assert len(results) == 2
    assert ("alice", "manages", "bob") in results
    assert ("charlie", "reports_to", "alice") in results


def test_knowledge_graph_to_context(kg):
    """Test converting to context string."""
    kg.add_triple("Alice", "manages", "Bob")
    kg.add_triple("Bob", "works_on", "Project X")

    context = kg.to_context("alice")
    assert "Known relationships:" in context
    assert "alice manages bob" in context


def test_knowledge_graph_to_context_empty(kg):
    """Test context with no matches."""
    context = kg.to_context("nonexistent")
    assert context == ""


def test_knowledge_graph_count(kg):
    """Test counting entities and relationships."""
    kg.add_triple("Alice", "manages", "Bob")
    kg.add_triple("Bob", "works_on", "Project X")

    counts = kg.count()
    assert counts["relationships"] == 2
    assert counts["entities"] == 3  # alice, bob, project x


def test_knowledge_graph_entities(kg):
    """Test getting all entities."""
    kg.add_triple("Alice", "manages", "Bob")
    kg.add_triple("Bob", "works_on", "Project X")

    entities = kg.entities()
    assert "alice" in entities
    assert "bob" in entities
    assert "project x" in entities


def test_knowledge_graph_neighbors(kg):
    """Test getting neighbors of an entity."""
    kg.add_triple("Alice", "manages", "Bob")
    kg.add_triple("Alice", "works_on", "Project X")

    neighbors = kg.neighbors("alice")
    assert "bob" in neighbors
    assert "project x" in neighbors


def test_knowledge_graph_clear(kg_with_persist):
    """Test clearing the graph."""
    kg, persist_path = kg_with_persist
    kg.add_triple("Alice", "manages", "Bob")

    # Save and verify file exists
    kg.save()
    assert os.path.exists(persist_path)

    # Clear
    kg.clear()
    assert len(kg.triples()) == 0
    assert not os.path.exists(persist_path)


def test_knowledge_graph_save_load(kg_with_persist):
    """Test saving and loading graph."""
    kg, persist_path = kg_with_persist

    # Create and save
    kg.add_triple("Alice", "manages", "Bob")
    kg.save()

    # Verify file contents
    with open(persist_path) as f:
        data = json.load(f)
    assert ("alice", "manages", "bob") in [tuple(t) for t in data["triples"]]


def test_knowledge_graph_load_on_init(tmp_path):
    """Test loading graph from existing file on init."""
    from piragi.knowledge_graph import KnowledgeGraph

    persist_path = str(tmp_path / "graph.json")

    # Create and save first instance
    kg1 = KnowledgeGraph(persist_path=persist_path)
    kg1.add_triple("Alice", "manages", "Bob")
    kg1.save()

    # Create second instance - should load existing data
    kg2 = KnowledgeGraph(persist_path=persist_path)
    assert ("alice", "manages", "bob") in kg2.triples()


def test_knowledge_graph_extract_and_add(kg):
    """Test entity extraction with LLM."""
    # Mock LLM client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '[["Alice", "manages", "Bob"]]'
    mock_client.chat.completions.create.return_value = mock_response

    triples = kg.extract_and_add(
        text="Alice manages Bob in the engineering team.",
        llm_client=mock_client,
        model="test-model",
    )

    assert len(triples) == 1
    assert triples[0] == ("Alice", "manages", "Bob")
    assert ("alice", "manages", "bob") in kg.triples()


def test_knowledge_graph_extract_handles_markdown(kg):
    """Test extraction handles markdown code blocks."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '```json\n[["Alice", "manages", "Bob"]]\n```'
    mock_client.chat.completions.create.return_value = mock_response

    triples = kg.extract_and_add(
        text="Alice manages Bob.",
        llm_client=mock_client,
        model="test-model",
    )

    assert len(triples) == 1
    assert ("alice", "manages", "bob") in kg.triples()


def test_knowledge_graph_extract_handles_errors(kg):
    """Test that extraction handles errors gracefully."""
    # Mock LLM client that raises error
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API error")

    triples = kg.extract_and_add(
        text="Some text",
        llm_client=mock_client,
        model="test-model",
    )

    # Should return empty list, not raise
    assert triples == []


def test_knowledge_graph_get_relations(kg):
    """Test getting relations for a specific entity."""
    kg.add_triple("Alice", "manages", "Bob")
    kg.add_triple("Bob", "works_on", "Project X")
    kg.add_triple("Charlie", "reports_to", "Alice")

    relations = kg.get_relations("alice")
    assert len(relations) == 2
    assert ("alice", "manages", "bob") in relations
    assert ("charlie", "reports_to", "alice") in relations
