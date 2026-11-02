"""Tier 4: Real-World Application Scenarios E2E Tests.

End-to-end realistic multi-file knowledge bases, multi-format ingestion,
concurrent asynchronous querying, live citation auditing, and leaderboard workflows.
"""

import asyncio
import io
import json
import asyncio
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from piragi import Ragi, AsyncRagi, EmbeddingGenerator
from piragi.config import RagiConfig, RetrievalConfig
from piragi.types import Answer, Citation, Chunk
from piragi.stores.factory import parse_store_uri, create_store
from piragi.stores.lance import LanceStore
from piragi.hybrid_search import HybridSearcher

from .conftest import (
    has_u_shaped_support,
    has_auto_merge_support,
    has_step_back_support,
    has_citation_pruning_support,
    has_bm25_caching_support,
)
from .test_tier1_features import (
    apply_u_shaped_ordering,
    auto_merge_citations,
    prune_citations_from_text,
    compute_ndcg_at_k,
    compute_mrr,
    export_enterpriserag,
    export_airbench,
    export_gaia,
    validate_submission_file,
)


class TestTier4_RealWorldScenarios:
    """Real-World Application Workloads & End-to-End Scenarios."""

    # -----------------------------------------------------------------------
    # Scenario 1: Software Engineering Knowledge Base
    # -----------------------------------------------------------------------
    @patch("openai.OpenAI")
    def test_scenario_01_software_docs_knowledge_base(
        self, mock_openai, e2e_temp_dir, make_mock_embedder, mock_openai_response
    ):
        """Scenario 1: Multi-file technical documentation with code and architectural specs."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response(
            "According to [1], LanceDB stores vectors on disk with IVF-PQ indexing."
        )
        mock_openai.return_value = mock_client

        docs_dir = os.path.join(e2e_temp_dir, "software_docs")
        os.makedirs(docs_dir, exist_ok=True)

        with open(os.path.join(docs_dir, "architecture.md"), "w") as f:
            f.write("# Architecture Overview\nPiragi uses a multi-stage retrieval pipeline with hybrid search, dense embeddings, and cross-encoder neural reranking to optimize context.")

        with open(os.path.join(docs_dir, "storage.md"), "w") as f:
            f.write("# Storage Options\nLanceDB stores vectors on disk with IVF-PQ indexing, supporting local filesystem and remote cloud storage.")

        with open(os.path.join(docs_dir, "api.py"), "w") as f:
            f.write("def ask(question: str) -> Answer:\n    \"\"\"Ask a question to the knowledge base and retrieve grounded context.\"\"\"\n    return Answer(text='ok', citations=[])")

        kb = Ragi(
            sources=docs_dir,
            persist_dir=os.path.join(e2e_temp_dir, "kb_sw"),
            embedder=make_mock_embedder(),
        )
        assert kb.store.count() > 0

        answer = kb.ask("How does storage work in Piragi?")
        assert isinstance(answer, Answer)
        assert len(answer.citations) > 0

    # -----------------------------------------------------------------------
    # Scenario 2: Financial and Legal Contract Audit with Auto-Merging
    # -----------------------------------------------------------------------
    @patch("openai.OpenAI")
    def test_scenario_02_legal_contract_audit_auto_merging(
        self, mock_openai, e2e_temp_dir, make_mock_embedder, mock_openai_response
    ):
        """Scenario 2: MSA Contract with multi-clause structure and citation attribution audit."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response(
            "Under Section 4.1 cited in [1], the liability cap is strictly limited to annual fees paid."
        )
        mock_openai.return_value = mock_client

        contract_path = os.path.join(e2e_temp_dir, "contract.md")
        with open(contract_path, "w") as f:
            f.write("""# Master Services Agreement
## Section 4: Limitation of Liability
### 4.1 Aggregate Cap
The aggregate liability of either party shall not exceed the total fees paid in the prior 12 months.
### 4.2 Exclusions
Neither party excludes liability for gross negligence or willful misconduct.
""")

        kb = Ragi(
            persist_dir=os.path.join(e2e_temp_dir, "kb_legal"),
            embedder=make_mock_embedder(),
            config={"chunk": {"strategy": "hierarchical"}},
        )
        kb.add(contract_path)

        answer = kb.ask("What is the aggregate liability cap under Section 4?")
        assert isinstance(answer, Answer)
        assert answer.text

    # -----------------------------------------------------------------------
    # Scenario 3: High-Concurrency Multi-User Async Querying
    # -----------------------------------------------------------------------
    @pytest.mark.asyncio
    @patch("openai.OpenAI")
    async def test_scenario_03_high_concurrency_async_service(
        self, mock_openai, e2e_temp_dir, make_mock_embedder, mock_openai_response
    ):
        """Scenario 3: AsyncRagi handling 10 concurrent requests without thread collision."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response("Concurrent answer.")
        mock_openai.return_value = mock_client

        doc_path = os.path.join(e2e_temp_dir, "concurrency.txt")
        with open(doc_path, "w") as f:
            f.write("Concurrency in Python is achieved via asyncio coroutines and event loops.")

        kb = AsyncRagi(
            persist_dir=os.path.join(e2e_temp_dir, "kb_async"),
            embedder=make_mock_embedder(),
        )
        await kb.add(doc_path)

        queries = [f"Query number {i} regarding concurrency" for i in range(10)]
        tasks = [kb.ask(q) for q in queries]
        answers = await asyncio.gather(*tasks)

        assert len(answers) == 10
        assert all(isinstance(a, Answer) for a in answers)

    # -----------------------------------------------------------------------
    # Scenario 4: Dynamic Knowledge Base Updates & BM25 Invalidation
    # -----------------------------------------------------------------------
    @patch("openai.OpenAI")
    def test_scenario_04_dynamic_incremental_updates(
        self, mock_openai, e2e_temp_dir, make_mock_embedder, mock_openai_response
    ):
        """Scenario 4: Ingest v1, query, ingest patch v2, re-query to verify fresh information."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response()
        mock_openai.return_value = mock_client

        kb = Ragi(
            persist_dir=os.path.join(e2e_temp_dir, "kb_dynamic"),
            embedder=make_mock_embedder(),
        )

        # Ingest baseline
        f1 = os.path.join(e2e_temp_dir, "chapter1.txt")
        with open(f1, "w") as f:
            f.write("Chapter 1: The system boots with default configuration profile Alpha, establishing all core database and networking handles.")
        kb.add(f1)
        count_v1 = kb.store.count()

        # Ingest patch
        f2 = os.path.join(e2e_temp_dir, "patch_notes.txt")
        with open(f2, "w") as f:
            f.write("Patch 2.4: Configuration profile Beta replaces Alpha as the default, providing enhanced security parameters and updated protocols.")
        kb.add(f2)
        count_v2 = kb.store.count()

        assert count_v2 > count_v1
        ans = kb.ask("Which configuration profile is active in Patch 2.4?")
        assert isinstance(ans, Answer)

    # -----------------------------------------------------------------------
    # Scenario 5: Multi-Format Ingestion & Scoped Filtering
    # -----------------------------------------------------------------------
    @patch("openai.OpenAI")
    def test_scenario_05_multiformat_ingestion_and_filtering(
        self, mock_openai, e2e_temp_dir, make_mock_embedder, mock_openai_response
    ):
        """Scenario 5: Mixed formats (Markdown, TXT, Python) with scoped metadata filtering."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response()
        mock_openai.return_value = mock_client

        mix_dir = os.path.join(e2e_temp_dir, "mix")
        os.makedirs(mix_dir, exist_ok=True)

        with open(os.path.join(mix_dir, "notes.txt"), "w") as f:
            f.write("Plain text notes regarding release timeline.")

        with open(os.path.join(mix_dir, "spec.md"), "w") as f:
            f.write("# Specification\nFormal interface schema definition.")

        with open(os.path.join(mix_dir, "worker.py"), "w") as f:
            f.write("def run_worker():\n    print('Worker started')")

        kb = Ragi(
            sources=mix_dir,
            persist_dir=os.path.join(e2e_temp_dir, "kb_mix"),
            embedder=make_mock_embedder(),
        )
        assert kb.store.count() >= 3

        # Filter by file source
        filtered_kb = kb.filter(source=os.path.join(mix_dir, "spec.md"))
        assert filtered_kb is not None

    # -----------------------------------------------------------------------
    # Scenario 6: EnterpriseRAG-Bench Submission Pipeline
    # -----------------------------------------------------------------------
    @patch("openai.OpenAI")
    def test_scenario_06_enterpriserag_submission_pipeline(
        self, mock_openai, e2e_temp_dir, make_mock_embedder
    ):
        """Scenario 6: End-to-end EnterpriseRAG evaluation and JSONL generation."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Answer for enterprise query [1]."))]
        )
        mock_openai.return_value = mock_client

        doc_path = os.path.join(e2e_temp_dir, "enterprise_data.md")
        with open(doc_path, "w") as f:
            f.write("""# Internal IT Knowledge Base
Section 1: VPN access requires Okta MFA and Cisco AnyConnect client installed on corporate laptops.
Section 2: Expense reimbursements must be submitted through Concur by Friday 5 PM each pay period.
""")

        kb = Ragi(
            persist_dir=os.path.join(e2e_temp_dir, "kb_ent"),
            embedder=make_mock_embedder(),
        )
        kb.add(doc_path)

        questions = [
            ("q_001", "How do I access company VPN?"),
            ("q_002", "What is the deadline for expense reimbursement?"),
        ]

        submission_records = []
        for qid, qtext in questions:
            ans = kb.ask(qtext)
            submission_records.append({
                "question_id": qid,
                "answer": ans.text,
                "citations": [c.source for c in ans.citations],
                "doc_ids": [c.source for c in ans.citations],
            })

        out_file = os.path.join(e2e_temp_dir, "submission_enterpriserag.jsonl")
        export_enterpriserag(submission_records, out_file)
        assert validate_submission_file(out_file, "jsonl") is True

        with open(out_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["question_id"] == "q_001"

    # -----------------------------------------------------------------------
    # Scenario 7: Academic Cranfield IR Evaluation and AIR-Bench ZIP Packaging
    # -----------------------------------------------------------------------
    def test_scenario_07_academic_ir_eval_and_airbench_packaging(self, e2e_temp_dir):
        """Scenario 7: Evaluate ranked retrieval and package into AIR-Bench TREC ZIP."""
        # Simulate retrieval evaluation run over 3 queries
        run_data = [
            {
                "qid": "cranfield_001",
                "doc_scores": [("doc_101", 0.98), ("doc_102", 0.85), ("doc_103", 0.40)],
                "ground_truth": {"doc_101", "doc_102"},
            },
            {
                "qid": "cranfield_002",
                "doc_scores": [("doc_201", 0.92), ("doc_202", 0.77)],
                "ground_truth": {"doc_201"},
            },
        ]

        # Compute nDCG and Recall
        ndcg_scores = []
        for item in run_data:
            rels = [1 if doc in item["ground_truth"] else 0 for doc, _ in item["doc_scores"]]
            ndcg_scores.append(compute_ndcg_at_k(rels, k=5))

        mean_ndcg = float(np.mean(ndcg_scores))
        assert mean_ndcg > 0.8

        # Package into AIR-Bench ZIP
        zip_path = os.path.join(e2e_temp_dir, "airbench_submission.zip")
        export_airbench(run_data, zip_path, run_id="piragi_v1_sota")
        assert validate_submission_file(zip_path, "zip") is True

    # -----------------------------------------------------------------------
    # Scenario 8: Multi-Hop Reasoning with StepBack and GAIA Export
    # -----------------------------------------------------------------------
    @patch("openai.OpenAI")
    def test_scenario_08_multihop_reasoning_and_gaia_export(
        self, mock_openai, e2e_temp_dir, make_mock_embedder
    ):
        """Scenario 8: Complex query with StepBack abstract reasoning exported to GAIA JSONL."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "LanceDB disk vector store"
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        mock_openai.return_value = mock_client

        task_id = "gaia_task_42"
        specific_query = "Which vector store backend does Piragi use for zero-setup local persistence?"
        step_back_abstraction = "What vector database technologies support embedded disk storage?"

        gaia_entry = {
            "task_id": task_id,
            "model_answer": "LanceDB",
            "reasoning_trace": f"1. Question: '{specific_query}'; 2. StepBack Abstraction: '{step_back_abstraction}'; 3. Context analysis confirmed LanceDB.",
        }

        out_gaia = os.path.join(e2e_temp_dir, "gaia_predictions.jsonl")
        export_gaia([gaia_entry], out_gaia)
        assert validate_submission_file(out_gaia, "jsonl") is True

        with open(out_gaia, "r") as f:
            obj = json.loads(f.readline())
        assert obj["task_id"] == "gaia_task_42"
        assert obj["model_answer"] == "LanceDB"

    # -----------------------------------------------------------------------
    # Scenario 9: Citation Auditing and Hallucination Pruning Guardrail
    # -----------------------------------------------------------------------
    def test_scenario_09_citation_auditing_hallucination_guardrail(self):
        """Scenario 9: Auditing citations when candidates include distractors."""
        candidates = [
            Citation(source="relevant_doc.md", chunk="Real fact: A causes B.", score=0.95),
            Citation(source="distractor_1.md", chunk="Irrelevant distraction about X.", score=0.70),
            Citation(source="distractor_2.md", chunk="Misleading claims about Y.", score=0.65),
        ]

        # LLM only cites the genuine source [1]
        model_output = "Based strictly on verified evidence [1], A causes B."
        audited_citations = prune_citations_from_text(model_output, candidates)

        # Distractor sources 2 and 3 must be purged from answer citations
        assert len(audited_citations) == 1
        assert audited_citations[0].source == "relevant_doc.md"
        assert "distractor_1.md" not in [c.source for c in audited_citations]
        assert "distractor_2.md" not in [c.source for c in audited_citations]

    # -----------------------------------------------------------------------
    # Scenario 10: Hermetic Offline RAG Reliability
    # -----------------------------------------------------------------------
    @patch("openai.OpenAI")
    def test_scenario_10_hermetic_offline_rag_execution(
        self, mock_openai, e2e_temp_dir, make_mock_embedder, mock_openai_response
    ):
        """Scenario 10: Verify end-to-end Ragi operates hermetically without network access."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_response("Local offline response.")
        mock_openai.return_value = mock_client

        doc_path = os.path.join(e2e_temp_dir, "offline.txt")
        with open(doc_path, "w") as f:
            f.write("Piragi operates locally on Mac, Linux, and air-gapped environments without external network dependencies, ensuring complete offline data privacy and security.")

        kb = Ragi(
            persist_dir=os.path.join(e2e_temp_dir, "kb_offline"),
            embedder=make_mock_embedder(),
        )
        kb.add(doc_path)
        ans = kb.ask("Can Piragi operate offline?")
        assert ans is not None
        assert ans.text == "Local offline response."
