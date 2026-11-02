"""Example: Different embedding configuration options."""

import os
import tempfile
from piragi import Ragi


def create_sample_doc():
    """Create a sample document."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(
            """# Embedding Models in Ragi

Ragi supports two types of embedding models:

## Local Models
Use sentence-transformers models that run locally on your machine.
No API calls, completely free, and works offline.

Examples:
- google/embeddinggemma-2 (768d multimodal & MRL text/code/image/audio embeddings)
- sentence-transformers/all-MiniLM-L6-v2 (fast, lightweight)
- nvidia/llama-embed-nemotron-8b (high quality, requires auth)
- any model from HuggingFace sentence-transformers

## Remote API Models
Connect to OpenAI-compatible embedding APIs.
Useful for cloud deployments or when you want to use proprietary models.

Examples:
- OpenAI text-embedding-3-small
- Custom embedding services
- Self-hosted embedding APIs
"""
        )
        return f.name


def main():
    """Demonstrate different embedding configuration options."""
    doc_path = create_sample_doc()

    try:
        print("=" * 60)
        print("Example 1: Local Embeddings (Default)")
        print("=" * 60)

        # Local embeddings with sentence-transformers
        kb1 = Ragi(
            doc_path,
            config={"embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2"}},
        )

        print(f"✓ Loaded {kb1.count()} chunks")
        print(f"  Model: {kb1.embedder.model_name}")
        print(f"  Remote API: {kb1.embedder.use_remote}")

        answer = kb1.ask("What are the two types of embedding models?")
        print(f"\nQ: What are the two types of embedding models?")
        print(f"A: {answer.text}\n")

        print("=" * 60)
        print("Example 2: Local Embeddings with GPU")
        print("=" * 60)

        # Local embeddings with GPU acceleration
        kb2 = Ragi(
            doc_path,
            config={
                "embedding": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "device": "cpu",  # Change to "cuda" if you have GPU
                }
            },
        )

        print(f"✓ Loaded {kb2.count()} chunks")
        print(f"  Model: {kb2.embedder.model_name}")
        print(f"  Device: cpu")
        print()

        print("=" * 60)
        print("Example 3: Remote API Embeddings (OpenAI)")
        print("=" * 60)

        # Check if OpenAI API key is available
        if os.getenv("OPENAI_API_KEY"):
            kb3 = Ragi(
                doc_path,
                config={
                    "embedding": {
                        "model": "text-embedding-3-small",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": os.getenv("OPENAI_API_KEY"),
                    }
                },
            )

            print(f"✓ Loaded {kb3.count()} chunks")
            print(f"  Model: {kb3.embedder.model_name}")
            print(f"  Remote API: {kb3.embedder.use_remote}")
            print(f"  Base URL: {kb3.embedder.base_url}")

            answer = kb3.ask("What is an example of a local model?")
            print(f"\nQ: What is an example of a local model?")
            print(f"A: {answer.text}\n")
        else:
            print("⚠️  Skipped: OPENAI_API_KEY not set")
            print("   Set it to test remote embeddings:")
            print('   export OPENAI_API_KEY="sk-..."')
            print()

        print("=" * 60)
        print("Example 4: Custom Embedding API")
        print("=" * 60)

        print("You can use any OpenAI-compatible embedding API:")
        print(
            """
kb = Ragi("./docs", config={
    "embedding": {
        "model": "custom-model-name",
        "base_url": "http://localhost:8080/v1",
        "api_key": "your-api-key"
    }
})
"""
        )

        print("=" * 60)
        print("\n✅ All examples completed!")
        print(
            "\nKey Takeaways:"
            "\n1. Local embeddings (sentence-transformers) are free and work offline"
            "\n2. Remote APIs require base_url and api_key configuration"
            "\n3. Mix and match: use local embeddings with cloud LLM or vice versa"
            "\n4. Default is local embeddings - no setup required!"
        )

    finally:
        # Cleanup
        os.unlink(doc_path)
        if "kb1" in locals():
            kb1.clear()
        if "kb2" in locals():
            kb2.clear()
        if "kb3" in locals():
            kb3.clear()


if __name__ == "__main__":
    main()
