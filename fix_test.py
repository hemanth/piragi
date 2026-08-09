import re

with open('/Users/lika/labs/piragi/tests/test_hybrid_search.py', 'r') as f:
    content = f.read()

# Remove BM25 import
content = content.replace(
    "from piragi.hybrid_search import BM25, HybridSearcher, create_hybrid_searcher",
    "from piragi.hybrid_search import HybridSearcher, create_hybrid_searcher"
)

# Remove TestBM25 class
test_bm25_pattern = re.compile(r'class TestBM25:.*?class TestHybridSearcher:', re.DOTALL)
content = test_bm25_pattern.sub('class TestHybridSearcher:', content)

with open('/Users/lika/labs/piragi/tests/test_hybrid_search.py', 'w') as f:
    f.write(content)
