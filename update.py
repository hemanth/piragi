import re

with open('/Users/lika/labs/piragi/src/piragi/hybrid_search.py', 'r') as f:
    content = f.read()

# 1. Update imports
content = content.replace(
    "import math\nfrom collections import defaultdict\nfrom typing import Any, Dict, List, Optional, Tuple",
    "from collections import defaultdict\nfrom typing import Any, Dict, List, Optional, Tuple\n\nfrom rank_bm25 import BM25Okapi"
)

# 2. Remove BM25 class
bm25_class_pattern = re.compile(r'class BM25:.*?class HybridSearcher:', re.DOTALL)
content = bm25_class_pattern.sub('class HybridSearcher:', content)

# 3. Update HybridSearcher._bm25 typing
content = content.replace(
    "self._bm25: Optional[BM25] = None",
    "self._bm25: Optional[BM25Okapi] = None"
)

# 4. Add _tokenize to HybridSearcher and update index_chunks
old_index_chunks = """    def index_chunks(self, chunks: List[str]) -> None:
        \"\"\"
        Index chunks for BM25 search.

        Args:
            chunks: List of chunk text strings
        \"\"\"
        self._chunk_texts = chunks
        self._chunk_to_idx = {text[:200]: i for i, text in enumerate(chunks)}
        self._bm25 = BM25()
        self._bm25.fit(chunks)
        logger.info(f"Indexed {len(chunks)} chunks for BM25 search")"""

new_index_chunks = """    def _tokenize(self, text: str) -> List[str]:
        \"\"\"Simple tokenization with lowercasing and basic cleanup.\"\"\"
        import re
        # Remove punctuation and split
        tokens = re.findall(r'\\b\\w+\\b', text.lower())
        # Filter very short tokens
        return [t for t in tokens if len(t) > 1]

    def index_chunks(self, chunks: List[str]) -> None:
        \"\"\"
        Index chunks for BM25 search.

        Args:
            chunks: List of chunk text strings
        \"\"\"
        self._chunk_texts = chunks
        self._chunk_to_idx = {text[:200]: i for i, text in enumerate(chunks)}
        
        tokenized_corpus = [self._tokenize(chunk) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Indexed {len(chunks)} chunks for BM25 search")"""

content = content.replace(old_index_chunks, new_index_chunks)

# 5. Update search function
old_search_bm25 = """        # Get BM25 scores for all indexed chunks
        bm25_scores = self._bm25.score(query)"""

new_search_bm25 = """        # Get BM25 scores for all indexed chunks
        query_tokens = self._tokenize(query)
        bm25_scores = self._bm25.get_scores(query_tokens)"""

content = content.replace(old_search_bm25, new_search_bm25)

with open('/Users/lika/labs/piragi/src/piragi/hybrid_search.py', 'w') as f:
    f.write(content)

with open('/Users/lika/labs/piragi/pyproject.toml', 'r') as f:
    toml = f.read()

toml = toml.replace(
    '"pysbd>=0.3.4",',
    '"pysbd>=0.3.4",\n    "rank-bm25>=0.2.2",'
)

with open('/Users/lika/labs/piragi/pyproject.toml', 'w') as f:
    f.write(toml)
