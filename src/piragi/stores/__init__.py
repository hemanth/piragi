"""
Vector store adapters for piragi.

Supports multiple backends:
- LanceDB (default, local)
- S3 (via LanceDB S3 support)
- PostgreSQL (pgvector)
- Pinecone

Example:
    >>> from piragi import Ragi
    >>> from piragi.stores import PineconeStore, PostgresStore
    >>>
    >>> # Use Pinecone
    >>> kb = Ragi("./docs", store=PineconeStore(api_key="...", index="my-index"))
    >>>
    >>> # Use Postgres with pgvector
    >>> kb = Ragi("./docs", store=PostgresStore(connection_string="postgres://..."))
    >>>
    >>> # Use S3-backed LanceDB
    >>> kb = Ragi("./docs", store="s3://my-bucket/indices")
"""

from .protocol import VectorStoreProtocol
from .lance import LanceStore
from .postgres import PostgresStore
from .pinecone import PineconeStore
try:
    from .supabase import SupabaseStore
except ImportError:
    SupabaseStore = None  # Optional dependency
from .factory import create_store, parse_store_uri

__all__ = [
    "VectorStoreProtocol",
    "LanceStore",
    "PostgresStore",
    "PineconeStore",
    "SupabaseStore",
    "create_store",
    "parse_store_uri",
]
