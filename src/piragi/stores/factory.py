"""Factory for creating vector stores from URIs and configs."""

from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse, parse_qs

from .protocol import VectorStoreProtocol
from .lance import LanceStore


def parse_store_uri(uri: str) -> Dict[str, Any]:
    """
    Parse a store URI into connection parameters.

    Supported formats:
        - Local path: ".piragi", "./data/vectors"
        - S3: "s3://bucket/path"
        - PostgreSQL: "postgres://user:pass@host:port/db"
        - Pinecone: "pinecone://index-name?api_key=...&environment=..."

    Args:
        uri: Store URI string

    Returns:
        Dict with 'type' and connection parameters
    """
    # Local path (no scheme or file://)
    if not "://" in uri or uri.startswith("file://"):
        path = uri.replace("file://", "")
        return {"type": "lance", "uri": path}

    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()

    if scheme == "s3":
        # S3-backed LanceDB
        return {"type": "lance", "uri": uri}

    elif scheme in ("postgres", "postgresql"):
        # PostgreSQL with pgvector
        return {
            "type": "postgres",
            "connection_string": uri,
        }

    elif scheme == "pinecone":
        # Pinecone: pinecone://index-name?api_key=...
        query_params = parse_qs(parsed.query)
        return {
            "type": "pinecone",
            "index_name": parsed.netloc or parsed.path.strip("/"),
            "api_key": query_params.get("api_key", [""])[0],
            "environment": query_params.get("environment", [None])[0],
            "namespace": query_params.get("namespace", ["default"])[0],
        }

    else:
        # Unknown scheme, treat as local path
        return {"type": "lance", "uri": uri}


def create_store(
    store: Union[str, Dict[str, Any], VectorStoreProtocol, None] = None,
    persist_dir: str = ".piragi",
    embedding_model: str = "all-mpnet-base-v2",
    vector_dimension: Optional[int] = None,
) -> VectorStoreProtocol:
    """
    Create a vector store from various input types.

    Args:
        store: One of:
            - None: Use default LanceStore with persist_dir
            - str: URI to parse (local path, s3://, postgres://, pinecone://)
            - dict: Store config with 'type' and parameters
            - VectorStoreProtocol: Use directly
        persist_dir: Default directory for local storage (if store is None)
        embedding_model: Embedding model for dimension inference
        vector_dimension: Explicit vector dimension

    Returns:
        VectorStoreProtocol implementation

    Examples:
        >>> # Default local store
        >>> store = create_store()
        >>>
        >>> # From URI
        >>> store = create_store("s3://my-bucket/indices")
        >>> store = create_store("postgres://user:pass@localhost/db")
        >>>
        >>> # From config dict
        >>> store = create_store({
        ...     "type": "pinecone",
        ...     "api_key": "...",
        ...     "index_name": "my-index"
        ... })
        >>>
        >>> # Pass through existing store
        >>> my_store = MyCustomStore()
        >>> store = create_store(my_store)  # Returns my_store unchanged
    """
    # Pass through if already a store
    if store is not None and isinstance(store, VectorStoreProtocol):
        return store

    # Parse URI string
    if isinstance(store, str):
        config = parse_store_uri(store)
    elif isinstance(store, dict):
        config = store
    elif store is None:
        config = {"type": "lance", "uri": persist_dir}
    else:
        raise ValueError(f"Invalid store type: {type(store)}")

    store_type = config.get("type", "lance")

    if store_type == "lance":
        return LanceStore(
            uri=config.get("uri", persist_dir),
            embedding_model=embedding_model,
            vector_dimension=vector_dimension,
        )

    elif store_type == "postgres":
        from .postgres import PostgresStore

        return PostgresStore(
            connection_string=config.get("connection_string"),
            host=config.get("host", "localhost"),
            port=config.get("port", 5432),
            database=config.get("database", "piragi"),
            user=config.get("user", "postgres"),
            password=config.get("password", ""),
            table_name=config.get("table_name", "chunks"),
            vector_dimension=vector_dimension or 768,
        )

    elif store_type == "pinecone":
        from .pinecone import PineconeStore

        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("Pinecone requires api_key")

        return PineconeStore(
            api_key=api_key,
            index_name=config.get("index_name", "piragi"),
            environment=config.get("environment"),
            namespace=config.get("namespace", "default"),
            vector_dimension=vector_dimension or 768,
        )

    else:
        raise ValueError(f"Unknown store type: {store_type}")
