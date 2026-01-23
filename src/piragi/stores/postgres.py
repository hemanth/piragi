"""PostgreSQL vector store using pgvector."""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, TYPE_CHECKING
import json

from ..types import Chunk, Citation

if TYPE_CHECKING:
    from ..embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PostgresStore:
    """
    PostgreSQL vector store using pgvector extension.

    Includes automatic connection resilience with configurable retry logic
    for handling transient failures and connection issues.

    Requires:
        pip install psycopg2-binary pgvector

    Examples:
        >>> store = PostgresStore(
        ...     connection_string="postgres://user:pass@localhost/db",
        ...     table_name="embeddings"
        ... )
        >>>
        >>> # Or with individual params
        >>> store = PostgresStore(
        ...     host="localhost",
        ...     database="mydb",
        ...     user="user",
        ...     password="pass"
        ... )
        >>>
        >>> # With custom retry settings
        >>> store = PostgresStore(
        ...     connection_string="postgres://...",
        ...     max_retries=5,
        ...     retry_delay=1.0
        ... )
        >>>
        >>> # Auto-infer dimensions from embedder
        >>> from piragi import EmbeddingGenerator
        >>> embedder = EmbeddingGenerator(model="all-MiniLM-L6-v2")
        >>> store = PostgresStore(
        ...     connection_string="postgres://...",
        ...     embedder=embedder  # Automatically detects 384 dimensions
        ... )
    """

    # Error messages that indicate a retriable error
    RETRIABLE_ERRORS = (
        "transaction is aborted",
        "server closed the connection",
        "connection already closed",
        "connection refused",
        "could not connect to server",
        "connection reset by peer",
        "SSL connection has been closed unexpectedly",
        "terminating connection due to administrator command",
    )

    def __init__(
        self,
        connection_string: Optional[str] = None,
        host: str = "localhost",
        port: int = 5432,
        database: str = "piragi",
        user: str = "postgres",
        password: str = "",
        table_name: str = "chunks",
        vector_dimension: Optional[int] = None,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        embedder: Optional["EmbeddingGenerator"] = None,
    ) -> None:
        """
        Initialize PostgreSQL store.

        Args:
            connection_string: Full connection string (overrides other params)
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            table_name: Table name for chunks
            vector_dimension: Dimension of embedding vectors. If not provided and
                embedder is given, dimensions will be auto-inferred. If neither
                is provided, defaults to 768.
            max_retries: Maximum number of retry attempts for transient failures (default: 3)
            retry_delay: Base delay between retries in seconds, doubles with each attempt (default: 0.5)
            embedder: Optional EmbeddingGenerator instance for auto-inferring vector dimensions.
                If provided and vector_dimension is not set, the embedder will be used to
                determine the correct dimension automatically.
        """
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
            self._psycopg2 = psycopg2
            self._register_vector = register_vector
        except ImportError:
            raise ImportError(
                "PostgresStore requires psycopg2 and pgvector. "
                "Install with: pip install piragi[postgres]"
            )

        self.table_name = table_name
        self._chunk_texts: List[str] = []

        # Determine vector dimension
        if vector_dimension is not None:
            self.vector_dimension = vector_dimension
        elif embedder is not None:
            # Auto-infer dimensions from embedder
            logger.info("Auto-inferring vector dimensions from embedder...")
            self.vector_dimension = embedder.get_dimensions()
            logger.info(f"Detected vector dimension: {self.vector_dimension}")
        else:
            # Default fallback
            self.vector_dimension = 768

        # Retry configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Store connection params for reconnection
        self._connection_string = connection_string
        self._connection_params = {
            "host": host,
            "port": port,
            "dbname": database,
            "user": user,
            "password": password,
        }

        # Connect
        self._connect()

        # Initialize schema
        self._init_schema()

    def _connect(self) -> None:
        """Establish database connection."""
        if self._connection_string:
            self.conn = self._psycopg2.connect(self._connection_string)
        else:
            self.conn = self._psycopg2.connect(**self._connection_params)

        # Register pgvector
        self._register_vector(self.conn)

    def _reconnect(self) -> None:
        """Close existing connection and establish a new one."""
        try:
            if hasattr(self, "conn") and self.conn:
                self.conn.close()
        except Exception:
            pass  # Ignore errors when closing broken connection

        self._connect()
        logger.info("Successfully reconnected to PostgreSQL")

    def _is_retriable_error(self, error: Exception) -> bool:
        """Check if an error is retriable."""
        error_msg = str(error).lower()
        return any(msg in error_msg for msg in self.RETRIABLE_ERRORS)

    def _execute_with_retry(self, operation: Callable[[], T]) -> T:
        """
        Execute an operation with automatic retry on transient failures.

        Args:
            operation: Callable that performs the database operation

        Returns:
            Result of the operation

        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except Exception as e:
                last_exception = e

                if not self._is_retriable_error(e):
                    # Non-retriable error, raise immediately
                    raise

                if attempt < self.max_retries:
                    # Calculate delay with exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Database operation failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

                    # Try to reconnect
                    try:
                        self._reconnect()
                    except Exception as reconnect_error:
                        logger.warning(f"Reconnection failed: {reconnect_error}")

        # All retries exhausted
        raise last_exception

    def _init_schema(self) -> None:
        """Create table and indexes if they don't exist."""
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    chunk_index INTEGER,
                    metadata JSONB,
                    embedding vector({self.vector_dimension})
                )
            """)

            # Create index for vector search
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                ON {self.table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)

            # Create index for source filtering
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_source_idx
                ON {self.table_name} (source)
            """)

            self.conn.commit()

        # Load existing chunk texts
        self._load_chunk_texts()

    def _load_chunk_texts(self) -> None:
        """Load all chunk texts for hybrid search."""
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT text FROM {self.table_name}")
            self._chunk_texts = [row[0] for row in cur.fetchall()]

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks with embeddings to the store."""
        if not chunks:
            return

        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError("All chunks must have embeddings")

        def _do_add():
            with self.conn.cursor() as cur:
                for chunk in chunks:
                    cur.execute(
                        f"""
                        INSERT INTO {self.table_name} (text, source, chunk_index, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            chunk.text,
                            chunk.source,
                            chunk.chunk_index,
                            json.dumps(chunk.metadata),
                            chunk.embedding,
                        ),
                    )
                    self._chunk_texts.append(chunk.text)

                self.conn.commit()

        self._execute_with_retry(_do_add)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_chunk_length: int = 100,
    ) -> List[Citation]:
        """Search for similar chunks using cosine similarity."""
        def _do_search() -> List[Citation]:
            with self.conn.cursor() as cur:
                # Build query
                where_clauses = [f"LENGTH(text) >= {min_chunk_length}"]

                if filters:
                    for key, value in filters.items():
                        where_clauses.append(f"metadata->>'{key}' = '{value}'")

                where_sql = " AND ".join(where_clauses)

                cur.execute(
                    f"""
                    SELECT text, source, metadata, 1 - (embedding <=> %s::vector) as score
                    FROM {self.table_name}
                    WHERE {where_sql}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, top_k),
                )

                citations = []
                for row in cur.fetchall():
                    citations.append(
                        Citation(
                            source=row[1],
                            chunk=row[0],
                            score=float(row[3]),
                            metadata=row[2] if row[2] else {},
                        )
                    )

                return citations

        return self._execute_with_retry(_do_search)

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source."""
        def _do_delete() -> int:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.table_name} WHERE source = %s",
                    (source,),
                )
                deleted = cur.rowcount
                self.conn.commit()

            # Reload chunk texts
            self._load_chunk_texts()

            return deleted

        return self._execute_with_retry(_do_delete)

    def count(self) -> int:
        """Return the number of chunks in the store."""
        def _do_count() -> int:
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                return cur.fetchone()[0]

        return self._execute_with_retry(_do_count)

    def clear(self) -> None:
        """Clear all data from the store."""
        def _do_clear():
            with self.conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.table_name}")
                self.conn.commit()
            self._chunk_texts = []

        self._execute_with_retry(_do_clear)

    def get_all_chunk_texts(self) -> List[str]:
        """Get all chunk texts for hybrid search."""
        return self._chunk_texts

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "conn") and self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
