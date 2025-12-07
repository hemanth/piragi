"""PostgreSQL vector store using pgvector."""

from typing import Any, Dict, List, Optional
import json

from ..types import Chunk, Citation


class PostgresStore:
    """
    PostgreSQL vector store using pgvector extension.

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
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        host: str = "localhost",
        port: int = 5432,
        database: str = "piragi",
        user: str = "postgres",
        password: str = "",
        table_name: str = "chunks",
        vector_dimension: int = 768,
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
            vector_dimension: Dimension of embedding vectors
        """
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
        except ImportError:
            raise ImportError(
                "PostgresStore requires psycopg2 and pgvector. "
                "Install with: pip install piragi[postgres]"
            )

        self.table_name = table_name
        self.vector_dimension = vector_dimension
        self._chunk_texts: List[str] = []

        # Connect
        if connection_string:
            self.conn = psycopg2.connect(connection_string)
        else:
            self.conn = psycopg2.connect(
                host=host,
                port=port,
                dbname=database,
                user=user,
                password=password,
            )

        # Register pgvector
        register_vector(self.conn)

        # Initialize schema
        self._init_schema()

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

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_chunk_length: int = 100,
    ) -> List[Citation]:
        """Search for similar chunks using cosine similarity."""
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

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source."""
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

    def count(self) -> int:
        """Return the number of chunks in the store."""
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            return cur.fetchone()[0]

    def clear(self) -> None:
        """Clear all data from the store."""
        with self.conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {self.table_name}")
            self.conn.commit()
        self._chunk_texts = []

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
