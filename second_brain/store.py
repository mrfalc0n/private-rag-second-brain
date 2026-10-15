"""Embeddings, the pgvector store, the record manager and the sync_state table."""

from datetime import datetime, timezone
from functools import lru_cache

from langchain_classic.indexes import SQLRecordManager
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text

from second_brain.config import database_url, settings


class PrefixedEmbeddings(Embeddings):
    """nomic-embed-text expects "search_document: " on documents and "search_query: " on queries."""

    def __init__(self, inner: Embeddings, document_prefix: str, query_prefix: str):
        self.inner = inner
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(self.query_prefix + text)


@lru_cache
def engine():
    return create_engine(database_url())


@lru_cache
def embeddings() -> Embeddings:
    cfg = settings()["embedding"]
    inner = OllamaEmbeddings(model=cfg["model"], base_url=cfg["base_url"])
    return PrefixedEmbeddings(inner, cfg["document_prefix"], cfg["query_prefix"])


def collection_name() -> str:
    # One collection per embedding model: vectors from different models are not comparable.
    return f"{settings()['instance_name']}__{settings()['embedding']['model']}"


@lru_cache
def vector_store() -> PGVector:
    return PGVector(
        embeddings=embeddings(),
        connection=engine(),
        collection_name=collection_name(),
        embedding_length=settings()["embedding"]["dimensions"],
        use_jsonb=True,
        create_extension=False,
    )


@lru_cache
def record_manager() -> SQLRecordManager:
    rm = SQLRecordManager(f"pgvector/{collection_name()}", engine=engine())
    rm.create_schema()
    return rm


SYNC_STATE_DDL = """
CREATE TABLE IF NOT EXISTS sync_state (
  source          TEXT PRIMARY KEY,
  cursor          TEXT,
  last_success_at TIMESTAMPTZ,
  last_error      TEXT,
  last_error_at   TIMESTAMPTZ,
  doc_count       INT
)
"""


# Full-text search: the parsed words of each chunk, stored once at write time and indexed, so keyword
# search doesn't re-parse thousands of chunks per query. Postgres keeps the column in sync by itself.
FTS_DDL = [
    "ALTER TABLE langchain_pg_embedding ADD COLUMN IF NOT EXISTS fts tsvector "
    "GENERATED ALWAYS AS (to_tsvector('english', coalesce(document, ''))) STORED",
    "DROP INDEX IF EXISTS langchain_pg_embedding_fts",  # old index on the expression
    "CREATE INDEX IF NOT EXISTS langchain_pg_embedding_fts_col ON langchain_pg_embedding USING gin (fts)",
]


def ensure_schema() -> None:
    with engine().begin() as conn:
        conn.execute(text(SYNC_STATE_DDL))
    vector_store()
    record_manager()
    with engine().begin() as conn:
        for ddl in FTS_DDL:
            conn.execute(text(ddl))


def get_cursor(source: str) -> str | None:
    with engine().begin() as conn:
        row = conn.execute(text("SELECT cursor FROM sync_state WHERE source = :s"), {"s": source}).first()
    return row[0] if row else None


def save_success(source: str, cursor: str, doc_count: int) -> None:
    with engine().begin() as conn:
        conn.execute(text("""
            INSERT INTO sync_state (source, cursor, last_success_at, doc_count)
            VALUES (:s, :c, :t, :n)
            ON CONFLICT (source) DO UPDATE
              SET cursor = :c, last_success_at = :t, doc_count = :n, last_error = NULL, last_error_at = NULL
        """), {"s": source, "c": cursor, "t": datetime.now(timezone.utc), "n": doc_count})


def save_error(source: str, error: str) -> None:
    with engine().begin() as conn:
        conn.execute(text("""
            INSERT INTO sync_state (source, last_error, last_error_at)
            VALUES (:s, :e, :t)
            ON CONFLICT (source) DO UPDATE SET last_error = :e, last_error_at = :t
        """), {"s": source, "e": error[:2000], "t": datetime.now(timezone.utc)})


def status() -> list[dict]:
    with engine().begin() as conn:
        rows = conn.execute(text("SELECT * FROM sync_state ORDER BY source")).mappings().all()
    return [dict(r) for r in rows]
