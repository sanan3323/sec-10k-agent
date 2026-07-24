"""Dense retrieval over the pgvector-indexed 10-K chunks.

This is the production form of what `notebooks/test_retrieval.py` proved out:
embed the query with the same BGE model the chunks were embedded with, then let
Postgres rank by cosine distance (`<=>`) using the HNSW index, optionally
pre-filtered on ticker / fiscal year / section via the composite btree.

The query embedding uses `fastembed` (ONNX, no torch) so it runs anywhere the
core deps are installed. Cosine distance is magnitude-invariant, so it does not
matter that the stored vectors and the query vector may be normalized
differently — only direction is compared.

Design notes:
- `build_search_sql` and `to_pgvector_literal` are pure functions so the query
  construction is unit-testable without a database or the ~1.3 GB embedding
  model. `Retriever.search` is the thin DB-touching wrapper around them.
- The embedding model and SQLAlchemy engine are both injectable, so tests can
  swap in fakes and the RAG layer can share one engine.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

from sqlalchemy import create_engine, text

from sec_10k_agent.config import get_settings
from sec_10k_agent.retrieval.models import RetrievedChunk

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_K = 5

# Columns selected for every search, in the order RetrievedChunk expects them.
_SELECT_COLUMNS = (
    "chunk_id",
    "ticker",
    "fiscal_year",
    "section",
    "section_title",
    "text",
    "prev_chunk_id",
    "next_chunk_id",
)


def to_pgvector_literal(vec: Sequence[float]) -> str:
    """Render a vector as the string literal pgvector accepts, e.g. '[0.1,0.2]'."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def build_search_sql(
    ticker: str | None = None,
    fiscal_year: int | None = None,
    section: str | None = None,
) -> tuple[str, dict[str, object]]:
    """Build the parameterized similarity-search SQL and its filter params.

    Returns (sql, params). The caller adds the `q` (vector literal) and `k`
    (limit) bind params. Filters are optional and combined with AND; omitting
    all of them searches the whole corpus.
    """
    conditions: list[str] = []
    params: dict[str, object] = {}
    if ticker is not None:
        conditions.append("ticker = :ticker")
        params["ticker"] = ticker
    if fiscal_year is not None:
        conditions.append("fiscal_year = :fiscal_year")
        params["fiscal_year"] = fiscal_year
    if section is not None:
        conditions.append("section = :section")
        params["section"] = section

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    columns = ", ".join(_SELECT_COLUMNS)
    sql = f"""
        SELECT {columns},
               embedding <=> CAST(:q AS vector) AS distance
        FROM text_chunks
        {where}
        ORDER BY distance ASC
        LIMIT :k
    """
    return sql, params


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str):  # type: ignore[no-untyped-def]
    """Load (and cache) a fastembed model. Import is local so the package
    imports cheaply and installs without the embedding stack present."""
    from fastembed import TextEmbedding

    return TextEmbedding(model_name)


def embed_query(query: str, model_name: str = DEFAULT_MODEL) -> list[float]:
    """Embed a single query string into a 1024-dim BGE vector."""
    model = _load_embedding_model(model_name)
    vec = next(iter(model.embed([query])))
    return [float(x) for x in vec]


class Retriever:
    """Dense retriever over `text_chunks`.

    Args:
        engine: A SQLAlchemy Engine. Defaults to one built from
            `settings.postgres_dsn`.
        embedder: A callable mapping a query string to a vector. Defaults to the
            fastembed BGE-large embedder. Injectable for testing.
        model_name: BGE model id, used only by the default embedder.
    """

    def __init__(
        self,
        engine: Engine | None = None,
        embedder: Callable[[str], Sequence[float]] | None = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self._engine = engine or create_engine(get_settings().postgres_dsn)
        self._model_name = model_name
        self._embedder = embedder or (lambda q: embed_query(q, model_name))

    def search(
        self,
        query: str,
        *,
        ticker: str | None = None,
        fiscal_year: int | None = None,
        section: str | None = None,
        k: int = DEFAULT_K,
    ) -> list[RetrievedChunk]:
        """Return the `k` chunks most similar to `query`, nearest first.

        Optional `ticker` / `fiscal_year` / `section` pre-filter the corpus
        before ranking (uses the composite btree index).
        """
        vec = self._embedder(query)
        sql, params = build_search_sql(ticker, fiscal_year, section)
        params["q"] = to_pgvector_literal(vec)
        params["k"] = k
        with self._engine.connect() as conn:
            rows = conn.execute(text(sql), params).fetchall()
        return [RetrievedChunk(**dict(row._mapping)) for row in rows]
