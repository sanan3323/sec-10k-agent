"""Unit tests for the retrieval query layer.

These run everywhere — no database and no embedding model. The pure functions
(`build_search_sql`, `to_pgvector_literal`) and the row-mapping in
`Retriever.search` are exercised with a fake engine and a fake embedder. The
live end-to-end path is covered by tests/test_retrieval_db.py.
"""

from __future__ import annotations

from sec_10k_agent.retrieval import (
    RetrievedChunk,
    Retriever,
    build_search_sql,
    to_pgvector_literal,
)


def test_to_pgvector_literal_formats_bracketed_csv() -> None:
    assert to_pgvector_literal([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"


def test_to_pgvector_literal_coerces_to_float() -> None:
    # ints (or numpy scalars) become plain floats in the literal.
    assert to_pgvector_literal([1, 2]) == "[1.0,2.0]"


def test_build_search_sql_no_filters_has_no_where() -> None:
    sql, params = build_search_sql()
    assert "WHERE" not in sql
    assert params == {}
    assert "ORDER BY distance ASC" in sql
    assert "LIMIT :k" in sql
    assert "embedding <=> CAST(:q AS vector)" in sql


def test_build_search_sql_all_filters() -> None:
    sql, params = build_search_sql(ticker="AAPL", fiscal_year=2024, section="Item 1A")
    assert params == {"ticker": "AAPL", "fiscal_year": 2024, "section": "Item 1A"}
    assert "WHERE ticker = :ticker AND fiscal_year = :fiscal_year AND section = :section" in sql


def test_build_search_sql_partial_filters_only_include_given() -> None:
    sql, params = build_search_sql(ticker="JPM")
    assert params == {"ticker": "JPM"}
    assert "ticker = :ticker" in sql
    # Only the ticker filter is bound; the other filter params never appear.
    # (The bare column names still show up in the SELECT list, so we check the
    # bind-parameter placeholders rather than the words.)
    assert ":fiscal_year" not in sql
    assert ":section" not in sql


def test_retrieved_chunk_score_is_inverse_distance() -> None:
    chunk = RetrievedChunk(chunk_id="c1", text="body", distance=0.25)
    assert chunk.score == 0.75


def test_retrieved_chunk_citation() -> None:
    chunk = RetrievedChunk(
        chunk_id="c1", ticker="AAPL", fiscal_year=2024, section="Item 1A", text="body", distance=0.1
    )
    assert chunk.citation() == "AAPL FY2024 Item 1A"


def test_retrieved_chunk_citation_falls_back_to_id() -> None:
    chunk = RetrievedChunk(chunk_id="c1", text="body", distance=0.1)
    assert chunk.citation() == "c1"


# --- Retriever.search with fakes (no DB, no model) ---------------------------


class _FakeRow:
    """Mimics a SQLAlchemy Row: exposes a dict-like `_mapping`."""

    def __init__(self, mapping: dict[str, object]) -> None:
        self._mapping = mapping


class _FakeConn:
    def __init__(self, engine: _FakeEngine) -> None:
        self._engine = engine

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, statement: object, params: dict[str, object]) -> _FakeConn:
        self._engine.last_params = params
        return self

    def fetchall(self) -> list[_FakeRow]:
        return self._engine.rows


class _FakeEngine:
    def __init__(self, rows: list[_FakeRow]) -> None:
        self.rows = rows
        self.last_params: dict[str, object] = {}

    def connect(self) -> _FakeConn:
        return _FakeConn(self)


def test_search_maps_rows_and_passes_query_params() -> None:
    rows = [
        _FakeRow(
            {
                "chunk_id": "AAPL__Item 1A__0001",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "section": "Item 1A",
                "section_title": "Risk Factors",
                "text": "Supply chain concentration in Asia...",
                "prev_chunk_id": None,
                "next_chunk_id": "AAPL__Item 1A__0002",
                "distance": 0.12,
            }
        )
    ]
    engine = _FakeEngine(rows)
    retriever = Retriever(engine=engine, embedder=lambda q: [0.0, 1.0, 0.0])

    results = retriever.search("supply chain risk", ticker="AAPL", section="Item 1A", k=3)

    assert len(results) == 1
    got = results[0]
    assert isinstance(got, RetrievedChunk)
    assert got.ticker == "AAPL"
    assert got.section_title == "Risk Factors"
    assert got.distance == 0.12
    # The embedded query is passed as a pgvector literal and k/filters are bound.
    assert engine.last_params["q"] == "[0.0,1.0,0.0]"
    assert engine.last_params["k"] == 3
    assert engine.last_params["ticker"] == "AAPL"
    assert engine.last_params["section"] == "Item 1A"


def test_search_without_filters_binds_only_query_and_limit() -> None:
    engine = _FakeEngine([])
    retriever = Retriever(engine=engine, embedder=lambda q: [1.0])
    retriever.search("anything")
    assert set(engine.last_params) == {"q", "k"}
