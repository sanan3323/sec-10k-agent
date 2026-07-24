"""Tests for BM25 lexical retrieval: pure ranking, plus BM25Index with a fake engine."""

from __future__ import annotations

from sec_10k_agent.retrieval.bm25 import BM25Index, rank_documents, tokenize


def test_tokenize_lowercases_and_splits_on_non_alnum() -> None:
    assert tokenize("Apple's Supply-Chain Risk!") == ["apple", "s", "supply", "chain", "risk"]


def test_tokenize_empty_string() -> None:
    assert tokenize("") == []


def test_rank_documents_prefers_exact_term_matches() -> None:
    docs = [
        "The company discusses unrelated topics like marketing strategy.",
        "TSMC is the primary foundry partner for semiconductor fabrication.",
        "General risk factors apply broadly across the industry.",
    ]
    ranked = rank_documents("TSMC foundry fabrication", docs, k=2)
    assert ranked[0][0] == 1  # the TSMC doc ranks first
    assert len(ranked) == 2


def test_rank_documents_respects_k() -> None:
    docs = ["risk factor one", "risk factor two", "risk factor three"]
    ranked = rank_documents("risk factor", docs, k=1)
    assert len(ranked) == 1


def test_rank_documents_empty_corpus() -> None:
    assert rank_documents("anything", [], k=5) == []


# --- BM25Index with a fake engine (no real DB) -------------------------------


class _FakeRow:
    def __init__(self, mapping: dict[str, object]) -> None:
        self._mapping = mapping


class _FakeConn:
    def __init__(self, rows: list[_FakeRow]) -> None:
        self._rows = rows
        self.last_params: dict[str, object] = {}

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, statement: object, params: dict[str, object]) -> _FakeConn:
        self.last_params = params
        return self

    def fetchall(self) -> list[_FakeRow]:
        return self._rows


class _FakeEngine:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = [_FakeRow(r) for r in rows]
        self.connect_calls = 0

    def connect(self) -> _FakeConn:
        self.connect_calls += 1
        return _FakeConn(self._rows)


def _row(chunk_id: str, text_: str, ticker: str = "AAPL") -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "ticker": ticker,
        "fiscal_year": 2024,
        "section": "Item 1A",
        "section_title": "Risk Factors",
        "text": text_,
        "prev_chunk_id": None,
        "next_chunk_id": None,
    }


def test_bm25_index_search_returns_chunks_ranked() -> None:
    rows = [
        _row("c1", "Apple relies on TSMC for chip fabrication."),
        _row("c2", "Unrelated marketing and brand strategy discussion."),
    ]
    engine = _FakeEngine(rows)
    index = BM25Index(engine=engine)  # type: ignore[arg-type]

    results = index.search("TSMC fabrication", ticker="AAPL", k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "c1"  # exact-term match ranks first
    assert all(0.0 <= r.distance <= 1.0 for r in results)


def test_bm25_index_caches_fetch_per_filter() -> None:
    engine = _FakeEngine([_row("c1", "supply chain risk")])
    index = BM25Index(engine=engine)  # type: ignore[arg-type]

    index.search("supply chain", ticker="AAPL", k=1)
    index.search("supply chain", ticker="AAPL", k=1)

    assert engine.connect_calls == 1  # second call served from cache


def test_bm25_index_empty_corpus_returns_empty() -> None:
    engine = _FakeEngine([])
    index = BM25Index(engine=engine)  # type: ignore[arg-type]
    assert index.search("anything", ticker="AAPL") == []
