"""Tests for the XBRL lookup tool, with a fake SQLAlchemy engine (no DB)."""

from __future__ import annotations

from sec_10k_agent.agent.tools import lookup_financial_metric


class _FakeRow:
    def __init__(self, mapping: dict[str, object]) -> None:
        self._mapping = mapping
        for k, v in mapping.items():
            setattr(self, k, v)


class _FakeConn:
    def __init__(self, rows: list[_FakeRow]) -> None:
        self._rows = rows
        self.last_sql: str = ""
        self.last_params: dict[str, object] = {}

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, statement: object, params: dict[str, object]) -> _FakeConn:
        self.last_sql = str(statement)
        self.last_params = params
        return self

    def fetchall(self) -> list[_FakeRow]:
        return self._rows


class _FakeEngine:
    def __init__(self, rows: list[_FakeRow]) -> None:
        self._conn = _FakeConn(rows)

    def connect(self) -> _FakeConn:
        return self._conn


def _fact_row(concept: str, value: float = 100.0, dims: dict | None = None) -> _FakeRow:
    return _FakeRow(
        {
            "ticker": "AAPL",
            "fiscal_year": 2024,
            "concept": concept,
            "value": value,
            "unit": "USD",
            "dimensions": dims or {},
        }
    )


def test_lookup_returns_typed_facts() -> None:
    engine = _FakeEngine([_fact_row("us-gaap:Revenues", value=391035000000.0)])
    facts = lookup_financial_metric("AAPL", 2024, "us-gaap:Revenues", engine=engine)  # type: ignore[arg-type]
    assert len(facts) == 1
    assert facts[0].ticker == "AAPL"
    assert facts[0].fiscal_year == 2024
    assert facts[0].value == 391035000000.0
    assert facts[0].unit == "USD"
    assert facts[0].dimensions == {}


def test_lookup_passes_exact_and_fuzzy_params() -> None:
    engine = _FakeEngine([])
    lookup_financial_metric("AAPL", 2024, "Revenues", engine=engine)  # type: ignore[arg-type]
    params = engine._conn.last_params
    assert params["ticker"] == "AAPL"
    assert params["fiscal_year"] == 2024
    assert params["concept"] == "Revenues"
    assert params["concept_like"] == "%Revenues%"
    assert "dims" not in params


def test_lookup_with_dimensions_adds_containment_filter() -> None:
    engine = _FakeEngine([])
    dims = {"srt:StatementGeographicalAxis": "country:CN"}
    lookup_financial_metric("AAPL", 2024, "us-gaap:Revenues", dimensions=dims, engine=engine)  # type: ignore[arg-type]
    params = engine._conn.last_params
    assert params["dims"] == '{"srt:StatementGeographicalAxis": "country:CN"}'
    assert "dimensions @>" in engine._conn.last_sql


def test_lookup_no_dimensions_omits_containment_clause() -> None:
    engine = _FakeEngine([])
    lookup_financial_metric("AAPL", 2024, "us-gaap:Revenues", engine=engine)  # type: ignore[arg-type]
    assert "dimensions @>" not in engine._conn.last_sql


def test_lookup_empty_result_is_not_an_error() -> None:
    engine = _FakeEngine([])
    facts = lookup_financial_metric("AAPL", 2024, "nonexistent:Concept", engine=engine)  # type: ignore[arg-type]
    assert facts == []


def test_lookup_defaults_missing_dimensions_to_empty_dict() -> None:
    row = _fact_row("us-gaap:Revenues", dims=None)
    row.dimensions = None  # simulate a NULL jsonb column
    engine = _FakeEngine([row])
    facts = lookup_financial_metric("AAPL", 2024, "us-gaap:Revenues", engine=engine)  # type: ignore[arg-type]
    assert facts[0].dimensions == {}
