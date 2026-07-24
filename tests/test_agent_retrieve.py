"""Tests for the retrieve node's mode dispatch, with faked retriever/tool."""

from __future__ import annotations

from sec_10k_agent.agent.retrieve import retrieve
from sec_10k_agent.agent.state import SubQuery, XBRLFact
from sec_10k_agent.retrieval.models import RetrievedChunk


class _FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.last_call: dict[str, object] = {}

    def search(self, query: str, **kwargs: object) -> list[RetrievedChunk]:
        self.last_call = {"query": query, **kwargs}
        return self._chunks


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(chunk_id="c1", ticker="AAPL", text="body", distance=0.1)


def _fact() -> XBRLFact:
    return XBRLFact(ticker="AAPL", fiscal_year=2024, concept="us-gaap:Revenues", value=1.0)


def test_semantic_subquery_calls_retriever() -> None:
    retriever = _FakeRetriever([_chunk()])
    sq = SubQuery(question="q", ticker="AAPL", fiscal_year=2024, section="Item 1A", mode="semantic")
    results = retrieve([sq], retriever, xbrl_lookup=lambda **kw: [])  # type: ignore[arg-type]
    assert len(results) == 1
    assert results[0].chunks == [_chunk()]
    assert retriever.last_call == {
        "query": "q",
        "ticker": "AAPL",
        "fiscal_year": 2024,
        "section": "Item 1A",
        "k": 5,
    }


def test_structured_xbrl_subquery_calls_tool_with_concept() -> None:
    calls = []

    def fake_lookup(**kwargs: object) -> list[XBRLFact]:
        calls.append(kwargs)
        return [_fact()]

    sq = SubQuery(
        question="q",
        ticker="AAPL",
        fiscal_year=2024,
        concept="total revenue",
        mode="structured_xbrl",
    )
    results = retrieve([sq], _FakeRetriever([]), xbrl_lookup=fake_lookup)  # type: ignore[arg-type]
    assert results[0].facts == [_fact()]
    assert calls == [{"ticker": "AAPL", "fiscal_year": 2024, "concept": "total revenue"}]


def test_structured_xbrl_falls_back_to_question_when_no_concept() -> None:
    calls = []
    sq = SubQuery(
        question="what is revenue?", ticker="AAPL", fiscal_year=2024, mode="structured_xbrl"
    )
    retrieve([sq], _FakeRetriever([]), xbrl_lookup=lambda **kw: calls.append(kw) or [])  # type: ignore[arg-type]
    assert calls[0]["concept"] == "what is revenue?"


def test_structured_xbrl_without_ticker_or_year_skips_lookup() -> None:
    called = False

    def fake_lookup(**kwargs: object) -> list[XBRLFact]:
        nonlocal called
        called = True
        return []

    sq = SubQuery(question="q", mode="structured_xbrl")  # no ticker/fiscal_year
    results = retrieve([sq], _FakeRetriever([]), xbrl_lookup=fake_lookup)  # type: ignore[arg-type]
    assert called is False
    assert results[0].facts == []


def test_retrieve_preserves_subquery_order() -> None:
    sqs = [
        SubQuery(question="a", mode="semantic"),
        SubQuery(question="b", ticker="AAPL", fiscal_year=2024, mode="structured_xbrl"),
    ]
    results = retrieve(sqs, _FakeRetriever([]), xbrl_lookup=lambda **kw: [])  # type: ignore[arg-type]
    assert [r.subquery.question for r in results] == ["a", "b"]
