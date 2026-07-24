"""Tests for the shared source-labeling helpers."""

from __future__ import annotations

from sec_10k_agent.agent.citations import (
    build_source_map,
    fact_key,
    format_source,
    format_sources_block,
)
from sec_10k_agent.agent.state import Retrieval, SubQuery, XBRLFact
from sec_10k_agent.retrieval.models import RetrievedChunk


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, ticker="AAPL", section="Item 1A", text="body", distance=0.1
    )


def _fact() -> XBRLFact:
    return XBRLFact(
        ticker="AAPL",
        fiscal_year=2024,
        concept="us-gaap:Revenues",
        value=391035000000.0,
        unit="USD",
    )


def _subquery() -> SubQuery:
    return SubQuery(question="q")


def test_fact_key_format() -> None:
    assert fact_key(_fact()) == "fact:AAPL:2024:us-gaap:Revenues"


def test_build_source_map_combines_chunks_and_facts_deduped() -> None:
    retrievals = [
        Retrieval(subquery=_subquery(), chunks=[_chunk("c1"), _chunk("c1")], facts=[_fact()]),
    ]
    sources = build_source_map(retrievals)
    assert set(sources) == {"c1", "fact:AAPL:2024:us-gaap:Revenues"}


def test_build_source_map_across_multiple_retrievals() -> None:
    retrievals = [
        Retrieval(subquery=_subquery(), chunks=[_chunk("c1")]),
        Retrieval(subquery=_subquery(), chunks=[_chunk("c2")]),
    ]
    sources = build_source_map(retrievals)
    assert set(sources) == {"c1", "c2"}


def test_format_source_chunk_includes_citation_and_text() -> None:
    rendered = format_source("c1", _chunk("c1"))
    assert "[c1]" in rendered
    assert "AAPL" in rendered
    assert "body" in rendered


def test_format_source_fact_includes_value_and_unit() -> None:
    rendered = format_source("fact:AAPL:2024:us-gaap:Revenues", _fact())
    assert "391035000000.0" in rendered
    assert "USD" in rendered


def test_format_sources_block_empty() -> None:
    assert format_sources_block({}) == "(no sources retrieved)"


def test_format_sources_block_multiple() -> None:
    sources = {"c1": _chunk("c1"), "fact:AAPL:2024:us-gaap:Revenues": _fact()}
    block = format_sources_block(sources)
    assert "[c1]" in block
    assert "[fact:AAPL:2024:us-gaap:Revenues]" in block
