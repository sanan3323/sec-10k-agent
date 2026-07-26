"""Tests for the finalize node."""

from __future__ import annotations

from sec_10k_agent.agent.finalize import finalize
from sec_10k_agent.agent.state import Claim, Retrieval, SubQuery, XBRLFact
from sec_10k_agent.agent.synthesize import NO_INFO_TEXT
from sec_10k_agent.retrieval.models import RetrievedChunk


def _chunk_retrieval() -> Retrieval:
    chunk = RetrievedChunk(
        chunk_id="c1", ticker="AAPL", fiscal_year=2024, section="Item 1A", text="body", distance=0.1
    )
    return Retrieval(subquery=SubQuery(question="q"), chunks=[chunk])


def _fact_retrieval() -> Retrieval:
    fact = XBRLFact(ticker="AAPL", fiscal_year=2024, concept="us-gaap:Revenues", value=1.0)
    return Retrieval(subquery=SubQuery(question="q"), facts=[fact])


def test_finalize_joins_verified_claims_with_chunk_citation() -> None:
    claims = [Claim(text="Apple discloses supply chain risk.", chunk_ids=["c1"], verified=True)]
    answer = finalize(claims, [_chunk_retrieval()])
    assert "Apple discloses supply chain risk." in answer
    assert "[Source: AAPL FY2024 10-K, Item 1A]" in answer


def test_finalize_joins_verified_claims_with_fact_citation() -> None:
    key = "fact:AAPL:2024:us-gaap:Revenues"
    claims = [Claim(text="Apple's FY2024 revenue was $1.", chunk_ids=[key], verified=True)]
    answer = finalize(claims, [_fact_retrieval()])
    assert "[Source: AAPL FY2024 10-K, us-gaap:Revenues]" in answer


def test_finalize_drops_unverified_claims() -> None:
    claims = [
        Claim(text="Verified claim.", chunk_ids=["c1"], verified=True),
        Claim(text="Unverified claim.", chunk_ids=["c1"], verified=False),
    ]
    answer = finalize(claims, [_chunk_retrieval()])
    assert "Verified claim." in answer
    assert "Unverified claim." not in answer


def test_finalize_no_verified_claims_returns_abstention() -> None:
    claims = [Claim(text="Unverified.", chunk_ids=["c1"], verified=False)]
    assert finalize(claims, [_chunk_retrieval()]) == NO_INFO_TEXT


def test_finalize_empty_claims_returns_abstention() -> None:
    assert finalize([], [_chunk_retrieval()]) == NO_INFO_TEXT


def test_finalize_dedupes_repeated_citation_label() -> None:
    claims = [Claim(text="A claim.", chunk_ids=["c1", "c1"], verified=True)]
    answer = finalize(claims, [_chunk_retrieval()])
    assert answer.count("[Source: AAPL FY2024 10-K, Item 1A]") == 1


def test_finalize_multiple_verified_claims_join_with_newline() -> None:
    claims = [
        Claim(text="First claim.", chunk_ids=["c1"], verified=True),
        Claim(text="Second claim.", chunk_ids=["c1"], verified=True),
    ]
    answer = finalize(claims, [_chunk_retrieval()])
    assert answer.count("\n") == 1
    assert answer.startswith("First claim.")
