"""Tests for the deterministic retrieval metrics."""

from __future__ import annotations

from sec_10k_agent.eval.dataset import CiteSpec, GoldenItem
from sec_10k_agent.eval.retrieval_metrics import (
    context_recall,
    hit_at_k,
    reciprocal_rank,
    spec_matches_chunk,
)
from sec_10k_agent.retrieval.models import RetrievedChunk


def _chunk(
    chunk_id: str, ticker: str, fy: int, section: str, distance: float = 0.1
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        ticker=ticker,
        fiscal_year=fy,
        section=section,
        text="body",
        distance=distance,
    )


def _item(specs: list[CiteSpec]) -> GoldenItem:
    return GoldenItem(id="t", bucket="single_fact", question="q?", answer="a", must_cite=specs)


def test_spec_matches_on_section_wildcard_year() -> None:
    chunk = _chunk("c1", "AAPL", 2024, "Item 1A")
    assert spec_matches_chunk(CiteSpec(ticker="AAPL", section="Item 1A"), chunk)
    assert not spec_matches_chunk(CiteSpec(ticker="AAPL", section="Item 7"), chunk)
    assert not spec_matches_chunk(CiteSpec(ticker="NVDA", section="Item 1A"), chunk)


def test_spec_matches_year_constraint() -> None:
    chunk = _chunk("c1", "AAPL", 2024, "Item 1A")
    assert spec_matches_chunk(CiteSpec(ticker="AAPL", fiscal_year=2024), chunk)
    assert not spec_matches_chunk(CiteSpec(ticker="AAPL", fiscal_year=2023), chunk)


def test_spec_chunk_id_is_exact_and_overrides_fields() -> None:
    chunk = _chunk("c1", "AAPL", 2024, "Item 1A")
    assert spec_matches_chunk(CiteSpec(chunk_id="c1", ticker="NVDA"), chunk)
    assert not spec_matches_chunk(CiteSpec(chunk_id="c2"), chunk)


def test_all_wildcard_spec_matches_nothing() -> None:
    chunk = _chunk("c1", "AAPL", 2024, "Item 1A")
    assert not spec_matches_chunk(CiteSpec(), chunk)


def test_context_recall_full_partial_none() -> None:
    chunks = [_chunk("c1", "AAPL", 2024, "Item 1A"), _chunk("c2", "NVDA", 2024, "Item 1A")]
    full = _item([CiteSpec(ticker="AAPL", section="Item 1A")])
    assert context_recall(full, chunks) == 1.0
    partial = _item(
        [CiteSpec(ticker="AAPL", section="Item 1A"), CiteSpec(ticker="JPM", section="Item 1A")]
    )
    assert context_recall(partial, chunks) == 0.5
    # No must_cite -> recall undefined.
    assert context_recall(_item([]), chunks) is None


def test_hit_at_k_respects_k() -> None:
    chunks = [
        _chunk("c1", "NVDA", 2024, "Item 1A"),
        _chunk("c2", "AAPL", 2024, "Item 1A"),
    ]
    item = _item([CiteSpec(ticker="AAPL", section="Item 1A")])
    assert hit_at_k(item, chunks, k=1) is False  # AAPL is at rank 2
    assert hit_at_k(item, chunks, k=2) is True


def test_reciprocal_rank() -> None:
    chunks = [
        _chunk("c1", "NVDA", 2024, "Item 1A"),
        _chunk("c2", "AAPL", 2024, "Item 1A"),
    ]
    item = _item([CiteSpec(ticker="AAPL", section="Item 1A")])
    assert reciprocal_rank(item, chunks) == 0.5
    miss = _item([CiteSpec(ticker="JPM", section="Item 1A")])
    assert reciprocal_rank(miss, chunks) == 0.0
    assert reciprocal_rank(_item([]), chunks) is None
