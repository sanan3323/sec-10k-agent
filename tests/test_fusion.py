"""Tests for reciprocal rank fusion."""

from __future__ import annotations

from sec_10k_agent.retrieval.fusion import reciprocal_rank_fusion
from sec_10k_agent.retrieval.models import RetrievedChunk


def _chunk(chunk_id: str, distance: float = 0.1) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=f"text for {chunk_id}", distance=distance)


def test_fusion_boosts_chunk_present_in_both_rankings() -> None:
    dense = [_chunk("a"), _chunk("b"), _chunk("c")]
    lexical = [_chunk("b"), _chunk("d"), _chunk("a")]
    fused = reciprocal_rank_fusion([dense, lexical])
    ids = [c.chunk_id for c in fused]
    # "a" (rank 1 dense, rank 3 lexical) and "b" (rank 2 dense, rank 1 lexical)
    # both appear in both lists; either could lead depending on RRF math, but
    # both must outrank items that appear in only one list.
    assert set(ids[:2]) == {"a", "b"}
    assert "c" in ids and "d" in ids
    assert ids.index("c") > 1
    assert ids.index("d") > 1


def test_fusion_deduplicates_by_chunk_id() -> None:
    dense = [_chunk("a"), _chunk("b")]
    lexical = [_chunk("a"), _chunk("b")]
    fused = reciprocal_rank_fusion([dense, lexical])
    assert len(fused) == 2
    assert {c.chunk_id for c in fused} == {"a", "b"}


def test_fusion_sets_fusion_score() -> None:
    fused = reciprocal_rank_fusion([[_chunk("a")], [_chunk("a")]], k=60)
    assert fused[0].fusion_score == 2 * (1.0 / 61)


def test_fusion_single_ranking_preserves_order() -> None:
    ranking = [_chunk("a"), _chunk("b"), _chunk("c")]
    fused = reciprocal_rank_fusion([ranking])
    assert [c.chunk_id for c in fused] == ["a", "b", "c"]


def test_fusion_empty_rankings() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_fusion_solo_appearance_ranks_below_double_appearance() -> None:
    dense = [_chunk("solo"), _chunk("shared")]
    lexical = [_chunk("shared")]
    fused = reciprocal_rank_fusion([dense, lexical])
    assert fused[0].chunk_id == "shared"
