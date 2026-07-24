"""Tests for Reranker with an injected score function (no ONNX model)."""

from __future__ import annotations

from sec_10k_agent.retrieval.models import RetrievedChunk
from sec_10k_agent.retrieval.reranker import Reranker


def _chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=text, distance=0.5)


def test_rerank_reorders_by_score_fn() -> None:
    chunks = [_chunk("a", "irrelevant"), _chunk("b", "the answer")]

    def fake_score(query: str, documents: list[str]) -> list[float]:
        # Score "b" higher regardless of input order.
        return [0.1 if doc == "irrelevant" else 0.9 for doc in documents]

    reranker = Reranker(score_fn=fake_score)
    ranked = reranker.rerank("q", chunks)
    assert [c.chunk_id for c in ranked] == ["b", "a"]
    assert ranked[0].rerank_score == 0.9
    assert ranked[1].rerank_score == 0.1


def test_rerank_respects_top_n() -> None:
    chunks = [_chunk("a", "x"), _chunk("b", "y"), _chunk("c", "z")]
    reranker = Reranker(score_fn=lambda q, docs: [0.3, 0.9, 0.1])
    ranked = reranker.rerank("q", chunks, top_n=2)
    assert len(ranked) == 2
    assert [c.chunk_id for c in ranked] == ["b", "a"]


def test_rerank_empty_input_returns_empty() -> None:
    reranker = Reranker(score_fn=lambda q, docs: [])
    assert reranker.rerank("q", []) == []


def test_rerank_preserves_other_chunk_fields() -> None:
    chunk = RetrievedChunk(
        chunk_id="a", ticker="AAPL", fiscal_year=2024, section="Item 1A", text="t", distance=0.2
    )
    reranker = Reranker(score_fn=lambda q, docs: [0.5])
    ranked = reranker.rerank("q", [chunk])
    assert ranked[0].ticker == "AAPL"
    assert ranked[0].fiscal_year == 2024
    assert ranked[0].distance == 0.2  # untouched; only rerank_score is added
