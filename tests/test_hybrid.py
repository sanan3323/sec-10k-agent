"""Tests for HybridRetriever, with faked dense/lexical/reranker legs.

No DB, no embedding model, no cross-encoder: every leg is injected so the
wiring (candidate counts, filter passthrough, fusion, optional reranking) is
tested in isolation.
"""

from __future__ import annotations

from sec_10k_agent.retrieval.hybrid import HybridRetriever
from sec_10k_agent.retrieval.models import RetrievedChunk


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=f"text {chunk_id}", distance=0.1)


class _FakeLeg:
    """Records the call it received and returns a canned ranking."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.last_call: dict[str, object] = {}

    def search(self, query: str, **kwargs: object) -> list[RetrievedChunk]:
        self.last_call = {"query": query, **kwargs}
        return self._chunks


class _FakeReranker:
    def __init__(self) -> None:
        self.last_call: tuple[str, list[str]] | None = None

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int | None = None
    ) -> list[RetrievedChunk]:
        self.last_call = (query, [c.chunk_id for c in chunks])
        # Reverse order so the test can tell reranking actually ran.
        reranked = list(reversed(chunks))
        return reranked[:top_n] if top_n is not None else reranked


def test_hybrid_fuses_dense_and_lexical() -> None:
    dense = _FakeLeg([_chunk("a"), _chunk("b")])
    lexical = _FakeLeg([_chunk("b"), _chunk("c")])
    hybrid = HybridRetriever(retriever=dense, bm25=lexical, reranker=None, use_reranker=False)  # type: ignore[arg-type]

    results = hybrid.search("q", ticker="AAPL", k=5)

    ids = {c.chunk_id for c in results}
    assert ids == {"a", "b", "c"}
    # "b" appears in both legs, so it should be first after fusion.
    assert results[0].chunk_id == "b"


def test_hybrid_passes_filters_and_candidate_k_to_both_legs() -> None:
    dense = _FakeLeg([_chunk("a")])
    lexical = _FakeLeg([_chunk("a")])
    hybrid = HybridRetriever(
        retriever=dense,  # type: ignore[arg-type]
        bm25=lexical,  # type: ignore[arg-type]
        reranker=None,
        use_reranker=False,
        dense_k=7,
        bm25_k=9,
    )
    hybrid.search("q", ticker="JPM", fiscal_year=2023, section="Item 1A", k=3)
    assert dense.last_call == {
        "query": "q",
        "ticker": "JPM",
        "fiscal_year": 2023,
        "section": "Item 1A",
        "k": 7,
    }
    assert lexical.last_call == {
        "query": "q",
        "ticker": "JPM",
        "fiscal_year": 2023,
        "section": "Item 1A",
        "k": 9,
    }


def test_hybrid_reranks_fused_candidates_when_reranker_given() -> None:
    dense = _FakeLeg([_chunk("a"), _chunk("b")])
    lexical = _FakeLeg([])
    reranker = _FakeReranker()
    hybrid = HybridRetriever(retriever=dense, bm25=lexical, reranker=reranker)  # type: ignore[arg-type]

    results = hybrid.search("q", k=2)

    assert reranker.last_call is not None
    query, ids_seen = reranker.last_call
    assert query == "q"
    assert set(ids_seen) == {"a", "b"}
    # The fake reranker reverses order -> "b" (fused rank 2) comes first.
    assert results[0].chunk_id == "b"


def test_hybrid_skips_reranking_when_use_reranker_false_and_no_reranker_given() -> None:
    dense = _FakeLeg([_chunk("a")])
    lexical = _FakeLeg([])
    hybrid = HybridRetriever(retriever=dense, bm25=lexical, use_reranker=False)  # type: ignore[arg-type]
    results = hybrid.search("q")
    assert [c.chunk_id for c in results] == ["a"]
    assert results[0].rerank_score is None


def test_hybrid_truncates_to_k_without_reranker() -> None:
    dense = _FakeLeg([_chunk("a"), _chunk("b"), _chunk("c")])
    lexical = _FakeLeg([])
    hybrid = HybridRetriever(retriever=dense, bm25=lexical, use_reranker=False)  # type: ignore[arg-type]
    results = hybrid.search("q", k=2)
    assert len(results) == 2
