"""HybridRetriever: dense + BM25 -> reciprocal rank fusion -> cross-encoder rerank.

This is the Phase 5 retrieval leg. It exposes the same `.search(query, *,
ticker=None, fiscal_year=None, section=None, k=...)` shape as `Retriever`, so
it drops into `RAGPipeline` (and the eval runner) with no changes to either --
both depend only on that duck-typed interface, not the concrete class.

Pipeline per query:
  1. Retrieve `dense_k` candidates from pgvector (semantic).
  2. Retrieve `bm25_k` candidates from BM25 (lexical), same filters.
  3. Fuse both rankings with RRF -- cheap, no model calls.
  4. Rerank the fused candidate set with a cross-encoder (expensive; only runs
     on the fused set, not the whole corpus) and keep the top `k`.

Step 4 is optional (`use_reranker=False`) so the fusion-only variant can be
measured separately from the rerank lift in eval comparisons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sec_10k_agent.retrieval.bm25 import BM25Index
from sec_10k_agent.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from sec_10k_agent.retrieval.reranker import Reranker
from sec_10k_agent.retrieval.retriever import DEFAULT_K, Retriever

if TYPE_CHECKING:
    from sec_10k_agent.retrieval.models import RetrievedChunk

# How many candidates each leg contributes to fusion, before reranking trims
# to the caller's requested k. Wider than k so fusion has real signal to mix.
DEFAULT_CANDIDATE_K = 20


class HybridRetriever:
    """Combines dense + lexical retrieval via RRF, then reranks the fusion.

    Args:
        retriever: Dense leg. Defaults to a `Retriever()` against `settings`.
        bm25: Lexical leg. Defaults to a `BM25Index` sharing the dense
            retriever's engine (one DB connection pool for both legs).
        reranker: Cross-encoder reranker. Defaults to `Reranker()`. Pass
            `use_reranker=False` to skip reranking entirely (fusion-only).
        dense_k / bm25_k: Candidates each leg contributes before fusion.
        rrf_k: The RRF damping constant (see fusion.py).
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        bm25: BM25Index | None = None,
        reranker: Reranker | None = None,
        *,
        use_reranker: bool = True,
        dense_k: int = DEFAULT_CANDIDATE_K,
        bm25_k: int = DEFAULT_CANDIDATE_K,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self._retriever = retriever or Retriever()
        self._bm25 = bm25 or BM25Index(engine=self._retriever.engine)
        self._reranker = (
            reranker if reranker is not None else (Reranker() if use_reranker else None)
        )
        self._dense_k = dense_k
        self._bm25_k = bm25_k
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        *,
        ticker: str | None = None,
        fiscal_year: int | None = None,
        section: str | None = None,
        k: int = DEFAULT_K,
    ) -> list[RetrievedChunk]:
        """Return the top `k` chunks after dense+lexical fusion and reranking."""
        dense = self._retriever.search(
            query, ticker=ticker, fiscal_year=fiscal_year, section=section, k=self._dense_k
        )
        lexical = self._bm25.search(
            query, ticker=ticker, fiscal_year=fiscal_year, section=section, k=self._bm25_k
        )
        fused = reciprocal_rank_fusion([dense, lexical], k=self._rrf_k)
        if self._reranker is not None:
            return self._reranker.rerank(query, fused, top_n=k)
        return fused[:k]
