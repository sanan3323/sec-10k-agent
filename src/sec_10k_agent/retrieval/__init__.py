"""sec_10k_agent.retrieval — dense (pgvector) + lexical (BM25) retrieval,
fused via RRF and reranked with a cross-encoder (Phase 5).

See docs/architecture.md for design.
"""

from __future__ import annotations

from sec_10k_agent.retrieval.bm25 import BM25Index, rank_documents, tokenize
from sec_10k_agent.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from sec_10k_agent.retrieval.hybrid import DEFAULT_CANDIDATE_K, HybridRetriever
from sec_10k_agent.retrieval.models import RetrievedChunk, SearchRetriever
from sec_10k_agent.retrieval.reranker import DEFAULT_RERANK_MODEL, Reranker
from sec_10k_agent.retrieval.retriever import (
    DEFAULT_K,
    DEFAULT_MODEL,
    Retriever,
    build_filter_clause,
    build_search_sql,
    embed_query,
    to_pgvector_literal,
)

__all__ = [
    "DEFAULT_CANDIDATE_K",
    "DEFAULT_K",
    "DEFAULT_MODEL",
    "DEFAULT_RERANK_MODEL",
    "DEFAULT_RRF_K",
    "BM25Index",
    "HybridRetriever",
    "Reranker",
    "RetrievedChunk",
    "Retriever",
    "SearchRetriever",
    "build_filter_clause",
    "build_search_sql",
    "embed_query",
    "rank_documents",
    "reciprocal_rank_fusion",
    "to_pgvector_literal",
    "tokenize",
]
