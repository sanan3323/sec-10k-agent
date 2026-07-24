"""sec_10k_agent.retrieval — dense retrieval over pgvector.

See docs/architecture.md for design. Phase 5 adds BM25 + reranking here.
"""

from __future__ import annotations

from sec_10k_agent.retrieval.models import RetrievedChunk
from sec_10k_agent.retrieval.retriever import (
    DEFAULT_K,
    DEFAULT_MODEL,
    Retriever,
    build_search_sql,
    embed_query,
    to_pgvector_literal,
)

__all__ = [
    "DEFAULT_K",
    "DEFAULT_MODEL",
    "RetrievedChunk",
    "Retriever",
    "build_search_sql",
    "embed_query",
    "to_pgvector_literal",
]
