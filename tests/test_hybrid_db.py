"""Live end-to-end test for HybridRetriever against the real pgvector corpus.

Skipped when the corpus is unreachable, same as tests/test_retrieval_db.py.
Downloads the ~1 GB reranker model on first run.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from sec_10k_agent.config import get_settings
from sec_10k_agent.retrieval import HybridRetriever, RetrievedChunk

pytestmark = pytest.mark.db


def _corpus_available() -> bool:
    try:
        engine = create_engine(get_settings().postgres_dsn)
        with engine.connect() as conn:
            n = conn.execute(
                text("SELECT count(*) FROM text_chunks WHERE embedding IS NOT NULL")
            ).scalar()
        return bool(n and n > 0)
    except (SQLAlchemyError, Exception):
        return False


requires_corpus = pytest.mark.skipif(
    not _corpus_available(),
    reason="Postgres/pgvector corpus not reachable — start docker-compose and load chunks.",
)


@pytest.fixture(scope="module")
def hybrid() -> HybridRetriever:
    return HybridRetriever()


@requires_corpus
def test_hybrid_search_returns_ranked_reranked_chunks(hybrid: HybridRetriever) -> None:
    results = hybrid.search("TSMC foundry fabrication risk", ticker="NVDA", k=5)
    assert results, "expected at least one chunk for an in-corpus query"
    assert all(isinstance(r, RetrievedChunk) for r in results)
    assert all(r.ticker == "NVDA" for r in results)
    # Reranked results carry a rerank_score.
    assert all(r.rerank_score is not None for r in results)


@requires_corpus
def test_hybrid_fusion_only_when_reranker_disabled() -> None:
    fusion_only = HybridRetriever(use_reranker=False)
    results = fusion_only.search("credit risk exposure", ticker="JPM", section="Item 1A", k=3)
    assert results
    assert all(r.ticker == "JPM" and r.section == "Item 1A" for r in results)
    assert all(r.rerank_score is None for r in results)
    assert all(r.fusion_score is not None for r in results)
