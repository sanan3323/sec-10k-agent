"""Live end-to-end retrieval tests against Postgres + pgvector.

Skipped automatically when the database is unreachable (e.g. in CI without the
docker-compose stack), so the suite stays green everywhere. Run locally with the
stack up (`docker compose -f docker-compose.dev.yml up -d`) and the corpus
loaded to exercise the real embed -> pgvector -> map path.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from sec_10k_agent.config import get_settings
from sec_10k_agent.retrieval import RetrievedChunk, Retriever

pytestmark = pytest.mark.db


def _corpus_available() -> bool:
    """True only if the DB is reachable and text_chunks has embedded rows."""
    try:
        engine = create_engine(get_settings().postgres_dsn)
        with engine.connect() as conn:
            n = conn.execute(
                text("SELECT count(*) FROM text_chunks WHERE embedding IS NOT NULL")
            ).scalar()
        return bool(n and n > 0)
    except SQLAlchemyError:
        return False
    except Exception:
        return False


requires_corpus = pytest.mark.skipif(
    not _corpus_available(),
    reason="Postgres/pgvector corpus not reachable — start docker-compose and load chunks.",
)


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever()


@requires_corpus
def test_search_returns_ranked_chunks(retriever: Retriever) -> None:
    results = retriever.search("supply chain and manufacturing risks", ticker="AAPL", k=5)
    assert results, "expected at least one chunk for an in-corpus query"
    assert all(isinstance(r, RetrievedChunk) for r in results)
    # Filter is honored.
    assert all(r.ticker == "AAPL" for r in results)
    # Distances are sorted ascending (nearest first).
    distances = [r.distance for r in results]
    assert distances == sorted(distances)
    # Bodies are populated.
    assert all(r.text.strip() for r in results)


@requires_corpus
def test_section_filter_restricts_results(retriever: Retriever) -> None:
    results = retriever.search("credit risk exposure", ticker="JPM", section="Item 1A", k=3)
    assert results
    assert all(r.section == "Item 1A" and r.ticker == "JPM" for r in results)


@requires_corpus
def test_k_limits_result_count(retriever: Retriever) -> None:
    results = retriever.search("revenue growth", k=2)
    assert len(results) <= 2
