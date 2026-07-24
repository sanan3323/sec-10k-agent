"""Live end-to-end RAG test: real retrieval + real generation.

Skipped unless BOTH are available — the pgvector corpus is reachable and a
generator is configured (XAI_API_KEY for Grok, or OLLAMA_BASE_URL for a local
model). This is the honest proof that a question flows all the way to a cited
answer; it never runs in CI (no DB, no key).
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from sec_10k_agent.config import get_settings
from sec_10k_agent.rag import RAGPipeline

pytestmark = [pytest.mark.llm, pytest.mark.db]


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


def _generator_available() -> bool:
    s = get_settings()
    return bool(s.xai_api_key or s.ollama_base_url)


requires_stack = pytest.mark.skipif(
    not (_corpus_available() and _generator_available()),
    reason="Needs the pgvector corpus AND a generator (XAI_API_KEY or OLLAMA_BASE_URL).",
)


@requires_stack
def test_end_to_end_answer_is_grounded_and_cited() -> None:
    pipeline = RAGPipeline()
    answer = pipeline.answer(
        "What supply chain or manufacturing risks does Apple disclose?",
        ticker="AAPL",
        k=5,
    )
    assert answer.sources, "expected retrieval to return chunks"
    assert answer.text.strip(), "expected a non-empty answer"
    # A grounded answer over in-corpus content should cite at least one source.
    assert answer.cited_indices, f"expected citations, got: {answer.text!r}"
    assert all(1 <= i <= len(answer.sources) for i in answer.cited_indices)
