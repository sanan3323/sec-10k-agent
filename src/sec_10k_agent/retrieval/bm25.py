"""Lexical (BM25) retrieval over the same `text_chunks` table dense search uses.

Dense embeddings miss exact terms that rarely co-occur with their semantic
neighbors -- ticker symbols, defined terms, exact figures. BM25 catches those.
Combined with dense search via reciprocal rank fusion (see fusion.py), this is
the "hybrid" half of Phase 5.

`build_filter_clause` from retriever.py is reused so the lexical leg searches
exactly the same filtered row set the dense leg does -- no drift between what
"ticker=AAPL" means on each side.

BM25 needs the whole candidate corpus in memory to rank (it's a corpus-relative
score), so each distinct filter combination fetches its rows once and caches
the built index; typical single-ticker corpora are ~1,000-1,300 chunks, cheap
to tokenize and hold in memory repeatedly.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi
from sqlalchemy import create_engine, text

from sec_10k_agent.config import get_settings
from sec_10k_agent.retrieval.models import RetrievedChunk
from sec_10k_agent.retrieval.retriever import SELECT_COLUMNS, build_filter_clause

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

DEFAULT_K = 5
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text_: str) -> list[str]:
    """Lowercase word tokenization. No stemming/stopwords -- keep it simple and
    let BM25's term-frequency weighting do the work."""
    return _TOKEN_RE.findall(text_.lower())


def rank_documents(query: str, documents: list[str], k: int) -> list[tuple[int, float]]:
    """Pure BM25 ranking: (index into `documents`, score) for the top `k`,
    highest score first. No DB, no I/O -- the unit-testable core."""
    if not documents:
        return []
    corpus_tokens = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
    return [(i, float(scores[i])) for i in order[:k]]


def _normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize into [0, 1] so BM25 scores can populate `distance` in
    a way that's at least ordinally sensible if inspected directly."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class BM25Index:
    """Lexical retriever over `text_chunks`, filter-compatible with `Retriever`."""

    def __init__(self, engine: Engine | None = None) -> None:
        self._engine = engine or create_engine(get_settings().postgres_dsn)

    @lru_cache(maxsize=32)  # noqa: B019 - bound instance cache; index lifetime == process lifetime
    def _fetch_rows(
        self, ticker: str | None, fiscal_year: int | None, section: str | None
    ) -> tuple[tuple[dict, ...], tuple[str, ...]]:
        where, params = build_filter_clause(ticker, fiscal_year, section)
        columns = ", ".join(SELECT_COLUMNS)
        sql = f"SELECT {columns} FROM text_chunks {where}"
        with self._engine.connect() as conn:
            rows = [dict(r._mapping) for r in conn.execute(text(sql), params).fetchall()]
        return tuple(rows), tuple(r["text"] for r in rows)

    def search(
        self,
        query: str,
        *,
        ticker: str | None = None,
        fiscal_year: int | None = None,
        section: str | None = None,
        k: int = DEFAULT_K,
    ) -> list[RetrievedChunk]:
        """Return the `k` chunks with the highest BM25 score, best first."""
        rows, texts = self._fetch_rows(ticker, fiscal_year, section)
        ranked = rank_documents(query, list(texts), k)
        normalized_scores = _normalize_scores([score for _, score in ranked])
        return [
            RetrievedChunk(**rows[i], distance=1.0 - norm)
            for (i, _score), norm in zip(ranked, normalized_scores, strict=True)
        ]
