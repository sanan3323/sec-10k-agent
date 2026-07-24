"""Retrieval result types.

`RetrievedChunk` is what the retriever returns and what the RAG pipeline
consumes. It mirrors the stored `Chunk` (see ingestion/models.py) plus the
similarity score for the query that surfaced it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """One chunk returned by a similarity search, with its distance to the query."""

    chunk_id: str
    ticker: str | None = None
    fiscal_year: int | None = None
    section: str | None = None
    section_title: str | None = None
    text: str
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    distance: float = Field(
        ...,
        description="Cosine distance from pgvector's `<=>` operator. Lower is closer; range [0, 2].",
    )
    fusion_score: float | None = Field(
        None, description="Reciprocal-rank-fusion score when combined via hybrid retrieval."
    )
    rerank_score: float | None = Field(
        None, description="Cross-encoder relevance score when reranked (Phase 5)."
    )

    @property
    def score(self) -> float:
        """Cosine similarity in [-1, 1]. Convenience inverse of `distance`."""
        return 1.0 - self.distance

    def citation(self) -> str:
        """Short human-readable source label, e.g. 'AAPL FY2024 Item 1A'."""
        parts = [p for p in (self.ticker, self.section) if p]
        if self.fiscal_year:
            parts.insert(1 if self.ticker else 0, f"FY{self.fiscal_year}")
        return " ".join(parts) or self.chunk_id
