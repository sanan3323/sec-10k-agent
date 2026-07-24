"""Cross-encoder reranking: re-score a candidate set of chunks jointly with
the query, rather than independently (as embeddings and BM25 both do).

Uses fastembed's ONNX cross-encoder (`BAAI/bge-reranker-base`) rather than
FlagEmbedding's `bge-reranker-v2-m3` -- FlagEmbedding depends on torch, which
has no wheel for Intel Mac, the same wall ADR-007 already hit for embeddings.
See ADR-010. Reranking is the last, most expensive stage of hybrid retrieval,
so it runs only on the fused candidate set (tens of chunks), not the whole
corpus.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import lru_cache

from sec_10k_agent.retrieval.models import RetrievedChunk

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"

ScoreFn = Callable[[str, list[str]], Iterable[float]]


@lru_cache(maxsize=2)
def _load_reranker(model_name: str):  # type: ignore[no-untyped-def]
    """Load (and cache) a fastembed cross-encoder. Import is local so the
    package imports cheaply without the reranker model downloaded."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name)


def _default_score_fn(model_name: str) -> ScoreFn:
    def score(query: str, documents: list[str]) -> list[float]:
        return [float(s) for s in _load_reranker(model_name).rerank(query, documents)]

    return score


class Reranker:
    """Cross-encoder reranker. Inject `score_fn` in tests to avoid the model."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        score_fn: ScoreFn | None = None,
    ) -> None:
        self._model_name = model_name
        self._score_fn = score_fn or _default_score_fn(model_name)

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int | None = None
    ) -> list[RetrievedChunk]:
        """Re-score `chunks` jointly with `query` and return them best-first,
        with `rerank_score` populated. Empty input returns empty output."""
        if not chunks:
            return []
        scores = list(self._score_fn(query, [c.text for c in chunks]))
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        ranked = [chunks[i].model_copy(update={"rerank_score": float(scores[i])}) for i in order]
        return ranked[:top_n] if top_n is not None else ranked
