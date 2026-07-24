"""Reciprocal rank fusion (RRF): combine multiple rankings of the same corpus
into one, using only rank position -- not the raw scores.

Dense (cosine distance) and lexical (BM25) scores live on different, unrelated
scales, so averaging or weighting them directly is arbitrary. RRF sidesteps
that by scoring each chunk 1/(k_rrf + rank) per ranking it appears in and
summing across rankings; a chunk that ranks well on either leg surfaces, and
one that ranks well on both surfaces highest. `k_rrf` (default 60, the
standard value from the original RRF paper) damps the influence of any single
top rank so one lucky #1 doesn't dominate.
"""

from __future__ import annotations

from sec_10k_agent.retrieval.models import RetrievedChunk

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], k: int = DEFAULT_RRF_K
) -> list[RetrievedChunk]:
    """Fuse multiple rankings of `RetrievedChunk` (each already sorted best
    first) into one, sorted by fused score descending. Chunks are deduplicated
    by `chunk_id`; the first occurrence's chunk data is kept, with
    `fusion_score` set to the combined score.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(chunk.chunk_id, chunk)

    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [first_seen[cid].model_copy(update={"fusion_score": scores[cid]}) for cid in ordered_ids]
