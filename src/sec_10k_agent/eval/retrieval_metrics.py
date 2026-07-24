"""Deterministic retrieval metrics — no LLM, no cost.

These score whether retrieval surfaced the chunks an item declares in
`must_cite`. They run on every eval and are the fast signal: if the right
chunks never get retrieved, no amount of generation quality can save the answer.

A CiteSpec matches a RetrievedChunk when every *set* field on the spec equals
the chunk's value (unset fields are wildcards). chunk_id, when given, is an
exact match and ignores the coarser fields.
"""

from __future__ import annotations

from sec_10k_agent.eval.dataset import CiteSpec, GoldenItem
from sec_10k_agent.retrieval.models import RetrievedChunk


def spec_matches_chunk(spec: CiteSpec, chunk: RetrievedChunk) -> bool:
    """True if `chunk` satisfies `spec`."""
    if spec.chunk_id is not None:
        return chunk.chunk_id == spec.chunk_id
    if spec.ticker is not None and chunk.ticker != spec.ticker:
        return False
    if spec.fiscal_year is not None and chunk.fiscal_year != spec.fiscal_year:
        return False
    if spec.section is not None and chunk.section != spec.section:
        return False
    # An all-wildcard spec matches nothing meaningful; treat as no-match.
    return any(v is not None for v in (spec.ticker, spec.fiscal_year, spec.section))


def context_recall(item: GoldenItem, chunks: list[RetrievedChunk]) -> float | None:
    """Fraction of `must_cite` specs that appear anywhere in `chunks`.

    Returns None when the item has no `must_cite` (e.g. negative controls),
    since recall is undefined there and should be excluded from averages.
    """
    if not item.must_cite:
        return None
    hits = sum(1 for spec in item.must_cite if any(spec_matches_chunk(spec, c) for c in chunks))
    return hits / len(item.must_cite)


def hit_at_k(item: GoldenItem, chunks: list[RetrievedChunk], k: int | None = None) -> bool | None:
    """Whether at least one `must_cite` spec is matched within the top-k chunks."""
    if not item.must_cite:
        return None
    top = chunks if k is None else chunks[:k]
    return any(spec_matches_chunk(spec, c) for spec in item.must_cite for c in top)


def reciprocal_rank(item: GoldenItem, chunks: list[RetrievedChunk]) -> float | None:
    """1 / rank of the first chunk matching any `must_cite` spec; 0.0 if none."""
    if not item.must_cite:
        return None
    for rank, chunk in enumerate(chunks, start=1):
        if any(spec_matches_chunk(spec, chunk) for spec in item.must_cite):
            return 1.0 / rank
    return 0.0
