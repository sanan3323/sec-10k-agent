"""Shared source-labeling helpers for the synthesize and verify nodes.

`synthesize` needs to show the LLM a labeled set of sources (chunks and XBRL
facts) and get claims back tagged by label; `verify` needs to resolve those
same labels back to the underlying evidence for the entailment check. Both
nodes rebuild the same map from `Retrieval.chunks`/`Retrieval.facts` (already
stored on `AgentState.retrievals`) rather than threading a second return value
through the pipeline.

Chunks are labeled by their own `chunk_id`. XBRL facts have no natural id, so
they get a synthetic `fact:TICKER:FY:CONCEPT` key.
"""

from __future__ import annotations

from sec_10k_agent.agent.state import Retrieval, XBRLFact
from sec_10k_agent.retrieval.models import RetrievedChunk

Source = RetrievedChunk | XBRLFact


def fact_key(fact: XBRLFact) -> str:
    """Synthetic, stable id for an XBRL fact, used as its citation label."""
    return f"fact:{fact.ticker}:{fact.fiscal_year}:{fact.concept}"


def build_source_map(retrievals: list[Retrieval]) -> dict[str, Source]:
    """Flatten every retrieved chunk/fact across all subqueries into one
    id -> source map, deduplicated by id (first occurrence wins)."""
    sources: dict[str, Source] = {}
    for retrieval in retrievals:
        for chunk in retrieval.chunks:
            sources.setdefault(chunk.chunk_id, chunk)
        for fact in retrieval.facts:
            sources.setdefault(fact_key(fact), fact)
    return sources


def format_source(source_id: str, source: Source) -> str:
    """Render one labeled source for the synthesize prompt."""
    if isinstance(source, RetrievedChunk):
        return f"[{source_id}] {source.citation()}\n{source.text.strip()}"
    dims = f" {source.dimensions}" if source.dimensions else ""
    unit = f" {source.unit}" if source.unit else ""
    return f"[{source_id}] {source.ticker} FY{source.fiscal_year} {source.concept}{dims} = {source.value}{unit}"


def format_sources_block(sources: dict[str, Source]) -> str:
    """Render the full id -> source map as a labeled block."""
    if not sources:
        return "(no sources retrieved)"
    return "\n\n".join(format_source(sid, s) for sid, s in sources.items())
