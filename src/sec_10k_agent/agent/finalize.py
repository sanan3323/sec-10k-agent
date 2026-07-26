"""Finalize node: format the user-facing answer from verified claims, with
`[Source: TICKER FYyyyy 10-K, Item N]`-style citations resolved from chunk_ids
(docs/architecture.md §4, `finalize` node).

Unverified claims are dropped here, not earlier -- `verify_citations` tags,
`finalize` decides what a dropped claim means for the answer as a whole (no
verified claims at all collapses to the honest abstention, matching Phase 3's
generator-level fallback for empty retrieval).
"""

from __future__ import annotations

from sec_10k_agent.agent.citations import Source, build_source_map
from sec_10k_agent.agent.state import Claim, Retrieval
from sec_10k_agent.agent.synthesize import NO_INFO_TEXT
from sec_10k_agent.retrieval.models import RetrievedChunk


def _citation_label(source: Source) -> str:
    if isinstance(source, RetrievedChunk):
        head = " ".join(
            p
            for p in (source.ticker, f"FY{source.fiscal_year}" if source.fiscal_year else None)
            if p
        )
        head = head or source.chunk_id
        return f"{head} 10-K, {source.section}" if source.section else f"{head} 10-K"
    return f"{source.ticker} FY{source.fiscal_year} 10-K, {source.concept}"


def finalize(claims: list[Claim], retrievals: list[Retrieval]) -> str:
    """Join verified claims into the final answer text, each suffixed with its
    resolved `[Source: ...]` citation(s). Returns the honest abstention if no
    claim passed verification."""
    sources = build_source_map(retrievals)
    verified = [c for c in claims if c.verified]
    if not verified:
        return NO_INFO_TEXT

    lines = []
    for claim in verified:
        labels = list(
            dict.fromkeys(
                _citation_label(sources[cid]) for cid in claim.chunk_ids if cid in sources
            )
        )
        suffix = "".join(f" [Source: {label}]" for label in labels)
        lines.append(f"{claim.text}{suffix}")
    return "\n".join(lines)
