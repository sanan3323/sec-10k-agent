"""Typed state threaded through the agent pipeline. See docs/architecture.md
§4 (Agent) for the node graph this backs.

Deviation from the architecture doc: `retrieval_modes` there is typed
`list[Literal["semantic", "structured_xbrl", "both"]]`. A list already
expresses "both" by containing both entries, so `"both"` as a third list
member is redundant; this drops it to `list[Literal["semantic",
"structured_xbrl"]]` and lets a plan request both by listing both.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from sec_10k_agent.retrieval.models import RetrievedChunk

RetrievalMode = Literal["semantic", "structured_xbrl"]


class CandidateFilters(BaseModel):
    """Coarse filter hints from the router. The decomposer refines these into
    concrete per-subquery filters."""

    tickers: list[str] = Field(default_factory=list)
    fiscal_years: list[int] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)


class RoutingPlan(BaseModel):
    """Structured retrieval plan produced by the router (agent/router.py)."""

    needs_decomposition: bool
    retrieval_modes: list[RetrievalMode]
    is_temporal: bool
    candidate_filters: CandidateFilters
    reasoning: str = ""


class SubQuery(BaseModel):
    """One decomposed unit of work: a question plus the exact filters and
    retrieval mode to run it with."""

    question: str
    ticker: str | None = None
    fiscal_year: int | None = None
    section: str | None = None
    mode: RetrievalMode = "semantic"


class XBRLFact(BaseModel):
    """One structured fact returned by the `lookup_financial_metric` tool."""

    ticker: str
    fiscal_year: int
    concept: str
    value: float
    unit: str | None = None
    dimensions: dict[str, str] = Field(default_factory=dict)


class Retrieval(BaseModel):
    """The result of running one SubQuery: either retrieved chunks (semantic)
    or structured facts (structured_xbrl), tagged back to the subquery."""

    subquery: SubQuery
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    facts: list[XBRLFact] = Field(default_factory=list)


class Claim(BaseModel):
    """One assertion in the draft answer, tagged with the chunk_id(s) or fact
    concept(s) it claims to rely on. `verified` is None until verify_citations
    runs; False means the entailment check rejected the claim."""

    text: str
    chunk_ids: list[str] = Field(default_factory=list)
    verified: bool | None = None


class AgentState(BaseModel):
    """Threaded through every node. Mirrors architecture.md's state fields,
    named for Python (draft_claims/verified_claims instead of a single
    draft_answer/verified_citations pair, since claims carry their own
    per-claim verification status)."""

    question: str
    routing_plan: RoutingPlan | None = None
    subqueries: list[SubQuery] = Field(default_factory=list)
    retrievals: list[Retrieval] = Field(default_factory=list)
    draft_claims: list[Claim] = Field(default_factory=list)
    verified_claims: list[Claim] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    retry_count: int = 0
    final_answer: str = ""
