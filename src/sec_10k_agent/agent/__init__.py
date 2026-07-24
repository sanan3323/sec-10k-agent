"""sec_10k_agent.agent — multi-hop router/decompose/retrieve/synthesize/verify
pipeline (Phase 6). See docs/architecture.md §4 and ADR-002 (hand-rolled
state machine, not LangGraph/PydanticAI).
"""

from __future__ import annotations

from sec_10k_agent.agent.json_utils import extract_json_object
from sec_10k_agent.agent.router import route
from sec_10k_agent.agent.state import (
    AgentState,
    CandidateFilters,
    Claim,
    Retrieval,
    RoutingPlan,
    SubQuery,
    XBRLFact,
)

__all__ = [
    "AgentState",
    "CandidateFilters",
    "Claim",
    "Retrieval",
    "RoutingPlan",
    "SubQuery",
    "XBRLFact",
    "extract_json_object",
    "route",
]
