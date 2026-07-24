"""sec_10k_agent.agent — multi-hop router/decompose/retrieve/synthesize/verify
pipeline (Phase 6). See docs/architecture.md §4 and ADR-002 (hand-rolled
state machine, not LangGraph/PydanticAI).
"""

from __future__ import annotations

from sec_10k_agent.agent.citations import build_source_map, fact_key, format_sources_block
from sec_10k_agent.agent.decompose import decompose
from sec_10k_agent.agent.json_utils import extract_json_object
from sec_10k_agent.agent.retrieve import retrieve
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
from sec_10k_agent.agent.synthesize import NO_INFO_TEXT, synthesize
from sec_10k_agent.agent.tools import lookup_financial_metric

__all__ = [
    "NO_INFO_TEXT",
    "AgentState",
    "CandidateFilters",
    "Claim",
    "Retrieval",
    "RoutingPlan",
    "SubQuery",
    "XBRLFact",
    "build_source_map",
    "decompose",
    "extract_json_object",
    "fact_key",
    "format_sources_block",
    "lookup_financial_metric",
    "retrieve",
    "route",
    "synthesize",
]
