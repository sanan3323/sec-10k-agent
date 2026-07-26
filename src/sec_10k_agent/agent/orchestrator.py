"""Agent orchestrator: the hand-rolled state machine (ADR-002) wiring
route -> decompose -> retrieve -> synthesize -> verify_citations -> finalize,
with one bounded retry back to retrieve when verification finds unsupported
claims (docs/architecture.md §4).

Each node is a plain function taking its dependencies as arguments (generator,
retriever) -- this function is the only place that knows the pipeline order,
so tests can exercise nodes individually (see test_agent_*.py) or the whole
run (test_agent_orchestrator.py) with the same fakes either way.
"""

from __future__ import annotations

from sec_10k_agent.agent.decompose import decompose
from sec_10k_agent.agent.finalize import finalize
from sec_10k_agent.agent.retrieve import XBRLLookup, retrieve
from sec_10k_agent.agent.router import route
from sec_10k_agent.agent.state import AgentState
from sec_10k_agent.agent.synthesize import synthesize
from sec_10k_agent.agent.tools import lookup_financial_metric
from sec_10k_agent.agent.verify import verify_citations
from sec_10k_agent.rag.llm import Generator
from sec_10k_agent.retrieval.models import SearchRetriever
from sec_10k_agent.retrieval.retriever import DEFAULT_K

MAX_RETRIES = 1


def run_agent(
    question: str,
    retriever: SearchRetriever,
    generator: Generator,
    k: int = DEFAULT_K,
    max_retries: int = MAX_RETRIES,
    xbrl_lookup: XBRLLookup = lookup_financial_metric,
) -> AgentState:
    """Run the full multi-hop pipeline for `question` and return the final
    AgentState (the routing plan, subqueries, retrievals, claims, and answer
    are all inspectable on it for tracing/debugging). `xbrl_lookup` is
    injectable (defaults to the real tool) so tests never touch the database."""
    state = AgentState(question=question)
    state.routing_plan = route(question, generator)
    state.subqueries = decompose(question, state.routing_plan, generator)
    state.tool_calls = [
        f"lookup_financial_metric(ticker={sq.ticker}, fiscal_year={sq.fiscal_year}, concept={sq.concept!r})"
        for sq in state.subqueries
        if sq.mode == "structured_xbrl"
    ]

    state.retrievals = retrieve(state.subqueries, retriever, xbrl_lookup=xbrl_lookup, k=k)
    state.draft_claims = synthesize(question, state.retrievals, generator)
    state.verified_claims = verify_citations(state.draft_claims, state.retrievals, generator)

    while state.retry_count < max_retries and any(not c.verified for c in state.verified_claims):
        state.retry_count += 1
        state.retrievals = retrieve(state.subqueries, retriever, xbrl_lookup=xbrl_lookup, k=k)
        state.draft_claims = synthesize(question, state.retrievals, generator)
        state.verified_claims = verify_citations(state.draft_claims, state.retrievals, generator)

    state.final_answer = finalize(state.verified_claims, state.retrievals)
    return state
