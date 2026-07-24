"""Decomposition: split a question into per-entity/per-year SubQueries
(docs/architecture.md §4, `decompose` node).

Runs an LLM call only when `routing_plan.needs_decomposition` -- e.g. "Compare
Apple and NVIDIA" splits by ticker, "How did Apple's risk factors evolve?"
splits by fiscal year. When decomposition isn't needed, this also folds in
architecture.md's separate `plan_retrievals` node: if the routing plan asks for
more than one retrieval mode (e.g. both semantic and structured_xbrl for one
entity/year), one SubQuery per mode is emitted directly, no LLM call needed --
mode selection for a single, already-scoped question doesn't require a model.
"""

from __future__ import annotations

from sec_10k_agent.agent.json_utils import extract_json_object
from sec_10k_agent.agent.state import RoutingPlan, SubQuery
from sec_10k_agent.rag.llm import Generator

DECOMPOSE_SYSTEM_PROMPT = """You are the query decomposer for a SEC 10-K question-answering system.

You receive the user's original question and a routing plan. Split the question into
one or more focused subqueries, each targeting exactly one ticker and (when the
question is about a specific fiscal year or a temporal comparison) one fiscal year.

Rules:
- Split by ticker when the question spans multiple companies (e.g. "compare Apple and
  NVIDIA" -> one subquery per ticker).
- Split by fiscal year when the question is temporal / tracks change over time (e.g.
  "how did risk factors evolve" -> one subquery per fiscal year in scope).
- If both apply, only cross tickers and years the question actually asks about --
  do not produce every combination unless the question requires it.
- Each subquery's "mode" must be one of the routing plan's retrieval_modes.
- Each subquery's "question" should be a natural, focused restatement for that single
  (ticker, fiscal_year) slice -- not the original multi-entity question verbatim.

If a subquery's mode is "structured_xbrl", also include "concept": a short financial-
metric phrase for that slice (e.g. "total revenue"). Omit or null it for "semantic".

Output ONLY a JSON object:
{"subqueries": [
  {"question": "...", "ticker": "...", "fiscal_year": <int or null>,
   "section": "<code or null>", "concept": "<phrase or null>",
   "mode": "semantic" | "structured_xbrl"}
]}"""


def _fan_out_by_mode(question: str, plan: RoutingPlan) -> list[SubQuery]:
    """No decomposition needed: one SubQuery per requested retrieval mode,
    using the plan's filters directly (singular fields only -- if the plan
    somehow carries more than one ticker/year without needing decomposition,
    that's a router inconsistency and we take the first as a safe default)."""
    f = plan.candidate_filters
    ticker = f.tickers[0] if f.tickers else None
    fiscal_year = f.fiscal_years[0] if f.fiscal_years else None
    section = f.sections[0] if f.sections else None
    modes = plan.retrieval_modes or ["semantic"]
    return [
        SubQuery(
            question=question,
            ticker=ticker,
            fiscal_year=fiscal_year,
            section=section,
            concept=plan.concept_hint if mode == "structured_xbrl" else None,
            mode=mode,
        )
        for mode in modes
    ]


def decompose(question: str, plan: RoutingPlan, generator: Generator) -> list[SubQuery]:
    """Return the SubQueries to retrieve for `question` given its RoutingPlan."""
    if not plan.needs_decomposition:
        return _fan_out_by_mode(question, plan)

    user_prompt = f"ORIGINAL QUESTION: {question}\nROUTING PLAN: {plan.model_dump_json()}"
    response = generator.complete(DECOMPOSE_SYSTEM_PROMPT, user_prompt, temperature=0.0)
    obj = extract_json_object(response.text)
    raw_subqueries = obj.get("subqueries") or []
    subqueries = [SubQuery.model_validate(sq) for sq in raw_subqueries]
    if not subqueries:
        raise ValueError(f"decomposer returned no subqueries for question: {question!r}")
    return subqueries
