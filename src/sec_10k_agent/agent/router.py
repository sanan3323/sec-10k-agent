"""Query router: an LLM call that turns a raw question into a structured
`RoutingPlan` (docs/architecture.md §0). Not keyword matching -- questions like
"How has Apple's exposure to the Chinese consumer changed?" have no obvious
keywords but are clearly time-aware and single-company.

Kept separate from decomposition/synthesis so it can run on a cheap model and
be evaluated on its own; routing is the main lever for both latency and cost.
"""

from __future__ import annotations

from sec_10k_agent.agent.json_utils import extract_json_object
from sec_10k_agent.agent.state import CandidateFilters, RoutingPlan
from sec_10k_agent.rag.llm import Generator
from sec_10k_agent.scope import FISCAL_YEARS, TICKERS

ROUTER_SYSTEM_PROMPT = f"""You are a query router for a SEC 10-K question-answering system.

Corpus scope: tickers {", ".join(TICKERS)}; fiscal years {min(FISCAL_YEARS)}-{max(FISCAL_YEARS)}. \
Common section codes: "Item 1" (business), "Item 1A" (risk factors), "Item 1C" (cybersecurity, \
FY2024+ only), "Item 7" (MD&A), "Item 7A" (market risk), "Item 8" (financial statements).

Given a user question, output ONLY a JSON object with this shape:
{{
  "needs_decomposition": <bool>,
  "retrieval_modes": [<"semantic" and/or "structured_xbrl">],
  "is_temporal": <bool>,
  "candidate_filters": {{"tickers": [...], "fiscal_years": [...], "sections": [...]}},
  "reasoning": "<one sentence>"
}}

Rules:
- needs_decomposition is true when the question spans more than one ticker, or asks to
  compare/track something across more than one fiscal year.
- retrieval_modes: use "structured_xbrl" for questions asking for a specific number
  (revenue, a financial metric, a dimensional breakdown like "Greater China revenue").
  Use "semantic" for narrative/risk-factor/strategy questions. Include both if the
  question needs a number AND narrative context.
- is_temporal is true for "how has X changed", "since", "evolved", "compared to
  last year", or any question spanning multiple fiscal years.
- candidate_filters should list every ticker/fiscal_year/section you can infer from
  the question and corpus scope; leave a list empty if nothing applies.
- If the question only names one ticker, only include that one ticker.
- Respond with ONLY the JSON object, no other text."""


def route(question: str, generator: Generator) -> RoutingPlan:
    """Produce a RoutingPlan for `question` via one LLM call."""
    user_prompt = f"QUESTION: {question}"
    response = generator.complete(ROUTER_SYSTEM_PROMPT, user_prompt, temperature=0.0)
    obj = extract_json_object(response.text)
    filters = CandidateFilters.model_validate(obj.get("candidate_filters") or {})
    return RoutingPlan(
        needs_decomposition=bool(obj.get("needs_decomposition", False)),
        retrieval_modes=list(obj.get("retrieval_modes") or ["semantic"]),
        is_temporal=bool(obj.get("is_temporal", False)),
        candidate_filters=filters,
        reasoning=str(obj.get("reasoning", "")).strip(),
    )
