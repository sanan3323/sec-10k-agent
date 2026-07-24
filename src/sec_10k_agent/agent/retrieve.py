"""Retrieve node: run each SubQuery against the mode it asked for
(docs/architecture.md §4, `retrieve` node -- folds in `plan_retrievals`'s
per-subquery dispatch decision, which is just reading `SubQuery.mode`).

"semantic" subqueries hit the chunk retriever (dense or hybrid -- whichever
`SearchRetriever` the caller passes in, same pattern as RAGPipeline).
"structured_xbrl" subqueries hit `lookup_financial_metric` instead, using
`SubQuery.concept` as the search term.
"""

from __future__ import annotations

from collections.abc import Callable

from sec_10k_agent.agent.state import Retrieval, SubQuery, XBRLFact
from sec_10k_agent.agent.tools import lookup_financial_metric
from sec_10k_agent.retrieval.models import SearchRetriever
from sec_10k_agent.retrieval.retriever import DEFAULT_K

XBRLLookup = Callable[..., list[XBRLFact]]


def _retrieve_one(
    subquery: SubQuery, retriever: SearchRetriever, xbrl_lookup: XBRLLookup, k: int
) -> Retrieval:
    if subquery.mode == "structured_xbrl":
        if subquery.ticker is None or subquery.fiscal_year is None:
            # Can't look up a fact without knowing which filing it's in.
            return Retrieval(subquery=subquery, facts=[])
        facts = xbrl_lookup(
            ticker=subquery.ticker,
            fiscal_year=subquery.fiscal_year,
            concept=subquery.concept or subquery.question,
        )
        return Retrieval(subquery=subquery, facts=facts)

    chunks = retriever.search(
        subquery.question,
        ticker=subquery.ticker,
        fiscal_year=subquery.fiscal_year,
        section=subquery.section,
        k=k,
    )
    return Retrieval(subquery=subquery, chunks=chunks)


def retrieve(
    subqueries: list[SubQuery],
    retriever: SearchRetriever,
    xbrl_lookup: XBRLLookup = lookup_financial_metric,
    k: int = DEFAULT_K,
) -> list[Retrieval]:
    """Run every subquery and return one Retrieval per subquery, in order.

    `xbrl_lookup` is injectable (defaults to the real `lookup_financial_metric`)
    so tests never touch the database.
    """
    return [_retrieve_one(sq, retriever, xbrl_lookup, k) for sq in subqueries]
