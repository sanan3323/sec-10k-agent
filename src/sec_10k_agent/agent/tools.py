"""Structured retrieval tool over `financial_facts` (the XBRL fact table).

This is how "What was Apple's Greater China revenue in FY2024?" becomes a
structured lookup instead of an embedding-search guess (docs/architecture.md
§4, `lookup_financial_metric` tool). Complements semantic retrieval for
numeric/dimensional questions the chunk-based search isn't built for.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sqlalchemy import create_engine, text

from sec_10k_agent.agent.state import XBRLFact
from sec_10k_agent.config import get_settings

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


def lookup_financial_metric(
    ticker: str,
    fiscal_year: int,
    concept: str,
    dimensions: dict[str, str] | None = None,
    engine: Engine | None = None,
) -> list[XBRLFact]:
    """Look up XBRL facts for `(ticker, fiscal_year, concept)`.

    `concept` matches exactly first (e.g. `"us-gaap:Revenues"`); rows that only
    match case-insensitively as a substring are still returned, but ranked
    after exact matches, so a caller doesn't need the precise XBRL tag name.

    `dimensions`, when given, filters to facts whose `dimensions` JSONB column
    contains every given (axis, member) pair -- e.g.
    `{"srt:StatementGeographicalAxis": "country:CN"}` for a geographic
    breakdown. Facts with no matching dimensional data return an empty list;
    that's a real answer ("the filings don't have this breakdown"), not an
    error.
    """
    engine = engine or create_engine(get_settings().postgres_dsn)
    conditions = [
        "ticker = :ticker",
        "fiscal_year = :fiscal_year",
        "(concept = :concept OR concept ILIKE :concept_like)",
    ]
    params: dict[str, object] = {
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "concept": concept,
        "concept_like": f"%{concept}%",
    }
    if dimensions:
        conditions.append("dimensions @> CAST(:dims AS jsonb)")
        params["dims"] = json.dumps(dimensions)

    sql = f"""
        SELECT ticker, fiscal_year, concept, value, unit, dimensions
        FROM financial_facts
        WHERE {" AND ".join(conditions)}
        ORDER BY (concept = :concept) DESC, concept
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return [
        XBRLFact(
            ticker=r.ticker,
            fiscal_year=r.fiscal_year,
            concept=r.concept,
            value=float(r.value),
            unit=r.unit,
            dimensions=r.dimensions or {},
        )
        for r in rows
    ]
