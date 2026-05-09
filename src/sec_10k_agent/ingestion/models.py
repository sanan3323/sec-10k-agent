"""Data models for the ingestion pipeline.

These are the types passed between ingestion stages: download - parse -
chunk - embed. They match the schemas in docs/architecture.md. Anything that
crosses a stage boundary uses these.

Internal helper types (parser intermediates, etc.) stay private to their
module.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Filing(BaseModel):
    """One 10-K filing. Created by the EDGAR client, consumed by the parser."""

    cik: str = Field(..., description="10-digit Central Index Key, zero-padded")
    ticker: str
    form: Literal["10-K"] = "10-K"
    fiscal_year: int = Field(..., description="The FY the filing covers, NOT the year filed")
    filing_date: date
    period_of_report: date = Field(..., description="The fiscal period end date, from EDGAR")
    accession_number: str = Field(
        ..., description="EDGAR accession number, e.g. '0000320193-24-000123'"
    )
    raw_html_path: Path = Field(..., description="Path to cached HTML on local disk")
    parsed_at: datetime | None = None


class Chunk(BaseModel):
    """A retrievable unit of text. Created by the chunker, embedded by the
    vector layer, returned by retrieval."""

    chunk_id: str = Field(..., description="f'{accession_number}__{section}__{idx:04d}'")
    cik: str
    ticker: str
    fiscal_year: int
    accession_number: str
    section: str = Field(..., description="Normalized item code, e.g. 'Item 1A'")
    section_title: str = Field(..., description="Human-readable section title")
    text: str
    token_count: int
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None


class XBRLFact(BaseModel):
    """One structured fact from an iXBRL filing.

    `dimensions` is the critical field — without it, segment and geographic
    facts can't be represented correctly. See architecture.md.
    """

    cik: str
    ticker: str
    fiscal_year: int
    accession_number: str
    concept: str = Field(..., description="e.g. 'us-gaap:Revenues'")
    value: Decimal
    unit: str = Field(..., description="e.g. 'USD', 'shares'")
    period_start: date
    period_end: date
    context_id: str = Field(..., description="Raw XBRL contextRef, kept for traceability")
    dimensions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "XBRL dimensional axes. Empty dict for total values. "
            "For Apple's Greater China revenue: "
            "{'srt:StatementGeographicalAxis': 'country:CN'}."
        ),
    )
