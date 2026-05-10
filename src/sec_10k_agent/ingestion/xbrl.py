"""XBRL extractor.

Takes the raw structured facts cached by `EdgarClient.get_xbrl_facts` and
turns them into validated `XBRLFact` rows. Writes the union of all filings'
facts to `data/processed/xbrl.parquet`.

Why a separate extraction step (instead of writing parquet directly from
the EDGAR client): caching raw fact dicts as JSON keeps the on-disk format
identical to what edgartools returns. If we change the `XBRLFact` schema or
add cleaning rules later, we re-extract from the cache without re-fetching
EDGAR.

The dimensions field is the whole point of this stage. A fact with
`dimensions={}` is a total. A fact with
`dimensions={"srt:StatementGeographicalAxis": "country:CN"}` is the Greater
China line. Without capturing dimensions, "What was Apple's Greater China
revenue in FY2024?" cannot be answered.
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from sec_10k_agent.ingestion.edgar_client import EdgarClient
from sec_10k_agent.ingestion.models import Filing, XBRLFact

logger = logging.getLogger(__name__)


class XBRLExtractor:
    """Converts raw fact dicts to validated `XBRLFact` rows."""

    def __init__(self, edgar_client: EdgarClient) -> None:
        self._client = edgar_client

    def extract(self, filing: Filing) -> list[XBRLFact]:
        """Fetch (or read from cache) and validate the facts for one filing."""
        raw_facts = self._client.get_xbrl_facts(filing.ticker, filing.fiscal_year)
        out: list[XBRLFact] = []
        skipped = 0
        for raw in raw_facts:
            fact = _coerce(raw, filing)
            if fact is None:
                skipped += 1
                continue
            out.append(fact)
        if skipped:
            logger.info(
                "%s FY%s: kept %d facts, skipped %d (non-numeric, malformed dates, etc.)",
                filing.ticker,
                filing.fiscal_year,
                len(out),
                skipped,
            )
        return out


#Coercion
    
def _coerce(raw: dict, filing: Filing) -> XBRLFact | None:
    """Convert one raw fact dict to an XBRLFact, or None if it can't be
    turned into a numeric, dated fact.

    Skipped:
    - Non-numeric values (textual disclosures live in the chunk corpus, not
      the fact table).
    - Facts with no period information.
    - Facts whose value is not a valid Decimal.
    """
    try:
        value = Decimal(str(raw["value"]))
    except (InvalidOperation, KeyError, ValueError):
        return None

    period_start = _parse_date(raw.get("period_start"))
    period_end = _parse_date(raw.get("period_end"))
    if period_start is None or period_end is None:
        return None

    return XBRLFact(
        cik=filing.cik,
        ticker=filing.ticker,
        fiscal_year=filing.fiscal_year,
        accession_number=filing.accession_number,
        concept=str(raw.get("concept", "")),
        value=value,
        unit=str(raw.get("unit", "")),
        period_start=period_start,
        period_end=period_end,
        context_id=str(raw.get("context_id", "")),
        dimensions=dict(raw.get("dimensions") or {}),
    )


def _parse_date(s: object) -> date | None:
    if s is None or s == "":
        return None
    if isinstance(s, date):
        return s
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


#CLI helper


def extract_all_cached(
    edgar_client: EdgarClient,
    raw_dir: Path,
    processed_dir: Path,
) -> tuple[int, int, list[tuple[str, str]]]:
    """Extract XBRL facts for every filing whose Filing metadata is in raw_dir.

    Writes the union to `processed_dir/xbrl.parquet`. Returns
    (n_filings_processed, n_facts_total, failures).
    """
    import pandas as pd

    extractor = XBRLExtractor(edgar_client)
    all_facts: list[dict] = []
    n_filings = 0
    failures: list[tuple[str, str]] = []

    for meta_path in sorted(raw_dir.glob("*.json")):
        # Skip the per-filing XBRL cache files (suffix _xbrl.json).
        if meta_path.stem.endswith("_xbrl"):
            continue
        try:
            filing = Filing.model_validate_json(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append((meta_path.name, f"unreadable filing metadata: {exc}"))
            continue

        try:
            facts = extractor.extract(filing)
            all_facts.extend(f.model_dump(mode="json") for f in facts)
            n_filings += 1
            logger.info(
                "  OK  %s FY%s  facts=%d", filing.ticker, filing.fiscal_year, len(facts)
            )
        except Exception as exc:
            failures.append((filing.accession_number, str(exc)))

    out_path = processed_dir / "xbrl.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if all_facts:
        # `dimensions` is dict-typed; pandas serializes to parquet via pyarrow
        # which handles maps natively. Explicit object dtype keeps it intact.
        df = pd.DataFrame(all_facts)
        df.to_parquet(out_path, index=False)
    else:
        # Write an empty file so downstream consumers always find a parquet.
        pd.DataFrame(
            columns=[
                "cik", "ticker", "fiscal_year", "accession_number",
                "concept", "value", "unit", "period_start", "period_end",
                "context_id", "dimensions",
            ]
        ).to_parquet(out_path, index=False)

    return n_filings, len(all_facts), failures
