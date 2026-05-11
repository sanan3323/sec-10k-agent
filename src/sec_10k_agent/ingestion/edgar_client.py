"""SEC EDGAR client.

Wraps `edgartools` to give us three things it doesn't enforce on its own:

1. A configurable rate limit, gated through one chokepoint.
2. Retries with exponential backoff for transient failures.
3. Local caching so we never re-download a filing we already have.

Filings are immutable once accepted by EDGAR, so cache invalidation isn't a
concern. The cache lives at `data/raw/`. Each (ticker, fiscal_year) pair
gets:
  - `data/raw/{ticker}_{fy}.html`  — the 10-K HTML
  - `data/raw/{ticker}_{fy}.json`  — Filing metadata (accession_number etc.)

If both exist, we never hit EDGAR.

Fiscal year matching: we filter by `period_of_report.year == fiscal_year`.
This works for our three tickers (AAPL, NVDA, JPM) — AAPL's FY ends in
September, NVDA's in late January, JPM's in December, and in all three cases
the company's "FY YYYY" matches the calendar year of the period end. If we
ever add a ticker with a more eccentric fiscal calendar, this assumption
needs revisiting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol, runtime_checkable

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sec_10k_agent.ingestion.models import Filing
from sec_10k_agent.ingestion.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class EdgarError(Exception):
    """Base class for EDGAR client errors."""


class FilingNotFound(EdgarError):
    """No 10-K found for the requested (ticker, fiscal_year)."""


@runtime_checkable
class _EdgartoolsBackend(Protocol):
    """The minimal slice of edgartools we depend on.

    Defined as a Protocol so tests can substitute a fake without monkey-
    patching the real `edgar` module. The real backend is a thin adapter
    that calls `edgar.set_identity`, `edgar.Company`, etc.
    """

    def set_identity(self, user_agent: str) -> None: ...

    def list_10k_filings(self, ticker: str) -> list[_FilingMeta]: ...

    def fetch_html(self, accession_number: str) -> str: ...

    def fetch_xbrl_facts(self, accession_number: str) -> list[dict]: ...
    """Returns a list of raw fact dicts. Each dict has keys:
        concept (str), value (str), unit (str), period_start (str ISO date),
        period_end (str ISO date), context_id (str), dimensions (dict[str,str]).
    Returns [] if the filing has no XBRL data."""


@dataclass(frozen=True)
class _FilingMeta:
    """Lightweight container for what edgartools returns about a filing."""

    cik: str
    ticker: str
    accession_number: str
    filing_date: str  # ISO format; parsed downstream
    period_of_report: str  # ISO format


class EdgarClient:
    """Throttled, cached, retried client for SEC EDGAR 10-Ks."""

    def __init__(
        self,
        user_agent: str,
        rate_limit_per_sec: float,
        cache_dir: Path,
        backend: _EdgartoolsBackend | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._limiter = RateLimiter(rate_limit_per_sec)
        # Lazy import so the package can be imported (and `--help` can run)
        # without `edgartools` being installed. Real callers install the
        # `vector` extra which pulls it in transitively, or install the dep
        # directly.
        if backend is None:
            backend = _RealEdgartoolsBackend()
        self._backend = backend
        self._backend.set_identity(user_agent)

    def get_10k(self, ticker: str, fiscal_year: int) -> Filing:
        """Return the 10-K for `ticker` covering FY `fiscal_year`.

        Reads from disk cache first. If not cached, hits EDGAR (throttled,
        retried) and writes both the HTML and the metadata JSON to the cache.

        Raises:
            FilingNotFound: no matching filing exists at EDGAR.
        """
        ticker = ticker.upper()
        cached = self._read_cache(ticker, fiscal_year)
        if cached is not None:
            logger.debug("cache hit: %s FY%s", ticker, fiscal_year)
            return cached

        logger.info("cache miss: %s FY%s — fetching from EDGAR", ticker, fiscal_year)
        meta = self._find_filing(ticker, fiscal_year)
        html = self._fetch_html(meta.accession_number)

        html_path = self._html_path(ticker, fiscal_year)
        html_path.write_text(html, encoding="utf-8")

        filing = Filing(
            cik=meta.cik,
            ticker=meta.ticker,
            fiscal_year=fiscal_year,
            filing_date=_parse_iso_date(meta.filing_date),
            period_of_report=_parse_iso_date(meta.period_of_report),
            accession_number=meta.accession_number,
            raw_html_path=html_path,
        )
        self._write_metadata(ticker, fiscal_year, filing)
        return filing

    def get_xbrl_facts(self, ticker: str, fiscal_year: int) -> list[dict]:
        """Return the structured XBRL facts for `(ticker, fiscal_year)`.

        Reads from disk cache first. If not cached, hits EDGAR (throttled,
        retried) and writes the JSON to the cache. Returns [] for filings
        with no XBRL data (very old filings).

        Each fact is a dict with keys: concept, value, unit, period_start,
        period_end, context_id, dimensions.
        """
        ticker = ticker.upper()
        # Resolve accession number from existing Filing cache (must exist;
        # `get_10k` is the prerequisite).
        filing = self._read_cache(ticker, fiscal_year)
        if filing is None:
            raise FilingNotFound(
                f"No cached Filing for {ticker} FY{fiscal_year}. "
                "Run `sec10k download` first."
            )

        cached = self._read_xbrl_cache(ticker, fiscal_year)
        if cached is not None:
            logger.debug("xbrl cache hit: %s FY%s (%d facts)", ticker, fiscal_year, len(cached))
            return cached

        logger.info("xbrl cache miss: %s FY%s — fetching from EDGAR", ticker, fiscal_year)
        facts = self._fetch_xbrl_facts(filing.accession_number)
        self._write_xbrl_cache(ticker, fiscal_year, facts)
        return facts

    # ─── Cache I/O ────────────────────────────────────────────────────────

    def _html_path(self, ticker: str, fy: int) -> Path:
        return self._cache_dir / f"{ticker}_{fy}.html"

    def _meta_path(self, ticker: str, fy: int) -> Path:
        return self._cache_dir / f"{ticker}_{fy}.json"

    def _xbrl_path(self, ticker: str, fy: int) -> Path:
        return self._cache_dir / f"{ticker}_{fy}_xbrl.json"

    def _read_xbrl_cache(self, ticker: str, fy: int) -> list[dict] | None:
        path = self._xbrl_path(ticker, fy)
        if not path.exists():
            return None
        try:
            import json

            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("corrupt xbrl cache for %s FY%s: %s — refetching", ticker, fy, exc)
            return None

    def _write_xbrl_cache(self, ticker: str, fy: int, facts: list[dict]) -> None:
        import json

        self._xbrl_path(ticker, fy).write_text(
            json.dumps(facts, indent=2, default=str), encoding="utf-8"
        )

    def _read_cache(self, ticker: str, fy: int) -> Filing | None:
        meta_path = self._meta_path(ticker, fy)
        html_path = self._html_path(ticker, fy)
        if not (meta_path.exists() and html_path.exists()):
            return None
        try:
            return Filing.model_validate_json(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            # Corrupt cache entry — log and re-fetch rather than raising.
            logger.warning("corrupt cache for %s FY%s: %s — refetching", ticker, fy, exc)
            return None

    def _write_metadata(self, ticker: str, fy: int, filing: Filing) -> None:
        self._meta_path(ticker, fy).write_text(
            filing.model_dump_json(indent=2), encoding="utf-8"
        )

    # ─── Throttled, retried EDGAR calls ───────────────────────────────────

    def _find_filing(self, ticker: str, fiscal_year: int) -> _FilingMeta:
        """Look up the 10-K for `ticker` covering FY `fiscal_year`."""
        filings = self._list_filings(ticker)
        matches = [
            f for f in filings if _parse_iso_date(f.period_of_report).year == fiscal_year
        ]
        if not matches:
            raise FilingNotFound(
                f"No 10-K found for {ticker} with period_of_report year == {fiscal_year}. "
                f"Available years: {sorted({_parse_iso_date(f.period_of_report).year for f in filings})}"
            )
        if len(matches) > 1:
            # Restatements / amended filings can produce multiples. Pick the
            # latest filing_date — that's the as-filed final record.
            matches.sort(key=lambda f: f.filing_date, reverse=True)
            logger.warning(
                "%s FY%s has %d matches; using latest filed (%s)",
                ticker,
                fiscal_year,
                len(matches),
                matches[0].filing_date,
            )
        return matches[0]

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def _list_filings(self, ticker: str) -> list[_FilingMeta]:
        self._limiter.wait()
        return self._backend.list_10k_filings(ticker)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def _fetch_html(self, accession_number: str) -> str:
        self._limiter.wait()
        return self._backend.fetch_html(accession_number)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def _fetch_xbrl_facts(self, accession_number: str) -> list[dict]:
        self._limiter.wait()
        return self._backend.fetch_xbrl_facts(accession_number)


# ─── edgartools adapter ─────────────────────────────────────────────────────


class _RealEdgartoolsBackend:
    """Adapter over the real `edgartools` library.

    Kept thin: this exists so we can mock `EdgarClient` in tests without
    monkey-patching the `edgar` module.
    """

    def set_identity(self, user_agent: str) -> None:
        from edgar import set_identity

        set_identity(user_agent)

    def list_10k_filings(self, ticker: str) -> list[_FilingMeta]:
        from edgar import Company

        company = Company(ticker)
        filings = company.get_filings(form="10-K")
        out: list[_FilingMeta] = []
        for f in filings:
            out.append(
                _FilingMeta(
                    cik=str(company.cik).zfill(10),
                    ticker=ticker.upper(),
                    accession_number=str(f.accession_no),
                    filing_date=str(f.filing_date),
                    period_of_report=str(f.period_of_report),
                )
            )
        return out

    def fetch_html(self, accession_number: str) -> str:
        from edgar import find

        filing = find(accession_number)
        # `.html()` returns the primary document HTML; for 10-Ks this is the
        # full filing.
        return str(filing.html())

    def fetch_xbrl_facts(self, accession_number: str) -> list[dict]:
        """Fetch structured XBRL facts via edgartools.

        Returns one dict per fact, in the schema the Protocol documents.
        Returns [] if the filing has no XBRL data (very old filings).

        Implementation: edgartools' `xbrl().facts` is a `FactsView` that
        doesn't iterate directly. We use `.to_dataframe()` to get a flat
        table and walk it. Dimensions live in per-axis columns named
        `dim_<namespace>_<LocalName>` (e.g.
        `dim_srt_StatementGeographicalAxis`); multi-dimensional facts
        populate multiple of these columns. We rebuild the dict-of-axes
        from those columns row by row.
        """
        from edgar import find

        filing = find(accession_number)
        try:
            xbrl_data = filing.xbrl()
        except Exception as exc:
            logger.warning("xbrl() raised for %s: %s", accession_number, exc)
            return []
        if xbrl_data is None:
            return []

        try:
            df = xbrl_data.facts.to_dataframe()
        except Exception as exc:
            logger.warning("to_dataframe failed for %s: %s", accession_number, exc)
            return []

        return _df_to_raw_facts(df)


# ─── helpers ────────────────────────────────────────────────────────────────


def _parse_iso_date(s: str | date) -> date:
    """Parse an ISO date string, or pass through if already a date."""
    if isinstance(s, date):
        return s
    return date.fromisoformat(s)


def _df_to_raw_facts(df: object) -> list[dict]:
    """Convert an edgartools `xbrl().facts.to_dataframe()` DataFrame into
    our raw fact dict schema.

    Public-ish (single underscore) so unit tests can construct a synthetic
    DataFrame and verify the column-to-dimension mapping without hitting
    edgartools or EDGAR.

    Schema produced (per fact):
        concept, value, unit, period_start, period_end, context_id, dimensions

    Dimension columns in the input are named `dim_<ns>_<LocalName>`, where
    `<ns>` may itself contain hyphens (e.g. `us-gaap`). We split on the
    FIRST underscore after `dim_` to recover the QName.

    Non-numeric facts (where `numeric_value` is NaN) are dropped — they're
    text disclosures that belong in the chunk corpus, not the fact table.
    """
    import pandas as pd

    # Build column -> QName mapping for dimension columns.
    dim_cols: dict[str, str] = {}
    for col in df.columns:  # type: ignore[attr-defined]
        if not str(col).startswith("dim_"):
            continue
        rest = str(col)[len("dim_"):]
        ns, sep, local = rest.partition("_")
        if not sep or not local:
            continue
        dim_cols[col] = f"{ns}:{local}"

    out: list[dict] = []
    for _, row in df.iterrows():  # type: ignore[attr-defined]
        # Drop non-numeric facts.
        nv = row.get("numeric_value")
        if nv is None or pd.isna(nv):
            continue

        # Period: instants populate `period_instant`; durations populate
        # `period_start` / `period_end`. Normalize to start/end equal for
        # instants so the downstream XBRLFact model is consistent.
        period_type = row.get("period_type", "")
        if period_type == "instant":
            instant = row.get("period_instant")
            if instant is None or pd.isna(instant):
                continue
            period_start = period_end = str(instant)
        else:
            ps = row.get("period_start")
            pe = row.get("period_end")
            if ps is None or pd.isna(ps) or pe is None or pd.isna(pe):
                continue
            period_start = str(ps)
            period_end = str(pe)

        # Collect every populated dim_* column for this row.
        dimensions: dict[str, str] = {}
        for col, qname in dim_cols.items():
            val = row.get(col)
            if val is not None and not pd.isna(val):
                dimensions[qname] = str(val)

        unit = row.get("unit_ref")
        unit_str = "" if (unit is None or pd.isna(unit)) else str(unit)

        out.append(
            {
                "concept": str(row.get("concept", "")),
                "value": str(row.get("value", "")),
                "unit": unit_str,
                "period_start": period_start,
                "period_end": period_end,
                "context_id": str(row.get("context_ref", "")),
                "dimensions": dimensions,
            }
        )
    return out
