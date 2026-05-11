"""Tests for the XBRL extractor.

The headline test is `test_greater_china_acceptance` — the architecture doc
says: if our parser can't represent "Apple's Greater China revenue in
FY2024" with the right dimensional axis, the parser is broken. We assert
exactly that against a synthetic XBRL fact set.

Uses a fake backend (no edgartools install needed for tests, no network).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from sec_10k_agent.ingestion import EdgarClient
from sec_10k_agent.ingestion.edgar_client import _FilingMeta
from sec_10k_agent.ingestion.models import Filing
from sec_10k_agent.ingestion.xbrl import XBRLExtractor, extract_all_cached

#Fake backend 


def _aapl_2024_facts() -> list[dict]:
    """Synthetic XBRL fact set covering the Greater China acceptance test
    plus a handful of representative facts.
    """
    period = {
        "period_start": "2023-10-01",
        "period_end": "2024-09-28",
    }
    return [
        # Total revenue (no dimensions)
        {
            "concept": "us-gaap:Revenues",
            "value": "391035000000",
            "unit": "USD",
            "context_id": "c-1",
            "dimensions": {},
            **period,
        },
        # Greater China revenue — the acceptance-test fact
        {
            "concept": "us-gaap:Revenues",
            "value": "66952000000",
            "unit": "USD",
            "context_id": "c-23",
            "dimensions": {"srt:StatementGeographicalAxis": "country:CN"},
            **period,
        },
        # Americas revenue
        {
            "concept": "us-gaap:Revenues",
            "value": "167045000000",
            "unit": "USD",
            "context_id": "c-21",
            "dimensions": {"srt:StatementGeographicalAxis": "srt:AmericasMember"},
            **period,
        },
        # iPhone product line
        {
            "concept": "us-gaap:Revenues",
            "value": "201183000000",
            "unit": "USD",
            "context_id": "c-31",
            "dimensions": {"us-gaap:ProductOrServiceAxis": "us-gaap:IPhoneMember"},
            **period,
        },
        # A balance-sheet item (instant context: start == end)
        {
            "concept": "us-gaap:Cash",
            "value": "29943000000",
            "unit": "USD",
            "context_id": "c-2",
            "dimensions": {},
            "period_start": "2024-09-28",
            "period_end": "2024-09-28",
        },
        # A non-numeric fact that should be skipped
        {
            "concept": "dei:DocumentType",
            "value": "10-K",
            "unit": "",
            "context_id": "c-1",
            "dimensions": {},
            **period,
        },
        # A malformed fact (unparseable date) — should be skipped, not crash
        {
            "concept": "us-gaap:Revenues",
            "value": "1000",
            "unit": "USD",
            "context_id": "c-99",
            "dimensions": {},
            "period_start": "garbage",
            "period_end": "garbage",
        },
    ]


class FakeBackend:
    def __init__(self) -> None:
        self.fetch_xbrl_calls: list[str] = []

    def set_identity(self, user_agent: str) -> None:
        pass

    def list_10k_filings(self, ticker: str) -> list[_FilingMeta]:
        return [
            _FilingMeta(
                cik="0000320193",
                ticker="AAPL",
                accession_number="0000320193-24-000123",
                filing_date="2024-11-01",
                period_of_report="2024-09-28",
            )
        ]

    def fetch_html(self, accession_number: str) -> str:
        return f"<html>{accession_number}</html>"

    def fetch_xbrl_facts(self, accession_number: str) -> list[dict]:
        self.fetch_xbrl_calls.append(accession_number)
        return _aapl_2024_facts()


#Fixtures


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "raw"
    d.mkdir()
    return d


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def client(cache_dir: Path, backend: FakeBackend) -> EdgarClient:
    return EdgarClient(
        user_agent="Test User test@example.com",
        rate_limit_per_sec=1000.0,
        cache_dir=cache_dir,
        backend=backend,
    )


@pytest.fixture
def downloaded_filing(client: EdgarClient) -> Filing:
    """Pre-download AAPL FY2024 so the XBRL test has a Filing in the cache."""
    return client.get_10k("AAPL", 2024)


#XBRL fetch + cache


def test_get_xbrl_facts_caches_to_disk(
    client: EdgarClient, downloaded_filing: Filing, cache_dir: Path
) -> None:
    client.get_xbrl_facts("AAPL", 2024)
    assert (cache_dir / "AAPL_2024_xbrl.json").exists()


def test_get_xbrl_facts_uses_cache_on_second_call(
    client: EdgarClient, downloaded_filing: Filing, backend: FakeBackend
) -> None:
    client.get_xbrl_facts("AAPL", 2024)
    client.get_xbrl_facts("AAPL", 2024)
    assert len(backend.fetch_xbrl_calls) == 1


def test_get_xbrl_facts_requires_filing_to_be_downloaded(client: EdgarClient) -> None:
    from sec_10k_agent.ingestion import FilingNotFound

    with pytest.raises(FilingNotFound):
        client.get_xbrl_facts("AAPL", 2024)


#Extractor


def test_extractor_returns_typed_xbrlfacts(
    client: EdgarClient, downloaded_filing: Filing
) -> None:
    extractor = XBRLExtractor(client)
    facts = extractor.extract(downloaded_filing)

    # 7 raw facts in; non-numeric ("10-K") and malformed-date ones get
    # dropped, so we expect 5.
    assert len(facts) == 5
    for fact in facts:
        assert isinstance(fact.value, Decimal)
        assert fact.ticker == "AAPL"
        assert fact.fiscal_year == 2024


def test_total_revenue_has_empty_dimensions(
    client: EdgarClient, downloaded_filing: Filing
) -> None:
    extractor = XBRLExtractor(client)
    facts = extractor.extract(downloaded_filing)

    totals = [
        f for f in facts
        if f.concept == "us-gaap:Revenues" and f.dimensions == {}
    ]
    assert len(totals) == 1
    assert totals[0].value == Decimal("391035000000")


def test_greater_china_acceptance(
    client: EdgarClient, downloaded_filing: Filing
) -> None:
    """Architecture-doc litmus test.

    Apple's Greater China revenue must come back as a Revenues fact at
    axis srt:StatementGeographicalAxis with member country:CN. If this
    fails, the dimensional capture is broken and structured retrieval in
    Phase 6 cannot work.
    """
    extractor = XBRLExtractor(client)
    facts = extractor.extract(downloaded_filing)

    matches = [
        f for f in facts
        if f.concept == "us-gaap:Revenues"
        and f.dimensions.get("srt:StatementGeographicalAxis") == "country:CN"
    ]
    assert len(matches) == 1, (
        f"Expected exactly one Greater China revenue fact; got {len(matches)}. "
        "If this fails, dimensional XBRL capture is broken."
    )
    assert matches[0].value == Decimal("66952000000")
    assert matches[0].unit == "USD"
    assert matches[0].period_end == date(2024, 9, 28)


def test_skips_non_numeric_facts(
    client: EdgarClient, downloaded_filing: Filing
) -> None:
    extractor = XBRLExtractor(client)
    facts = extractor.extract(downloaded_filing)

    # The DocumentType fact has a non-numeric value ("10-K") and must be skipped.
    assert not any(f.concept == "dei:DocumentType" for f in facts)


def test_skips_facts_with_unparseable_dates(
    client: EdgarClient, downloaded_filing: Filing
) -> None:
    extractor = XBRLExtractor(client)
    facts = extractor.extract(downloaded_filing)

    # The fact with garbage dates must be skipped, not crash extraction.
    assert all(f.context_id != "c-99" for f in facts)


def test_balance_sheet_instant_has_equal_start_end(
    client: EdgarClient, downloaded_filing: Filing
) -> None:
    extractor = XBRLExtractor(client)
    facts = extractor.extract(downloaded_filing)

    cash = [f for f in facts if f.concept == "us-gaap:Cash"]
    assert len(cash) == 1
    assert cash[0].period_start == cash[0].period_end


#End-to-end: extract_all_cached → parquet 


def test_extract_all_cached_writes_parquet(
    client: EdgarClient, downloaded_filing: Filing, tmp_path: Path
) -> None:
    import pandas as pd

    raw_dir = downloaded_filing.raw_html_path.parent
    processed_dir = tmp_path / "processed"
    n_filings, n_facts, failures = extract_all_cached(
        client, raw_dir, processed_dir
    )

    assert failures == []
    assert n_filings == 1
    assert n_facts == 5

    parquet_path = processed_dir / "xbrl.parquet"
    assert parquet_path.exists()

    df = pd.read_parquet(parquet_path)
    assert len(df) == 5
    # Round-trip the Greater China assertion via the parquet itself —
    # this confirms dimensions survive the parquet write.
    china = df[
        (df["concept"] == "us-gaap:Revenues")
        & df["dimensions"].apply(
            lambda d: d.get("srt:StatementGeographicalAxis") == "country:CN"
        )
    ]
    assert len(china) == 1


#DataFrame -> raw facts helper
# Tests the function inside the real edgartools adapter that converts
# `xbrl().facts.to_dataframe()` output into our raw fact dict schema.
# The DataFrame columns here mirror what edgartools 5.x actually returns.


def test_df_to_raw_facts_extracts_dimensions() -> None:
    import pandas as pd

    from sec_10k_agent.ingestion.edgar_client import _df_to_raw_facts

    df = pd.DataFrame(
        [
            # Greater China revenue — the acceptance test in DataFrame form.
            {
                "concept": "us-gaap:Revenues",
                "value": "66952000000",
                "numeric_value": 66952000000.0,
                "unit_ref": "USD",
                "period_type": "duration",
                "period_start": "2023-10-01",
                "period_end": "2024-09-28",
                "period_instant": None,
                "context_ref": "c-23",
                "dim_srt_StatementGeographicalAxis": "country:CN",
                "dim_us-gaap_StatementBusinessSegmentsAxis": None,
            },
            # Total revenue — no dimensions populated.
            {
                "concept": "us-gaap:Revenues",
                "value": "391035000000",
                "numeric_value": 391035000000.0,
                "unit_ref": "USD",
                "period_type": "duration",
                "period_start": "2023-10-01",
                "period_end": "2024-09-28",
                "period_instant": None,
                "context_ref": "c-1",
                "dim_srt_StatementGeographicalAxis": None,
                "dim_us-gaap_StatementBusinessSegmentsAxis": None,
            },
            # Balance-sheet item: instant period, both dims null.
            {
                "concept": "us-gaap:Cash",
                "value": "29943000000",
                "numeric_value": 29943000000.0,
                "unit_ref": "USD",
                "period_type": "instant",
                "period_start": None,
                "period_end": None,
                "period_instant": "2024-09-28",
                "context_ref": "c-2",
                "dim_srt_StatementGeographicalAxis": None,
                "dim_us-gaap_StatementBusinessSegmentsAxis": None,
            },
            # Non-numeric fact (DocumentType) — should be dropped.
            {
                "concept": "dei:DocumentType",
                "value": "10-K",
                "numeric_value": None,
                "unit_ref": None,
                "period_type": "duration",
                "period_start": "2023-10-01",
                "period_end": "2024-09-28",
                "period_instant": None,
                "context_ref": "c-1",
                "dim_srt_StatementGeographicalAxis": None,
                "dim_us-gaap_StatementBusinessSegmentsAxis": None,
            },
        ]
    )

    facts = _df_to_raw_facts(df)

    # Non-numeric fact dropped.
    assert len(facts) == 3

    # Greater China line came through with the right axis qname.
    china = [
        f for f in facts
        if f["concept"] == "us-gaap:Revenues"
        and f["dimensions"].get("srt:StatementGeographicalAxis") == "country:CN"
    ]
    assert len(china) == 1
    assert china[0]["value"] == "66952000000"
    assert china[0]["unit"] == "USD"
    assert china[0]["period_end"] == "2024-09-28"

    # Total revenue: no dimensions on a fact with all dim_* columns null.
    total = [
        f for f in facts
        if f["concept"] == "us-gaap:Revenues" and f["dimensions"] == {}
    ]
    assert len(total) == 1

    # Instant period: start and end equal.
    cash = [f for f in facts if f["concept"] == "us-gaap:Cash"]
    assert len(cash) == 1
    assert cash[0]["period_start"] == cash[0]["period_end"] == "2024-09-28"


def test_df_to_raw_facts_handles_qname_with_hyphenated_namespace() -> None:
    """The hyphen in `us-gaap` is the trickiest column-name parse case.
    Make sure it survives the split."""
    import pandas as pd

    from sec_10k_agent.ingestion.edgar_client import _df_to_raw_facts

    df = pd.DataFrame(
        [
            {
                "concept": "us-gaap:Revenues",
                "value": "201183000000",
                "numeric_value": 201183000000.0,
                "unit_ref": "USD",
                "period_type": "duration",
                "period_start": "2023-10-01",
                "period_end": "2024-09-28",
                "period_instant": None,
                "context_ref": "c-31",
                "dim_us-gaap_ProductOrServiceAxis": "us-gaap:IPhoneMember",
            }
        ]
    )

    facts = _df_to_raw_facts(df)
    assert len(facts) == 1
    # The namespace prefix `us-gaap` (with hyphen) must come through intact.
    assert facts[0]["dimensions"] == {
        "us-gaap:ProductOrServiceAxis": "us-gaap:IPhoneMember"
    }
