"""Tests for the EDGAR client.

We use a fake backend that satisfies the `_EdgartoolsBackend` protocol so
none of these tests need network or the real `edgartools` package installed.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import pytest

from sec_10k_agent.ingestion import EdgarClient, FilingNotFound
from sec_10k_agent.ingestion.edgar_client import _FilingMeta
from sec_10k_agent.ingestion.rate_limiter import RateLimiter


# Test doubles
class FakeBackend:
    """In-memory backend. Records what was called for assertions."""

    def __init__(self, filings_by_ticker: dict[str, list[_FilingMeta]]) -> None:
        self._filings = filings_by_ticker
        self.identity_calls: list[str] = []
        self.list_calls: Counter[str] = Counter()
        self.fetch_calls: Counter[str] = Counter()

    def set_identity(self, user_agent: str) -> None:
        self.identity_calls.append(user_agent)

    def list_10k_filings(self, ticker: str) -> list[_FilingMeta]:
        self.list_calls[ticker] += 1
        if ticker.upper() not in self._filings:
            raise RuntimeError(f"unknown ticker in fake backend: {ticker}")
        return self._filings[ticker.upper()]

    def fetch_html(self, accession_number: str) -> str:
        self.fetch_calls[accession_number] += 1
        return f"<html><body>fake 10-K for {accession_number}</body></html>"


def _aapl_filings() -> list[_FilingMeta]:
    """A minimal slice of AAPL's 10-K history that covers a few of the years
    in our scope."""
    return [
        _FilingMeta(
            cik="0000320193",
            ticker="AAPL",
            accession_number="0000320193-25-000001",
            filing_date="2025-11-01",
            period_of_report="2025-09-27",
        ),
        _FilingMeta(
            cik="0000320193",
            ticker="AAPL",
            accession_number="0000320193-24-000123",
            filing_date="2024-11-01",
            period_of_report="2024-09-28",
        ),
        _FilingMeta(
            cik="0000320193",
            ticker="AAPL",
            accession_number="0000320193-23-000106",
            filing_date="2023-11-03",
            period_of_report="2023-09-30",
        ),
    ]


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "raw"
    d.mkdir()
    return d


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend({"AAPL": _aapl_filings()})


@pytest.fixture
def client(cache_dir: Path, backend: FakeBackend) -> EdgarClient:
    # Very high rate so tests don't sleep.
    return EdgarClient(
        user_agent="Test User test@example.com",
        rate_limit_per_sec=1000.0,
        cache_dir=cache_dir,
        backend=backend,
    )


# EdgarClient
def test_get_10k_returns_correct_fiscal_year(client: EdgarClient) -> None:
    filing = client.get_10k("AAPL", 2024)
    assert filing.ticker == "AAPL"
    assert filing.fiscal_year == 2024
    assert filing.period_of_report.year == 2024
    assert filing.accession_number == "0000320193-24-000123"


def test_get_10k_writes_html_and_metadata_to_cache(client: EdgarClient, cache_dir: Path) -> None:
    client.get_10k("AAPL", 2024)
    assert (cache_dir / "AAPL_2024.html").exists()
    assert (cache_dir / "AAPL_2024.json").exists()


def test_get_10k_uses_cache_on_second_call(client: EdgarClient, backend: FakeBackend) -> None:
    client.get_10k("AAPL", 2024)
    client.get_10k("AAPL", 2024)
    # First call: 1 list + 1 fetch. Second call: cache hit, no backend calls.
    assert backend.list_calls["AAPL"] == 1
    assert backend.fetch_calls["0000320193-24-000123"] == 1


def test_get_10k_normalizes_ticker_case(client: EdgarClient) -> None:
    filing = client.get_10k("aapl", 2024)
    assert filing.ticker == "AAPL"


def test_get_10k_raises_when_no_matching_year(client: EdgarClient) -> None:
    with pytest.raises(FilingNotFound) as exc_info:
        client.get_10k("AAPL", 1999)
    # Error message should list the years that ARE available — helps debugging.
    msg = str(exc_info.value)
    assert "2024" in msg


def test_get_10k_picks_latest_when_multiple_match(
    cache_dir: Path,
) -> None:
    """If a year has two filings (e.g. amended), pick the latest filing_date."""
    duplicate_year = [
        _FilingMeta(
            cik="0000320193",
            ticker="AAPL",
            accession_number="OLD-ORIGINAL",
            filing_date="2024-11-01",
            period_of_report="2024-09-28",
        ),
        _FilingMeta(
            cik="0000320193",
            ticker="AAPL",
            accession_number="NEW-AMENDED",
            filing_date="2025-02-15",
            period_of_report="2024-09-28",
        ),
    ]
    backend = FakeBackend({"AAPL": duplicate_year})
    client = EdgarClient(
        user_agent="Test test@example.com",
        rate_limit_per_sec=1000.0,
        cache_dir=cache_dir,
        backend=backend,
    )
    filing = client.get_10k("AAPL", 2024)
    assert filing.accession_number == "NEW-AMENDED"


def test_set_identity_called_once_at_init(backend: FakeBackend, cache_dir: Path) -> None:
    EdgarClient(
        user_agent="Jane Doe jane@example.com",
        rate_limit_per_sec=1000.0,
        cache_dir=cache_dir,
        backend=backend,
    )
    assert backend.identity_calls == ["Jane Doe jane@example.com"]


def test_corrupt_metadata_triggers_refetch(
    client: EdgarClient, cache_dir: Path, backend: FakeBackend
) -> None:
    client.get_10k("AAPL", 2024)
    # Corrupt the metadata file.
    (cache_dir / "AAPL_2024.json").write_text("not valid json", encoding="utf-8")
    client.get_10k("AAPL", 2024)
    # Backend was hit a second time because cache was unreadable.
    assert backend.list_calls["AAPL"] == 2


# RateLimiter
def test_rate_limiter_spaces_calls() -> None:
    # 10 calls at 50/sec should take >= 9 * 0.02 = 0.18s
    # (first call is free; only the gaps cost). We assert a generous lower
    # bound to avoid flakes on slow CI.
    limiter = RateLimiter(rate_per_sec=50.0)
    start = time.monotonic()
    for _ in range(10):
        limiter.wait()
    elapsed = time.monotonic() - start
    assert elapsed >= 9 * (1 / 50) * 0.9  # 10% slack for timer resolution


def test_rate_limiter_rejects_nonpositive_rate() -> None:
    with pytest.raises(ValueError):
        RateLimiter(rate_per_sec=0)
    with pytest.raises(ValueError):
        RateLimiter(rate_per_sec=-1.0)
