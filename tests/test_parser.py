"""Tests for the FilingParser.

Most tests work on synthetic text (parser.parse_text) so we don't need real
10-K HTML or BeautifulSoup parsing. One test covers HTML-to-text via a tiny
synthetic HTML doc to exercise that path.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from sec_10k_agent.ingestion.models import Filing
from sec_10k_agent.ingestion.parsed_filing import (
    read_from_cache,
    write_to_cache,
)
from sec_10k_agent.ingestion.parser import FilingParser


@pytest.fixture
def filing(tmp_path: Path) -> Filing:
    html_path = tmp_path / "AAPL_2024.html"
    html_path.write_text("<html><body>placeholder</body></html>", encoding="utf-8")
    return Filing(
        cik="0000320193",
        ticker="AAPL",
        fiscal_year=2024,
        filing_date=date(2024, 11, 1),
        period_of_report=date(2024, 9, 28),
        accession_number="0000320193-24-000123",
        raw_html_path=html_path,
    )


@pytest.fixture
def parser() -> FilingParser:
    return FilingParser()


# Section detection


def _make_section(item: str, body: str) -> str:
    """Build a section block with header line + body."""
    return f"Item {item}.\n\n{body}\n\n"


def test_extracts_known_items_from_text(parser: FilingParser, filing: Filing) -> None:
    body_1 = "Apple designs and sells consumer electronics. " * 50
    body_1a = "We face risks related to global supply chains. " * 50
    body_7 = "Net sales increased year over year. " * 50
    text = _make_section("1", body_1) + _make_section("1A", body_1a) + _make_section("7", body_7)

    parsed = parser.parse_text(text, filing)

    items = [s.item for s in parsed.sections]
    assert items == ["1", "1A", "7"]
    assert parsed.sections[0].title == "Business"
    assert parsed.sections[1].title == "Risk Factors"


def test_section_paragraphs_are_split(parser: FilingParser, filing: Filing) -> None:
    text = (
        "Item 1A.\n\n"
        + ("First paragraph about supply chain risk. " * 10)
        + "\n\n"
        + ("Second paragraph about regulatory risk. " * 10)
        + "\n\n"
        + ("Third paragraph about cybersecurity. " * 10)
        + "\n\n"
    )
    parsed = parser.parse_text(text, filing)
    assert len(parsed.sections) == 1
    assert len(parsed.sections[0].paragraphs) == 3


def test_strips_table_of_contents(parser: FilingParser, filing: Filing) -> None:
    """A TOC at the top has all Items packed close together; the real
    sections come later with real content. The parser should pick the
    real ones, not the TOC entries.
    """
    toc = "Table of Contents\nItem 1. Business 5\nItem 1A. Risk Factors 12\nItem 7. MD&A 35\n"
    real_body = (
        "Item 1.\n\n"
        + "Real business section. " * 100
        + "\n\nItem 1A.\n\n"
        + "Real risk factors. " * 100
        + "\n\nItem 7.\n\n"
        + "Real MD&A. " * 100
    )
    text = toc + "\n\n" + real_body

    parsed = parser.parse_text(text, filing)

    # All three items are extracted, and their bodies contain the real
    # content, not the TOC strings.
    items = {s.item: " ".join(s.paragraphs) for s in parsed.sections}
    assert "Real business section." in items["1"]
    assert "Real risk factors." in items["1A"]
    assert "Real MD&A." in items["7"]


def test_missing_item_recorded_in_notes(parser: FilingParser, filing: Filing) -> None:
    text = _make_section("1", "Just business stuff. " * 50)
    parsed = parser.parse_text(text, filing)
    notes_blob = " | ".join(parsed.notes)
    assert "1A" in notes_blob  # missing Item 1A is flagged


def test_handles_no_headers_gracefully(parser: FilingParser, filing: Filing) -> None:
    parsed = parser.parse_text("Just some text with no item headers at all.", filing)
    assert parsed.sections == []
    assert any("no Item headers" in n for n in parsed.notes)


# HTML-to-text path


def test_parse_html_strips_scripts_and_styles(parser: FilingParser, filing: Filing) -> None:
    html = (
        """
    <html><head><style>body{color:red}</style></head>
    <body>
      <script>alert("x")</script>
      <p>Item 1.</p>
      <p>"""
        + ("Apple designs consumer electronics. " * 50)
        + """</p>
    </body></html>
    """
    )
    parsed = parser.parse_html(html, filing)
    items = [s.item for s in parsed.sections]
    assert "1" in items
    full_text = " ".join(parsed.sections[0].paragraphs)
    assert "alert" not in full_text  # script content gone
    assert "color:red" not in full_text  # style content gone


# Cache I/O
def test_parsed_filing_round_trip(parser: FilingParser, filing: Filing, tmp_path: Path) -> None:
    text = _make_section("1A", "Risk language. " * 50)
    parsed = parser.parse_text(text, filing)

    write_to_cache(parsed, tmp_path)
    loaded = read_from_cache(tmp_path, filing.accession_number, parsed.parser_version)

    assert loaded is not None
    assert loaded.filing.accession_number == filing.accession_number
    assert [s.item for s in loaded.sections] == [s.item for s in parsed.sections]


def test_stale_parser_version_returns_none(
    parser: FilingParser, filing: Filing, tmp_path: Path
) -> None:
    text = _make_section("1A", "Risk language. " * 50)
    parsed = parser.parse_text(text, filing)
    write_to_cache(parsed, tmp_path)

    loaded = read_from_cache(tmp_path, filing.accession_number, "9.9.9-newer")
    assert loaded is None


def test_corrupt_cache_returns_none(filing: Filing, tmp_path: Path) -> None:
    bad_path = tmp_path / "parsed" / f"{filing.accession_number}.json"
    bad_path.parent.mkdir(parents=True)
    bad_path.write_text("not json", encoding="utf-8")
    loaded = read_from_cache(tmp_path, filing.accession_number, "1.0.0")
    assert loaded is None


def test_missing_cache_returns_none(filing: Filing, tmp_path: Path) -> None:
    loaded = read_from_cache(tmp_path, filing.accession_number, "1.0.0")
    assert loaded is None
