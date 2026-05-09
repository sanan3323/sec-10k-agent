"""Tests for the Chunker.

Uses WordCountTokenCounter (deterministic, no network) so CI doesn't pull
the BGE tokenizer. Token thresholds are scaled down accordingly.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from sec_10k_agent.ingestion.chunker import Chunker
from sec_10k_agent.ingestion.models import Filing
from sec_10k_agent.ingestion.parsed_filing import ParsedFiling, Section
from sec_10k_agent.ingestion.tokenizer import WordCountTokenCounter


@pytest.fixture
def filing(tmp_path: Path) -> Filing:
    html_path = tmp_path / "AAPL_2024.html"
    html_path.write_text("placeholder", encoding="utf-8")
    return Filing(
        cik="0000320193",
        ticker="AAPL",
        fiscal_year=2024,
        filing_date=date(2024, 11, 1),
        period_of_report=date(2024, 9, 28),
        accession_number="0000320193-24-000123",
        raw_html_path=html_path,
    )


def _make_parsed(filing: Filing, sections: list[Section]) -> ParsedFiling:
    return ParsedFiling(
        filing=filing,
        sections=sections,
        parsed_at=datetime.now(UTC),
        parser_version="test",
    )


def _chunker() -> Chunker:
    # Thresholds in WORDS for tests. Roughly mirrors BGE token ratios.
    return Chunker(
        token_counter=WordCountTokenCounter(),
        max_tokens=50,
        target_tokens=40,
        min_tokens=10,
    )


# Basic shape
def test_one_paragraph_one_chunk(filing: Filing) -> None:
    section = Section(
        item="1A",
        title="Risk Factors",
        paragraphs=["A medium length paragraph about supply chain risks. " * 4],
    )
    chunks = _chunker().chunk(_make_parsed(filing, [section]))
    assert len(chunks) == 1
    assert chunks[0].section == "Item 1A"
    assert chunks[0].section_title == "Risk Factors"
    assert chunks[0].ticker == "AAPL"


def test_chunk_id_format(filing: Filing) -> None:
    section = Section(
        item="1A",
        title="Risk Factors",
        paragraphs=["A medium length paragraph. " * 4],
    )
    chunks = _chunker().chunk(_make_parsed(filing, [section]))
    expected = f"{filing.accession_number}__Item_1A__0000"
    assert chunks[0].chunk_id == expected


# Big paragraph splitting


def test_oversized_paragraph_splits_at_sentences(filing: Filing) -> None:
    # Build a paragraph with many sentences so it exceeds max_tokens (50 words).
    sentences = [f"This is sentence number {i} and it has a few words in it." for i in range(20)]
    big_paragraph = " ".join(sentences)
    section = Section(item="1A", title="Risk Factors", paragraphs=[big_paragraph])
    chunks = _chunker().chunk(_make_parsed(filing, [section]))

    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 50  # max_tokens hard cap


def test_split_pieces_are_in_order(filing: Filing) -> None:
    sentences = [f"Sentence number {i:02d} contains a few words here." for i in range(10)]
    big = " ".join(sentences)
    section = Section(item="1A", title="Risk Factors", paragraphs=[big])
    chunks = _chunker().chunk(_make_parsed(filing, [section]))

    # First chunk's first sentence should be the first sentence;
    # last chunk's last sentence should be the last sentence.
    assert "number 00" in chunks[0].text
    assert "number 09" in chunks[-1].text


# Tiny-paragraph merging


def test_tiny_paragraphs_merge_forward(filing: Filing) -> None:
    section = Section(
        item="1A",
        title="Risk Factors",
        paragraphs=[
            "Tiny first.",  # 2 words, below min
            "A medium length follow-up paragraph that has plenty of content.",
        ],
    )
    chunks = _chunker().chunk(_make_parsed(filing, [section]))
    assert len(chunks) == 1
    assert "Tiny first." in chunks[0].text
    assert "follow-up" in chunks[0].text


def test_tiny_paragraph_does_not_merge_across_section_boundary(filing: Filing) -> None:
    sec_1 = Section(
        item="1",
        title="Business",
        paragraphs=["Tiny ending."],  # below min, but it's the last paragraph in section
    )
    sec_1a = Section(
        item="1A",
        title="Risk Factors",
        paragraphs=["A medium paragraph in the next section. " * 2],
    )
    chunks = _chunker().chunk(_make_parsed(filing, [sec_1, sec_1a]))

    # Tiny paragraph in Item 1 stays in Item 1, even though it's below min.
    item_1_chunks = [c for c in chunks if c.section == "Item 1"]
    item_1a_chunks = [c for c in chunks if c.section == "Item 1A"]
    assert len(item_1_chunks) == 1
    assert "Tiny ending." in item_1_chunks[0].text
    assert "Tiny ending." not in item_1a_chunks[0].text


# prev/next chain


def test_prev_next_chain_is_correct(filing: Filing) -> None:
    section = Section(
        item="1A",
        title="Risk Factors",
        paragraphs=[
            "First paragraph with enough words to be its own chunk. " * 2,
            "Second paragraph with enough words to be its own chunk. " * 2,
            "Third paragraph with enough words to be its own chunk. " * 2,
        ],
    )
    chunks = _chunker().chunk(_make_parsed(filing, [section]))
    assert len(chunks) == 3

    assert chunks[0].prev_chunk_id is None
    assert chunks[0].next_chunk_id == chunks[1].chunk_id
    assert chunks[1].prev_chunk_id == chunks[0].chunk_id
    assert chunks[1].next_chunk_id == chunks[2].chunk_id
    assert chunks[2].prev_chunk_id == chunks[1].chunk_id
    assert chunks[2].next_chunk_id is None


def test_prev_next_spans_section_boundaries(filing: Filing) -> None:
    """The prev/next chain spans Item boundaries — at retrieval time we
    expand context across sections in document order."""
    sec_1 = Section(
        item="1",
        title="Business",
        paragraphs=["Business paragraph with several words to make a chunk. " * 2],
    )
    sec_1a = Section(
        item="1A",
        title="Risk Factors",
        paragraphs=["Risk paragraph with several words to make a chunk. " * 2],
    )
    chunks = _chunker().chunk(_make_parsed(filing, [sec_1, sec_1a]))

    assert len(chunks) == 2
    assert chunks[0].next_chunk_id == chunks[1].chunk_id
    assert chunks[1].prev_chunk_id == chunks[0].chunk_id


# Invariants


def test_no_chunk_exceeds_max_tokens(filing: Filing) -> None:
    # Mix of long and short paragraphs.
    section = Section(
        item="1A",
        title="Risk Factors",
        paragraphs=[
            ("Sentence here. " * 30).strip(),  # very long
            "Tiny.",
            ("Medium length paragraph. " * 10).strip(),
            ("Sentence here. " * 100).strip(),  # absurdly long
        ],
    )
    chunks = _chunker().chunk(_make_parsed(filing, [section]))
    assert all(c.token_count <= 50 for c in chunks), [(c.chunk_id, c.token_count) for c in chunks]


def test_constructor_validates_thresholds() -> None:
    tc = WordCountTokenCounter()
    with pytest.raises(ValueError):
        Chunker(tc, max_tokens=10, target_tokens=20, min_tokens=5)  # target > max
    with pytest.raises(ValueError):
        Chunker(tc, max_tokens=100, target_tokens=50, min_tokens=80)  # min > target
    with pytest.raises(ValueError):
        Chunker(tc, max_tokens=100, target_tokens=50, min_tokens=0)  # min == 0
