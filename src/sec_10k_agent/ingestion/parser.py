"""10-K parser.

Reads cached 10-K HTML from disk, extracts clean text, regex-detects Item
section boundaries, splits each section into paragraphs, and returns a
`ParsedFiling`.

Approach: text-based, not HTML-structure-based. We strip the HTML to plain
text first, then operate purely on text. This is more robust across years
and filers than walking the HTML DOM, which varies wildly.

How section detection works:
- Find every "Item N" or "Item NA" header in the text.
- Each filing has a Table of Contents near the top that lists every Item.
  We distinguish TOC entries from real section headers by the gap to the
  next Item header: TOC entries are tightly packed (a few hundred chars
  apart), real sections are far apart (usually thousands of chars).
- For each Item, the real section starts at the LAST occurrence of that
  Item's header, since the TOC always comes first in document order.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from bs4 import BeautifulSoup

from sec_10k_agent.ingestion.models import Filing
from sec_10k_agent.ingestion.parsed_filing import ParsedFiling, Section

logger = logging.getLogger(__name__)


# Bump when parsing logic changes in a way that invalidates cached output.
PARSER_VERSION = "1.0.0"


# Items we extract. The set covers every 10-K Item that exists across our
# scope years. Items 1B (unresolved staff comments) and 1C (cybersecurity,
# 2023+) are included even though they're frequently empty, because their
# absence is informative.
KNOWN_ITEMS: tuple[tuple[str, str], ...] = (
    ("1", "Business"),
    ("1A", "Risk Factors"),
    ("1B", "Unresolved Staff Comments"),
    ("1C", "Cybersecurity"),
    ("2", "Properties"),
    ("3", "Legal Proceedings"),
    ("4", "Mine Safety Disclosures"),
    ("5", "Market for Registrant's Common Equity"),
    ("6", "Reserved"),
    ("7", "Management's Discussion and Analysis"),
    ("7A", "Quantitative and Qualitative Disclosures About Market Risk"),
    ("8", "Financial Statements and Supplementary Data"),
    ("9", "Changes in and Disagreements With Accountants"),
    ("9A", "Controls and Procedures"),
    ("9B", "Other Information"),
    ("9C", "Foreign Jurisdictions That Prevent Inspections"),
    ("10", "Directors, Executive Officers and Corporate Governance"),
    ("11", "Executive Compensation"),
    ("12", "Security Ownership"),
    ("13", "Certain Relationships and Related Transactions"),
    ("14", "Principal Accountant Fees and Services"),
    ("15", "Exhibits and Financial Statement Schedules"),
    ("16", "Form 10-K Summary"),
)
_ITEM_TITLES: dict[str, str] = dict(KNOWN_ITEMS)
_ITEM_ORDER: list[str] = [code for code, _ in KNOWN_ITEMS]


# Matches lines that start (after optional horizontal whitespace) with
# "Item NN" or "Item NNA". Case-insensitive. Captures the item code and
# the rest of the header line as a title hint.
#
# Important: only horizontal whitespace ([ \t]) is allowed between tokens.
# `\s` would match newlines, which would let the regex jump out of the
# header line and capture body content into the title group.
_ITEM_HEADER_RE = re.compile(
    r"(?im)(?:^|[.!?]\s+)[ \t]*(?:PART\s+[I-V]+[ \t,]+)?ITEM\s+([1-9]\d?[A-C]?)\b[ \t.:]*(.*)$"
)

# Minimum chars between two consecutive Item headers for the gap to count
# as "real content" rather than a Table of Contents entry. 500 is enough
# to filter most TOCs; real sections are typically thousands of chars.
_TOC_GAP_THRESHOLD = 100


class FilingParser:
    """Parses 10-K filings to ParsedFiling."""

    def parse_filing(self, filing: Filing) -> ParsedFiling:
        """Read the filing's cached HTML and parse it."""
        html = filing.raw_html_path.read_text(encoding="utf-8")
        return self.parse_html(html, filing)

    def parse_html(self, html: str, filing: Filing) -> ParsedFiling:
        """Parse from an HTML string. Public for testing."""
        text = _html_to_text(html)
        return self.parse_text(text, filing)

    def parse_text(self, text: str, filing: Filing) -> ParsedFiling:
        """Parse from already-cleaned text. Public for testing."""
        sections, notes = _extract_sections(text)
        return ParsedFiling(
            filing=filing,
            sections=sections,
            parsed_at=datetime.now(UTC),
            parser_version=PARSER_VERSION,
            notes=notes,
        )


# HTML to text


def _html_to_text(html: str) -> str:
    """Extract clean plain text from 10-K HTML."""
    soup = BeautifulSoup(html, "lxml")

    """Decompose noisy tags"""
    for tag in soup(["script", "style", "head", "title"]):
        tag.decompose()

    for tag in soup.find_all(["div", "p", "tr", "br"]):
        tag.insert_before("\n")
        tag.insert_after("\n")

    text = soup.get_text(separator="\n", strip=True)
    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize line endings
    text = re.sub(r"\r\n?", "\n", text)

    return text.strip()


# Section detection


def _extract_sections(text: str) -> tuple[list[Section], list[str]]:
    """Find Item-level sections in the text.

    Returns (sections, notes). `notes` collects warnings about missing or
    suspicious sections — surfaced in `ParsedFiling.notes` for debugging.
    """
    notes: list[str] = []

    # All header occurrences with their start position in the text.
    occurrences: list[tuple[int, str, str]] = []
    for match in _ITEM_HEADER_RE.finditer(text):
        item_code = match.group(1).upper()
        if item_code not in _ITEM_TITLES:
            continue
        title_on_line = match.group(2).strip()
        occurrences.append((match.start(), item_code, title_on_line))

    if not occurrences:
        notes.append("no Item headers found in text")
        return [], notes

    # Pick the "real" occurrence of each item: the LAST occurrence whose
    # gap to the next-found header (any item) is wide enough to look like
    # real content. This filters TOC entries, which are tightly packed.
    real_starts = _pick_real_occurrences(occurrences)

    # Build sections by slicing text between real starts in document order.
    real_starts_sorted = sorted(real_starts.items(), key=lambda kv: kv[1][0])

    sections: list[Section] = []
    for i, (item_code, (start_pos, _title_on_line)) in enumerate(real_starts_sorted):
        end_pos = real_starts_sorted[i + 1][1][0] if i + 1 < len(real_starts_sorted) else len(text)
        body = text[start_pos:end_pos]
        # Trim the header line itself from the body.
        body = re.sub(_ITEM_HEADER_RE, "", body, count=1).lstrip()
        paragraphs = _split_paragraphs(body)
        if not paragraphs:
            notes.append(f"Item {item_code} found but body is empty")
            continue
        title = _ITEM_TITLES[item_code]
        sections.append(Section(item=item_code, title=title, paragraphs=paragraphs))

    # Note any expected items we didn't extract.
    extracted = {s.item for s in sections}
    for code in _ITEM_ORDER:
        if code not in extracted:
            notes.append(f"Item {code} not found")

    return sections, notes


def _pick_real_occurrences(
    occurrences: list[tuple[int, str, str]],
) -> dict[str, tuple[int, str]]:
    """For each Item code, return (start_pos, title_on_line) of its real
    section header — i.e. the last occurrence whose distance to the next
    occurrence (of any item) is > _TOC_GAP_THRESHOLD.

    If no occurrence of an item has a wide enough gap, that item is treated
    as missing.
    """
    real: dict[str, tuple[int, str]] = {}
    for i, (pos, item_code, title_on_line) in enumerate(occurrences):
        next_pos = occurrences[i + 1][0] if i + 1 < len(occurrences) else len(occurrences) * 10**9
        gap = next_pos - pos
        if gap > _TOC_GAP_THRESHOLD:
            # Last-write-wins: a later wide-gap occurrence overwrites an
            # earlier one. The actual content always follows the TOC.
            real[item_code] = (pos, title_on_line)
    return real


# Paragraph splitting


def _split_paragraphs(text: str) -> list[str]:
    """Split section body into paragraphs.

    Paragraphs are separated by 2+ newlines. Within a paragraph, single
    newlines (often soft-wrapping from the HTML extraction) are folded into
    spaces. Empty paragraphs are dropped.
    """
    raw = re.split(r"\n{2,}", text)
    out: list[str] = []
    for chunk in raw:
        cleaned = re.sub(r"\s+", " ", chunk).strip()
        if cleaned:
            out.append(cleaned)
    return out


# Convenience for the CLI


def parse_all_cached(
    raw_dir: Path,
    processed_dir: Path,
    *,
    force: bool = False,
) -> tuple[int, int, list[tuple[str, str]]]:
    """Parse every filing in `raw_dir/*.json` whose ParsedFiling is missing
    or has a stale parser version. Returns (parsed, skipped, failures).
    Used by the `sec10k parse` CLI command.
    """
    from sec_10k_agent.ingestion.parsed_filing import (
        read_from_cache,
        write_to_cache,
    )

    parser = FilingParser()
    parsed_count = 0
    skipped = 0
    failures: list[tuple[str, str]] = []

    for meta_path in sorted(raw_dir.glob("*.json")):
        try:
            filing = Filing.model_validate_json(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append((meta_path.name, f"unreadable filing metadata: {exc}"))
            continue

        if not force:
            cached = read_from_cache(processed_dir, filing.accession_number, PARSER_VERSION)
            if cached is not None:
                skipped += 1
                continue

        try:
            parsed = parser.parse_filing(filing)
            write_to_cache(parsed, processed_dir)
            parsed_count += 1
        except Exception as exc:
            failures.append((filing.accession_number, str(exc)))

    return parsed_count, skipped, failures
