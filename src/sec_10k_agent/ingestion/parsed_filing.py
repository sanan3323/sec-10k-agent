"""Parsed filing intermediate.

The output of the parser. Cached to disk at
`data/processed/parsed/{accession_number}.json` so re-chunking is fast and
re-parsing only happens when the parser version changes.

The chunker reads from here, never from raw HTML. That separation matters:
parser fixes (Item boundary detection, footnote handling, table extraction)
won't all be perfect on the first pass, and we don't want every parser fix
to trigger a full re-embed of the corpus.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from sec_10k_agent.ingestion.models import Filing


class Section(BaseModel):
    """One Item section of a 10-K, pre-split into paragraphs."""

    item: str = Field(..., description="Normalized item code, e.g. '1A', '7'")
    title: str = Field(..., description="Human-readable, e.g. 'Risk Factors'")
    paragraphs: list[str] = Field(..., description="Paragraphs in document order")

    @property
    def display_section(self) -> str:
        """The form used in citations and chunk metadata, e.g. 'Item 1A'."""
        return f"Item {self.item}"


class ParsedFiling(BaseModel):
    """The parser's output for one filing, ready for chunking."""

    filing: Filing
    sections: list[Section]
    parsed_at: datetime
    parser_version: str = Field(
        ...,
        description=(
            "Bumped when parser logic changes in ways that should invalidate "
            "the cache. Compared on read; mismatched files are re-parsed."
        ),
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Warnings from parsing, e.g. 'Item 1C not found'",
    )


# Cache I/O


def cache_path(processed_dir: Path, accession_number: str) -> Path:
    return processed_dir / "parsed" / f"{accession_number}.json"


def write_to_cache(parsed: ParsedFiling, processed_dir: Path) -> Path:
    path = cache_path(processed_dir, parsed.filing.accession_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
    return path


def read_from_cache(
    processed_dir: Path,
    accession_number: str,
    expected_parser_version: str,
) -> ParsedFiling | None:
    """Return the cached ParsedFiling, or None if it's missing, corrupt, or
    has a stale parser_version."""
    path = cache_path(processed_dir, accession_number)
    if not path.exists():
        return None
    try:
        parsed = ParsedFiling.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if parsed.parser_version != expected_parser_version:
        return None
    return parsed
