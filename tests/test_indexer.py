"""Tests for the indexer's pure frame-preparation helper (no DB, no model)."""

from __future__ import annotations

import pandas as pd
import pytest

from sec_10k_agent.ingestion.indexer import select_chunk_columns


def test_select_chunk_columns_keeps_known_order_drops_extra() -> None:
    df = pd.DataFrame(
        {
            "text": ["a"],
            "chunk_id": ["c1"],
            "ticker": ["AAPL"],
            "junk": [1],  # dropped
        }
    )
    out = select_chunk_columns(df)
    # CHUNK_COLUMNS order: chunk_id before ticker before text; junk removed.
    assert list(out.columns) == ["chunk_id", "ticker", "text"]
    assert "junk" not in out.columns


def test_select_chunk_columns_requires_text_and_chunk_id() -> None:
    with pytest.raises(ValueError, match="missing required column 'text'"):
        select_chunk_columns(pd.DataFrame({"chunk_id": ["c1"]}))
    with pytest.raises(ValueError, match="missing required column 'chunk_id'"):
        select_chunk_columns(pd.DataFrame({"text": ["a"]}))
