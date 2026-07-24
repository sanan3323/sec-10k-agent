"""Tests for the golden-set schema and loader, including the real seed file."""

from __future__ import annotations

import pytest

from sec_10k_agent.eval.dataset import GoldenItem, load_golden_set


def test_real_golden_set_loads_and_validates() -> None:
    items = load_golden_set()  # default path: data/eval/golden.jsonl
    assert len(items) >= 20
    ids = [it.id for it in items]
    assert len(ids) == len(set(ids)), "ids must be unique"
    # All three buckets represented.
    buckets = {it.bucket for it in items}
    assert buckets == {"single_fact", "synthesis", "temporal"}
    # Every item has a grading target.
    assert all(it.grading_reference() for it in items)
    # Corpus is AAPL/NVDA/JPM only — no stray tickers in filters/must_cite.
    allowed = {"AAPL", "NVDA", "JPM"}
    for it in items:
        for spec in it.must_cite:
            assert spec.ticker in allowed


def test_seed_has_negative_controls() -> None:
    items = load_golden_set()
    negatives = [it for it in items if it.is_negative_control()]
    assert negatives, "expected at least one negative-control item"
    assert all(not it.must_cite for it in negatives)


def test_single_fact_requires_answer() -> None:
    with pytest.raises(ValueError, match="needs an `answer`"):
        GoldenItem(id="x", bucket="single_fact", question="q?")


def test_synthesis_requires_rubric() -> None:
    with pytest.raises(ValueError, match="needs an `answer_rubric`"):
        GoldenItem(id="x", bucket="synthesis", question="q?", answer="not a rubric field")


def test_loader_skips_comments_and_blanks(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "g.jsonl"
    p.write_text(
        '# a comment\n\n{"id": "a", "bucket": "single_fact", "question": "q?", "answer": "yes"}\n',
        encoding="utf-8",
    )
    items = load_golden_set(p)
    assert len(items) == 1
    assert items[0].id == "a"


def test_loader_rejects_duplicate_ids(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "g.jsonl"
    line = '{"id": "dup", "bucket": "single_fact", "question": "q?", "answer": "a"}'
    p.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate id"):
        load_golden_set(p)


def test_loader_reports_bad_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "g.jsonl"
    p.write_text("{not valid json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_golden_set(p)
