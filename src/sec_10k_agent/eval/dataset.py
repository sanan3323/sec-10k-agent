"""Golden eval set: schema and loader.

The on-disk format is documented in data/eval/SCHEMA.md — one JSON object per
line in golden.jsonl. These models are the typed target that file validates
against. Three buckets drive different scoring paths:

- single_fact: has a reference `answer`; judged for correctness against it.
- synthesis / temporal: open-ended; judged against `answer_rubric`.

`must_cite` lists chunks that retrieval MUST surface for the item to be
answerable; it powers the deterministic retrieval metrics (no LLM needed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from sec_10k_agent.config import get_settings

Bucket = Literal["single_fact", "synthesis", "temporal"]


class CiteSpec(BaseModel):
    """A chunk that must appear in retrieval. Unset fields are wildcards."""

    ticker: str | None = None
    fiscal_year: int | None = None
    section: str | None = None
    chunk_id: str | None = None


class EvalFilters(BaseModel):
    """Optional pre-filter hints. The router should reach the same set on its own."""

    ticker: str | None = None
    fiscal_year: int | None = None
    section: str | None = None


class GoldenItem(BaseModel):
    """One graded question in the golden set."""

    id: str
    bucket: Bucket
    question: str
    filters: EvalFilters | None = None
    answer: str | None = None
    answer_rubric: str | None = None
    must_cite: list[CiteSpec] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_grading_target(self) -> GoldenItem:
        # single_fact needs a reference answer; synthesis/temporal need a rubric.
        # A negative-control single_fact (must_cite empty) still carries an answer
        # describing the expected abstention, so this holds for it too.
        if self.bucket == "single_fact" and not self.answer:
            raise ValueError(f"{self.id}: single_fact item needs an `answer`")
        if self.bucket in ("synthesis", "temporal") and not self.answer_rubric:
            raise ValueError(f"{self.id}: {self.bucket} item needs an `answer_rubric`")
        return self

    def grading_reference(self) -> str:
        """The text the judge grades the answer against (answer or rubric)."""
        return self.answer or self.answer_rubric or ""

    def is_negative_control(self) -> bool:
        """True for items where the corpus has no answer and the agent should abstain."""
        return self.bucket == "single_fact" and not self.must_cite


def load_golden_set(path: Path | None = None) -> list[GoldenItem]:
    """Load and validate golden.jsonl. Blank lines and `#` comments are skipped."""
    if path is None:
        path = get_settings().eval_dir / "golden.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Golden set not found at {path}")

    items: list[GoldenItem] = []
    seen_ids: set[str] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON — {exc}") from exc
        item = GoldenItem.model_validate(obj)
        if item.id in seen_ids:
            raise ValueError(f"{path}:{lineno}: duplicate id {item.id!r}")
        seen_ids.add(item.id)
        items.append(item)
    return items
