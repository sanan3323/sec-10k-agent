"""sec_10k_agent.eval — golden-set evaluation harness (Phase 4).

Deterministic retrieval metrics plus an LLM-as-judge for faithfulness and
correctness. See docs/architecture.md and data/eval/SCHEMA.md.
"""

from __future__ import annotations

from sec_10k_agent.eval.dataset import CiteSpec, EvalFilters, GoldenItem, load_golden_set
from sec_10k_agent.eval.judge import Judge, JudgeScore, build_judge
from sec_10k_agent.eval.report import format_markdown, to_json
from sec_10k_agent.eval.retrieval_metrics import context_recall, hit_at_k, reciprocal_rank
from sec_10k_agent.eval.runner import BucketAggregate, EvalReport, ItemResult, run_eval

__all__ = [
    "BucketAggregate",
    "CiteSpec",
    "EvalFilters",
    "EvalReport",
    "GoldenItem",
    "ItemResult",
    "Judge",
    "JudgeScore",
    "build_judge",
    "context_recall",
    "format_markdown",
    "hit_at_k",
    "load_golden_set",
    "reciprocal_rank",
    "run_eval",
    "to_json",
]
