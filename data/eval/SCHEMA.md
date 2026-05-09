# Golden eval set — schema

Phase 4 fills `golden.jsonl` with ≥100 hand-written items across three buckets. This file documents the per-line schema so Phase 1–3 code that touches eval data has a target to type against.

## Schema

Each line in `golden.jsonl` is one JSON object:

```json
{
  "id": "ut-001",
  "bucket": "single_fact",
  "question": "What was Tesla's automotive revenue in FY2023?",
  "filters": {"ticker": "TSLA", "fiscal_year": 2023},
  "answer": "$82.42 billion (FY2023, automotive segment).",
  "answer_rubric": null,
  "must_cite": [
    {"ticker": "TSLA", "fiscal_year": 2023, "section": "Item 7"}
  ],
  "tags": ["revenue", "segment_reporting"]
}
```

| Field | Type | Notes |
|---|---|---|
| `id` | str | Unique. Prefix indicates bucket: `ut-` single_fact, `sy-` synthesis, `tm-` temporal. |
| `bucket` | `"single_fact" \| "synthesis" \| "temporal"` | Determines scoring path. |
| `question` | str | Verbatim user input. |
| `filters` | dict (optional) | Pre-filter hints; the router should arrive at the same set independently. |
| `answer` | str (single_fact only) | Ground truth, exact-match scorable. |
| `answer_rubric` | str (synthesis/temporal) | Rubric for LLM-as-judge. |
| `must_cite` | list[dict] | Chunks that MUST appear in retrieval (ticker / fy / section, optionally chunk_id). |
| `tags` | list[str] | Free-form for slicing metrics. |

## Target distribution

≥100 items total:
- ~40 single-fact lookups
- ~35 multi-section synthesis
- ~25 cross-document / temporal

Sample size matters: at n=40 a faithfulness gain of 0.15 has confidence intervals wide enough to be meaningless. At n=100+ the same gain is publishable.
