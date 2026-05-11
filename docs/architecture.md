# Architecture

This is the system design doc. Update it when decisions change. PRs that change the architecture should update this file. Decisions that need a written record (like "why pgvector?") live in [`design-decisions.md`](design-decisions.md) as ADRs.

## Goals

- Answer multi-hop, time-aware, comparison questions over 10-K filings, with citations.
- Make every component swappable: embeddings, reranker, LLM, vector store.
- Run an eval in CI on every PR.
- Stay cheap during development. Target: under $5/day.

## Out of scope (for v1)

- 10-Q, 8-K, proxy statements, S-1s, foreign filings.
- Real-time freshness. Filings are downloaded in batches.
- Anything that looks like investment advice. The system reports what filings say. It doesn't tell you what to do.

## Data model

### Filing
```
Filing {
  cik: str             # 10-digit Central Index Key, zero-padded
  ticker: str
  form: "10-K"
  fiscal_year: int     # the FY the filing covers, NOT the year it was filed
  filing_date: date
  period_of_report: date
  accession_number: str
  raw_html_path: Path
  parsed_at: datetime | None
}
```

### Chunk
```
Chunk {
  chunk_id: str        # f"{accession}__{section_slug}__{idx:04d}"
  cik: str
  ticker: str
  fiscal_year: int
  accession_number: str
  section: str         # "Item 1A", "Item 7", etc.
  section_title: str   # "Risk Factors", "MD&A", etc.
  text: str            # paragraph-aligned, capped at 480 BGE tokens
  token_count: int
  prev_chunk_id: str | None   # for context expansion at retrieval time
  next_chunk_id: str | None
}
```

### XBRLFact
```
XBRLFact {
  cik: str
  ticker: str
  fiscal_year: int
  accession_number: str
  concept: str         # e.g. "us-gaap:Revenues"
  value: Decimal
  unit: str            # e.g. "USD", "shares"
  period_start: date
  period_end: date
  context_id: str      # raw XBRL contextRef, kept for traceability
  dimensions: dict[str, str]
  # ^ XBRL dimensional axes. Required for segment, geographic, and product-line
  # facts. Without this we can't represent "Greater China revenue for FY2024":
  # that fact lives at axis srt:StatementGeographicalAxis with member country:CN.
  # Total revenue has dimensions={}; the Greater China line has
  # dimensions={"srt:StatementGeographicalAxis": "country:CN"}. The parser must
  # capture every (axis, member) pair from the XBRL context, not just the period.
}
```

Acceptance test for the XBRL parser: it can answer "What was Apple's Greater China revenue in FY2024?" using only the XBRL fact table. If it can only answer "What was Apple's total revenue in FY2024?", the parser is broken.

## Components

### 0. Query router

Sits in front of the agent. Takes the raw user question, returns a structured retrieval plan. It's a small LLM call with a Pydantic output schema. Not keyword matching, not regex.

**Input:** the raw question string.

**Output:**
```
RoutingPlan {
  needs_decomposition: bool      # true if the question covers multiple entities or years
  retrieval_modes: list[Literal["semantic", "structured_xbrl", "both"]]
  is_temporal: bool              # true if "evolved", "since", "compared to last year", etc.
  candidate_filters: {
    tickers: list[str]
    fiscal_years: list[int]
    sections: list[str]          # e.g. ["Item 1A", "Item 7"]
  }
  reasoning: str                 # one sentence; logged for debugging
}
```

LLM call rather than rules because questions like *"How has Apple's exposure to the Chinese consumer changed?"* have no obvious keywords but are clearly time-aware, need both semantic and structured retrieval, and target one company. Rules don't handle that; an LLM with a tight schema and a few examples does.

The router runs before decomposition because the decomposer needs the plan to know how to split the question. *"Compare Apple and NVIDIA"* splits by ticker. *"How did Apple's risk factors evolve?"* splits by fiscal year. The router decides which axis applies.

Kept as a separate node so it can use a smaller, cheaper model than synthesis, and so it can be tuned and evaluated on its own. Routing is the main lever for both latency and cost.

### 1. Ingestion

Implementation summary; full design in the source files.

- **EDGAR client** (`ingestion/edgar_client.py`). Wraps `edgartools` behind a small Protocol so tests can substitute a fake. Throttled to 5 req/sec via `RateLimiter` (SEC's hard ceiling is 10/sec; we leave headroom). `tenacity` retries on transient failures. Always sends a `User-Agent` from env. Caches `(ticker, fy)` to disk: `data/raw/{ticker}_{fy}.html` plus `{ticker}_{fy}.json` metadata. Filings are immutable after acceptance, so cache invalidation is not needed.

- **Section parser** (`ingestion/parser.py`). Runs on extracted text, not HTML structure. Strips `<script>`, `<style>`, `<head>` via BeautifulSoup, applies NFKC unicode normalization (handles `&nbsp;` and similar), then injects `\n` before block-level tags so headers can't be glued together by squashed HTML. Detects Item headers with a regex restricted to horizontal whitespace (so the regex never crosses a line boundary). Filters Table-of-Contents matches by gap-to-next-header: TOC entries are tightly packed, real sections are far apart. The `_TOC_GAP_THRESHOLD` is 100 chars (lowered from 500 to catch incorporated-by-reference stubs like "See Annual Report"). Output is a `ParsedFiling` cached at `data/processed/parsed/{accession}.json`. The parser carries a `PARSER_VERSION` string; cache entries with stale versions are re-parsed.

- **Chunker** (`ingestion/chunker.py`). Section-aware; never crosses an Item boundary except via the prev/next chain. Three rules:
  1. Default: one chunk per paragraph.
  2. Paragraph longer than `max_tokens` (480 BGE tokens): split at sentence boundaries, then if any sentence is still over the cap, recursively word-split it. This handles legal boilerplate and sanctioned-party lists that have no periods.
  3. Paragraph shorter than `min_tokens` (80): merge forward into the next paragraph in the same section. Never merges across an Item boundary.

  No token-level overlap. Context expansion happens at retrieval time via `prev_chunk_id` / `next_chunk_id`, which span Item boundaries. The chunker asserts `token_count <= max_tokens` on every chunk before returning. See [ADR-004](design-decisions.md).

- **XBRL extractor** (`ingestion/xbrl.py`, Phase 1c). Pulls structured facts from iXBRL inline tags. Writes `data/processed/xbrl.parquet`. Capturing dimensions is required, not optional.

### 2. Storage

One Postgres database for everything. See [ADR-001](design-decisions.md).

- **`filings`** — one row per 10-K. Btree indexes on `(ticker, fiscal_year)`.
- **`chunks`** — chunk text, metadata, and `embedding vector(1024)`. HNSW index on `embedding` (cosine), btree indexes on `ticker`, `fiscal_year`, `section`.
- **`xbrl_facts`** — flat fact table. `dimensions` is `JSONB` with a GIN index for axis lookups.
- **`eval_runs`** — eval execution data (Phase 4 onward).

With one DB, "give me chunks for AAPL FY2025 Item 1A near vector X" is one SQL query with one `JOIN` against `filings`. No cross-system glue.

### 3. Retrieval

- **Dense retrieval.** pgvector cosine search with metadata filters in the SQL `WHERE`.
- **Sparse retrieval.** BM25 over the chunks parquet, or Postgres `tsvector` if it benchmarks better. Decision in Phase 5.
- **Fusion.** Reciprocal Rank Fusion (k=60), top-k=20.
- **Reranker.** `BAAI/bge-reranker-v2-m3` cross-encoder. Narrows 20 to 5.
- **Context expansion.** At synthesis time, top-N retrieved chunks pull in their `prev_chunk_id` and `next_chunk_id` neighbors. Expansion is bounded to 1 hop in each direction per retrieved chunk, with a total cap of 8,000 tokens across all expanded context. If the cap is hit, expansion stops and the un-expanded chunks are sent. The 1-hop limit prevents runaway expansion when several top-K chunks happen to be neighbors.

### 4. Agent (Phase 6)

> **Framework: not yet picked.** Options are LangGraph, PydanticAI, and a hand-rolled state machine. Decision when Phase 6 starts. See [ADR-002](design-decisions.md).

The node graph, regardless of framework:

State carries `question`, `routing_plan`, `subqueries[]`, `retrievals[]`, `draft_answer`, `verified_citations[]`, `tool_calls[]`.

Nodes:
- **`route`.** Produces the `RoutingPlan` from §0.
- **`decompose`.** Runs only if `routing_plan.needs_decomposition`. Splits into N sub-queries with explicit `(ticker, fiscal_year)` filters per sub-query.
- **`plan_retrievals`.** Per sub-query, decides between semantic retrieval and the XBRL tool, based on `routing_plan.retrieval_modes`.
- **`retrieve`.** Runs the plan, fills `retrievals[]`.
- **`synthesize`.** LLM call. Structured output: a list of claims, each tagged with the chunk_id(s) it relies on.
- **`verify_citations`.** No LLM. For each (claim, chunk_id) pair, runs an entailment check. Could be a small NLI model or a strict-prompt LLM judge — picked in Phase 6. Unverified claims get dropped or sent back to `retrieve` for one more try (max 1 retry).
- **`finalize`.** Formats the user-facing answer with `[Source: AAPL FY2025 10-K, Item 1A]` citations, resolved from chunk_ids.

Edges are conditional. `verify_citations` either loops back to `retrieve` or moves on to `finalize`.

The agent also has a `lookup_financial_metric` tool that queries `xbrl_facts`. That's how "What was Apple's Greater China revenue in FY2024?" becomes a structured lookup instead of an embedding-search guess.

### 5. Evals (Phase 4)

- **Golden set.** 100+ hand-written questions in `data/eval/golden.jsonl`. Three buckets:
  - ~40 single-fact lookups (exact-match scorable).
  - ~35 multi-section synthesis (rubric-scored by an LLM judge).
  - ~25 cross-document or temporal (rubric + retrieval-trajectory scoring).

  At n=40 the confidence intervals on a "0.74 → 0.89" lift swallow the gain. 100+ is the minimum for honest before/after numbers.

- **Cost.** ~600–1,200 judge calls per full pass (100 questions × variants × judges). Gemini Flash free tier covers this. Generator-side cost (Grok) is the bigger line item.

- **Metrics.** RAGAS: context precision, context recall, faithfulness, answer relevancy. For the agent: number of retrievals, latency, cost.

- **LLM-as-judge.** Generator: Grok (xAI). Judge: Gemini 2.5 Flash. Different model families to avoid self-grading bias. Judge prompt is committed and versioned. Ollama (Llama-class) is the documented fallback. See [ADR-003](design-decisions.md).

- **CI.** `eval.yml` runs the harness on every PR and posts metric deltas vs. main. Added in Phase 4.

### 6. Observability (Phase 1 onward)

- Langfuse traces every LLM call, retrieval, and tool call.
- Each trace gets tagged with `eval_run_id` if it's part of an eval, `user_query_id` otherwise.
- A cost aggregator reads Langfuse at the end of each run and writes to `eval_runs` in Postgres.

## Failure modes we expect

| Failure | What it looks like | How we catch it |
|---|---|---|
| Section parser misses an Item boundary | Risk-factor questions return MD&A text | Single-fact eval bucket; spot-check chunk metadata distribution |
| Embeddings don't capture financial language | "Headwinds" returns weather content | A/B BGE-large baseline against Voyage on synthesis bucket |
| Router over-splits simple questions | Slow, expensive, no quality gain | Trajectory eval: number of retrievals per question type, flag outliers |
| XBRL parser ignores dimensions | Geographic and segment questions return wrong totals | Acceptance test: AAPL Greater China revenue must come back with the right axis/member |
| Citation says `Item 1A` but text is from `Item 7` | Hallucinated citation | `verify_citations` node + an eval metric that re-resolves every chunk_id |
| Long answers drop middle citations | Later sources missing from the answer | Eval includes 6-source synthesis questions |
| Generator and judge agree too easily | Inflated faithfulness scores | Periodic spot-audit by a third model on a 10% sample |
| Incorporated-by-reference content invisible | JPM Item 7/8, NVDA Item 8 are 1-chunk stubs | Known limitation; flagged in README; Exhibit 13 follow-up |

## Open questions

- NLI model vs. LLM judge for citation verification. Decide in Phase 6 based on latency, cost, and how well it agrees with the rubric.
- How to handle filings that restate prior years. Out of scope for v1. Track in a `restatements.md` log if it comes up.
- Cross-encoder reranker latency on CPU might be the bottleneck. If so, swap to Cohere Rerank API in Phase 5.
- Sparse retrieval: `rank_bm25` over parquet vs. Postgres `tsvector`. Bench both in Phase 5.
- Exhibit 13 / incorporated-by-reference parsing path. JPM and NVDA need it for full MD&A and financials coverage. Decide in Phase 1d or defer to post-v1.
