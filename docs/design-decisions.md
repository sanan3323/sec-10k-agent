# Architecture Decision Records

Each ADR captures one consequential design choice: what we picked, what we considered, why we picked it, and what would make us revisit. ADRs are append-only; if a decision changes, write a new ADR that supersedes the old one — don't edit history.

---

## ADR-001 — Postgres + pgvector instead of a dedicated vector DB

**Status:** Accepted
**Date:** 01-04-2026

### Context

The v1 corpus is ~5,000 chunks. We need metadata filtering on ticker, year, and section. We also need a relational store for XBRL facts, filing metadata, and eval-run logs.

### Decision

Postgres 16 with the `pgvector` extension. HNSW index for the embedding column. B-tree indexes on metadata columns used as filters. JSONB + GIN on the XBRL `dimensions` column.

### Alternatives considered

- **Qdrant** — purpose-built vector DB. Excellent metadata filtering, distributed scale, gRPC.
- **Pinecone / Weaviate / Chroma** — managed offerings with similar features at this scale.

### Reasoning

5K vectors don't justify a specialized vector DB. Sharding, distributed indexes, billion-vector throughput — none of it gets used at this scale.

Running two databases would mean operating both, syncing what's in each, and writing cross-system glue every time a query needs to join vectors with metadata or XBRL facts. Postgres + pgvector keeps everything in one place. A query like "give me chunks for AAPL FY2025 Item 1A near vector X" is one SQL statement with one JOIN.

HNSW recall in pgvector is competitive with dedicated vector DBs at this scale. Setup is one `docker compose up`.

The decision is reversible. The retrieval interface is built against a `VectorStore` protocol, so swapping backends later is one adapter, not a refactor of callers.

### What would change our mind

- Corpus growth past ~1M vectors, where pgvector HNSW build/query time degrades.
- A need for per-tenant vector index isolation, which Postgres doesn't model cleanly.
- A clear retrieval-quality gap on the golden eval set after Phase 5 tuning.

---

## ADR-002 — Agent framework: deferred to Phase 6

**Status:** Pending
**Date:** TBD

The agent layer (Phase 6) will be evaluated against three options: LangGraph, PydanticAI, and a hand-rolled state machine. Decision criteria, candidates' tradeoffs, and the final pick will be written here when Phase 6 begins, not before. Pre-committing now would defeat the point of the evaluation.

---

## ADR-003 — Generator and judge model selection

**Status:** Accepted (provisional)
**Date:** 08-05-2026

**Generator:** Grok via xAI API (project budget constraint, OpenAI-SDK-compatible).
**Judge:** Gemini 2.5 Flash via Google's free tier.

The two have to be different model families to avoid self-judge bias in Phase 4 evals. Ollama with a Llama-class model is the documented fallback if Gemini's free-tier rate limits become a problem under eval load.

Revisit if Grok pricing changes the budget math, Gemini's free tier gets stricter, or empirical agreement between Gemini and a third judge (sample audit) is below 0.7 Cohen's kappa.

---

## ADR-004 — Chunking: paragraph-aligned, no overlap, prev/next chain

**Status:** Accepted
**Date:** 

### Context

Standard RAG tutorials use fixed-token chunks with sliding-window overlap (e.g. 512 tokens with 80-token overlap). This duplicates text on disk, double-counts terms in BM25, cuts mid-sentence in ways that hurt embedding quality, and pays for context that's only useful if a question happens to land on a chunk boundary.

### Decision

Paragraph-aligned chunking with no token-level overlap. Context expansion happens at retrieval time via `prev_chunk_id` / `next_chunk_id` pointers stored on each chunk.

Three rules in the chunker:

1. Default: one chunk per paragraph.
2. Paragraph longer than `max_tokens` (480 BGE tokens): split at sentence boundaries. If a single "sentence" still exceeds the cap (legal boilerplate, sanctioned-party lists), recursively word-split it.
3. Paragraph shorter than `min_tokens` (80): merge forward into the next paragraph in the same section. Never merges across an Item boundary.

Hard cap: `token_count <= max_tokens` is asserted on every chunk. The 480 cap leaves 32 tokens of headroom under BGE-large's 512-token hard limit for the special tokens (`[CLS]`, `[SEP]`) added at embed time.

### Reasoning

The architecture already commits to retrieval-time context expansion via prev/next pointers. Sliding-window overlap on top of that pays for the same thing twice: duplicated tokens during indexing, plus expansion at query time anyway. Eliminating overlap removes ~15% of indexed text without losing context.

The chunk-size cap is a "fail loud" guardrail. BGE silently truncates input over 512 tokens, which would be an invisible quality bug. The assertion forces the chunker to handle long paragraphs explicitly.

The recursive word-split fallback is what made this robust on real filings. JPM has paragraphs in regulatory boilerplate and sanctioned-party lists with 800+ words and no periods. Without word-split, the assertion would trip on those.

### What would change our mind

- A retrieval-quality benchmark showing that overlap meaningfully helps (test in Phase 5 if numbers are unconvincing).
- A switch to a long-context embedding model (>2K tokens) where the cap and the merge-forward rule become irrelevant.

---

## ADR-005 — Parser: regex on extracted text, not edgartools' typed accessors

**Status:** Accepted
**Date:** 

### Context

`edgartools` exposes a typed `TenK` object with attributes like `business`, `risk_factors`, `mda`. The library version targeted Item-by-Item access. Our parser extracts the same content but does it with regex over plain text instead.

### Decision

Strip HTML to text via BeautifulSoup, normalize Unicode (NFKC), inject newlines before block-level tags, then run a single regex (`^[ \t]*item[ \t]+([1-9]\d?[A-C]?)\b...`) to find Item headers. Filter Table-of-Contents matches via a gap-to-next-header threshold.

### Reasoning

A text-based parser handles all years and filers identically. It doesn't depend on `edgartools` keeping its typed-accessor API stable across versions, and it doesn't need a fallback path when an attribute is missing.

The TOC-stripping heuristic (gap threshold) made one big real-world fix necessary: incorporated-by-reference stubs (JPM Item 7/8 pointing at Exhibit 13) are short, sometimes only a few hundred characters. The threshold was lowered from 500 to 100 to catch them. False positives are bounded by the constraint that a TOC has many Items in close succession, while real sections are spread out.

The parser carries a `PARSER_VERSION` string. Cached `ParsedFiling` JSONs are re-parsed when the version changes. Re-parsing is cheap (no network); re-chunking and re-embedding is more expensive, which is why the chunker is a separate stage.

### What would change our mind

- A 10-K with an Item structure the regex can't find. Most likely cause: a non-Latin character in "Item" (we've seen non-breaking spaces, but NFKC handles those). Track on a per-filing basis via the `notes` field on `ParsedFiling`.
- `edgartools` adding a stable, version-pinned section accessor that handles TOC stripping and incorporated-by-reference natively.

---

## ADR-006 — Incorporated-by-reference content (JPM Exhibit 13, NVDA Item 8)

**Status:** Accepted as a known limitation
**Date:** 

### Context

JPM files Items 7 (MD&A) and 8 (Financial Statements) as one-paragraph stubs that incorporate Exhibit 13 by reference. The actual MD&A and financial statements are in Exhibit 13, which is a separate document with its own URL on EDGAR. NVDA does the same thing for Item 8.

The current parser captures the stub (1 chunk) and flags missing content correctly, but does not follow the reference to Exhibit 13.

### Decision

Treat Exhibit 13 / incorporated-by-reference content as a known limitation in v1. Document it in the README. Decide in Phase 1d (or defer to post-v1) whether to add a secondary fetch path for Exhibit 13.

### Reasoning

Adding Exhibit 13 support is real work — a separate fetcher, a separate parser path, and a way to reconcile chunks back to the parent 10-K's metadata. The cost-benefit comes down to: AAPL files inline, NVDA mostly files inline (Item 7 is there, just not Item 8), only JPM is a full miss on MD&A. We'd be adding ~30% of corpus complexity for one ticker's worth of MD&A content.

The eval set can compensate: questions about JPM's MD&A that depend on Exhibit 13 content can be flagged as out of scope until the limitation is fixed.

---

## ADR-007 — Incorporated-by-reference content (JPM Exhibit 13, NVDA Item 8)

**Status:** Accepted
**Date:** 10-05-2026

### Context

Switched from sentence-transformers (PyTorch backend) to fastembed (ONNX runtime) for inference. sentence-transformers requires PyTorch, which has unreliable wheels on Intel Mac and forces a numpy version downgrade that conflicts with our other deps. fastembed runs the same BAAI/bge-large-en-v1.5 weights via ONNX with no PyTorch dependency. Output vectors are cosine-equivalent to the PyTorch version. Revisit if we move to GPU inference or need a model fastembed doesn't host.


### What would change our mind

- Adding a fourth ticker that also incorporates by reference (typical for big banks).
- An eval bucket dedicated to JPM that the system can't answer at all without Exhibit 13.
- A decision to expand v1 scope to include other Big Four banks (BAC, WFC, C), all of which use the same incorporation pattern.
