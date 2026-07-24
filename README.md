# sec-10k-agent

A RAG agent for SEC 10-K filings. Multi-hop questions, every claim cited, evals run in CI.

**Status:** Phase 5 implemented and measured, kept opt-in. `sec10k ask`/`sec10k eval --hybrid` add BM25 lexical fusion (RRF) and cross-encoder reranking on top of dense retrieval. Measured against the Phase 4 dense-only baseline on the 26-item seed set, the result is mixed, not a clean win: faithfulness rose (0.89 → 0.94) but correctness fell (0.73 → 0.62), driven almost entirely by the temporal bucket (0.67 → 0.37) — the cross-encoder reranks each chunk by topical relevance alone, with no fiscal-year diversity awareness, so multi-year comparison questions can end up served several same-year chunks instead of one-per-year coverage. Both baselines are committed under `data/eval/baselines/` for comparison. **Hybrid stays opt-in (`--dense-only` is still the default)** pending diversity-aware selection — a natural fit for Phase 6's router, which already plans to detect `is_temporal`. Phase 4 (eval harness), Phase 3 (RAG MVP), Phase 2 (pgvector retrieval), and Phase 1 (ingestion) are done. Phase 1c (XBRL) is partial; see Known limitations. Phase 6 (agent: router + decomposition + multi-hop) is next.

---

## What it does

Answers questions about US public-company 10-K filings. Examples in current scope:

- *How did Apple's risk factor language change from FY2021 to FY2025?*
- *Compare NVIDIA and JPMorgan on disclosure of AI-related operational risk.*
- *What does JPMorgan say about credit risk in their most recent filing?*
- *What was Apple's Greater China revenue in FY2024?* (XBRL lookup, not embedding search)


These need multi-hop retrieval, time awareness, and exact citations. Naive RAG misses on all three.
Built as a portfolio project to demonstrate production-grade RAG patterns over real regulated filings. Not for use in investment decisions.

## Why this project

Portfolio project. Picked because it forces four things tutorial RAG demos skip:

1. Multi-hop retrieval over long structured documents.
2. Time awareness — fiscal years, filing dates, language that drifts year over year.
3. Citations that get verified before the answer goes out, not just appended.
4. Evals for questions with no single correct answer (comparisons, summaries).

## Scope

- **3 tickers**, three sectors: `AAPL` (consumer tech), `NVDA` (semis / AI infra), `JPM` (banking).
- **5 fiscal years:** FY2021–FY2025.
- **15 10-K filings**, ~3,800 chunks after parsing.
- 10-K only. No 10-Q, 8-K, or other forms.

## Architecture

```
                 ┌──────────────┐
   user question │  FastAPI /   │
   ─────────────▶│  Streamlit   │
                 └──────┬───────┘
                        ▼
              ┌─────────────────────┐
              │  Agent              │
              │  (framework picked  │
              │   in Phase 6)       │
              │                     │
              │  ┌───────────────┐  │  router — an LLM
              │  │    route      │  │  call. Decides:
              │  └───────┬───────┘  │  semantic or XBRL,
              │          ▼          │  whether to split
              │  ┌───────────────┐  │  the question, and
              │  │  decompose    │  │  whether it's about
              │  └───────┬───────┘  │  changes over time.
              │          ▼          │
              │  ┌───────────────┐  │
              │  │ plan_retrieve │  │
              │  └───────┬───────┘  │
              │          ▼          │
              │  ┌───────────────┐  │  one retrieval per
              │  │  retrieve_*   │──┼──year or per ticker
              │  └───────┬───────┘  │
              │          ▼          │
              │  ┌───────────────┐  │
              │  │  synthesize   │  │
              │  └───────┬───────┘  │
              │          ▼          │
              │  ┌───────────────┐  │  drop the answer
              │  │ verify_cites  │──┼──if any citation
              │  └───────┬───────┘  │  doesn't check out
              │          ▼          │
              │  ┌───────────────┐  │
              │  │   finalize    │  │
              │  └───────────────┘  │
              └──────────┬──────────┘
                         ▼
        ┌────────────────────────────────┐
        │  Hybrid retrieval                │
        │  ┌──────────┐    ┌──────────┐   │
        │  │  BM25    │    │ pgvector │   │
        │  │          │    │  (HNSW)  │   │
        │  └────┬─────┘    └────┬─────┘   │
        │       └────┬──────────┘         │
        │            ▼                    │
        │    Reciprocal Rank Fusion       │
        │            ▼                    │
        │      Cross-encoder reranker     │
        └────────────────────────────────┘
                         ▲
                         │
              ┌──────────┴──────────┐
              │  Postgres           │
              │  • chunks + vectors │
              │  • XBRL facts       │
              │    (with dims)      │
              │  • filing metadata  │
              └─────────────────────┘
```

Detail in [docs/architecture.md](docs/architecture.md). Decision history in [docs/design-decisions.md](docs/design-decisions.md).

## Tech stack

| Layer | Tooling |
|---|---|
| Ingestion | `edgartools`, `httpx` (rate-limited), `BeautifulSoup`, iXBRL extraction with full dimensional capture |
| Storage | Postgres 16 + pgvector. One DB for vectors, metadata, and XBRL facts. See [ADR-001](docs/design-decisions.md). |
| Retrieval | Hybrid BM25 + dense (pgvector HNSW) + cross-encoder reranker |
| Embeddings | `BAAI/bge-large-en-v1.5`, local via `sentence-transformers`. Free, ~1.3 GB. Voyage A/B in Phase 5. |
| Reranker | `BAAI/bge-reranker-base` via fastembed (ONNX) — see [ADR-010](docs/design-decisions.md) |
| Agent framework | Picked in Phase 6. Options: LangGraph, PydanticAI, hand-rolled state machine. See [ADR-002](docs/design-decisions.md). |
| Generator LLM | Grok via xAI API, accessed through the `openai` SDK with `base_url=https://api.x.ai/v1`. Provider swap is one config value. |
| Judge LLM | Gemini 2.5 Flash, free tier. Different model family from the generator. See [ADR-003](docs/design-decisions.md). |
| Evals | RAGAS metrics, LLM-as-judge with versioned prompts, 100+ hand-written questions |
| Observability | Langfuse, set up from Phase 1 |
| API / UI | FastAPI, Streamlit (MVP), rate limiting, Redis cache |
| Deploy | AWS App Runner, GitHub Actions for CI |

## Roadmap

| Phase | Status | Focus | Public output |
|---|---|---|---|
| 0 | done | Repo, scope, architecture, ADRs | This README + a launch post |
| 1a | done | EDGAR client (download + cache + retry + throttle) | `data/raw/` populated |
| 1b | done | Section parser + chunker (no overlap, prev/next chain) | `data/processed/chunks.parquet` (3,761 rows) |
| 1c | partial | XBRL extractor with dimensions | `xbrl.parquet` exists; AAPL Greater China acceptance test unresolved — see [Known limitations](#known-limitations) |
| 2 | done | pgvector indexing + filtered retrieval | 3,761 chunks with 1024-dim BGE embeddings; HNSW + btree indexes; retrieval verified end-to-end |
| 3 | done | Single-hop RAG MVP + Streamlit UI with retrieval trace | `sec10k ask` + Streamlit; cited answers via Grok or local Ollama. **Blog post #1**: Section-Aware Chunking Without Overlap |
| 4 | done | Eval harness: golden set, retrieval metrics + LLM-judge (faithfulness/correctness), weekly CI | Baseline: faith 0.89 / correct 0.73 / recall 92% on the 26-item seed. See [ADR-009](docs/design-decisions.md) on judge-vs-RAGAS |
| 5 | done | Hybrid retrieval (BM25+dense via RRF) + cross-encoder reranking, opt-in | Faithfulness lift (0.89→0.94) but correctness regression on temporal (0.67→0.37) — mixed result, documented honestly, kept opt-in. **Blog post #2**: when hybrid retrieval hurts |
| 6 | pending | Agent: pick framework, router, decomposition, multi-hop | Multi-hop eval bucket |
| 7 | pending | Production: deploy, observability, caching, cost tracking | Live demo |
| 8 | pending | Launch writeup | **Blog post #3** |

## Project layout

```
sec-10k-agent/
├── src/sec_10k_agent/
│   ├── ingestion/        # EDGAR client, parser, chunker, XBRL extractor (1c)
│   ├── retrieval/        # pgvector + BM25 + reranking (Phase 2+)
│   ├── agent/            # router + state machine + tools (Phase 6)
│   ├── eval/             # Golden set, RAGAS, judge (Phase 4)
│   ├── api/              # FastAPI + Streamlit UI (Phase 3+)
│   ├── observability/    # Langfuse, cost tracking
│   ├── cli.py            # `sec10k` command
│   ├── config.py         # pydantic-settings, lazy via get_settings()
│   └── scope.py          # tickers + fiscal years
├── data/
│   ├── raw/              # EDGAR cache (gitignored)
│   ├── processed/        # parsed/, chunks.parquet, xbrl.parquet (gitignored)
│   └── eval/             # Golden eval set + schema (committed)
├── tests/
├── notebooks/
├── docs/
│   ├── architecture.md
│   └── design-decisions.md
├── scripts/              # CLIs and DB init SQL
└── .github/workflows/    # CI
```

## Quick start

### Prerequisites

- Python 3.12 (pinned in `.python-version`; `uv python install 3.12` if missing)
- [uv](https://docs.astral.sh/uv/)
- Docker (for local Postgres + Langfuse)

### Setup

```bash
# 1. Install
uv sync --extra dev

# 2. Env
cp .env.example .env
# Fill in: SEC_USER_AGENT (required), XAI_API_KEY, GEMINI_API_KEY

# 3. Local infra (Postgres + Langfuse)
docker compose -f docker-compose.dev.yml up -d

# 4. One-time Langfuse setup:
#    Open http://localhost:3000, create an account (local-only),
#    create a project, copy the public + secret keys into .env.
```

### What works today

```bash
# Download all 15 filings in scope (~30 seconds with throttling)
uv run sec10k download

# Parse cached HTML to ParsedFiling intermediates
uv run sec10k parse

# Chunk parsed filings to data/processed/chunks.parquet
uv run sec10k chunk
# --word-count for a fast first pass without downloading the BGE tokenizer
```

After all three, `data/processed/chunks.parquet` has 3,761 rows. Each chunk carries `chunk_id`, `ticker`, `fiscal_year`, `section`, `text`, `token_count`, plus `prev_chunk_id` / `next_chunk_id` for retrieval-time context expansion.

Then index the chunks into pgvector and ask questions (Phase 2 + 3):

```bash
# One-time: embed the chunks and load them into Postgres (docker compose must be up)
uv run sec10k index

# Ask a question. Needs a generator: set XAI_API_KEY (Grok), OPENROUTER_API_KEY,
# or OLLAMA_BASE_URL (local, $0) in .env. Filters map onto the retriever's pre-filter.
uv run sec10k ask "What supply chain risks does Apple disclose?" --ticker AAPL

# Or launch the Streamlit UI (question box + retrieval trace)
uv run streamlit run src/sec_10k_agent/api/streamlit_app.py
```

Every answer cites the sources it used (`[1]`, `[2]`, …), and the UI shows which retrieved chunks were cited versus merely retrieved. Grok/OpenRouter answer in seconds; a local Ollama model on CPU can take minutes per query.

Score the whole golden set (Phase 4):

```bash
# Retrieval metrics only (free, no LLM): context recall, hit@k, MRR
uv run sec10k eval --no-judge

# Full run adds LLM-judged faithfulness + correctness (needs a generator + judge key)
uv run sec10k eval --out data/eval/baselines/latest
```

## Known limitations

- **JPM Items 7 and 8** are filed in Exhibit 13 ("incorporated by reference"), not inline in the 10-K. The parser captures the reference stub, not the underlying MD&A and financial statements. Will be addressed in a follow-up phase.
- **NVDA Item 8** is similarly external. Same fix path.
- **Items 1B, 4, 6, 9, 11, 14, 16** are routinely missing or marked "Reserved" in real filings — that's normal SEC behavior, not a parser bug. The parser flags missing items in its `notes` field on the ParsedFiling intermediate.
- **Item 1C (Cybersecurity)** only appears for fiscal years ending after Dec 15, 2023, per SEC rule. Earlier filings legitimately don't have it.
- **XBRL dimensional capture is partial.** The XBRL extractor runs and produces `data/processed/xbrl.parquet` with structured facts, but the AAPL Greater China acceptance test from the architecture doc still returns 0 rows. Two root causes: (a) pandas-to-parquet round-trip pollutes dict columns with None values across the union of all axes, and (b) we have not yet identified Apple's actual geographic member name (`country:CN` was a guess). To be resolved before Phase 4 evals include any XBRL-dependent questions.

## Rules of thumb

- No answer without a citation.
- Measure before changing. Every architecture change re-runs the eval.
- Boring parts first. Parser and chunker quality cap everything downstream.
- Production basics in Phase 1, not Phase 7. Observability and config validation belong early.
- Defer reversible decisions. Agent framework, reranker, sparse retriever — pick when you actually need them.

## License

MIT

## Credits

Data from [SEC EDGAR](https://www.sec.gov/edgar).
