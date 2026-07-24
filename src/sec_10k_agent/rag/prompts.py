"""Prompt construction for single-hop RAG.

The system prompt is the whole ballgame for citation grounding: the model must
answer *only* from the numbered context and cite the sources it used. Context
formatting and citation parsing are pure functions so they can be unit-tested
without an LLM.
"""

from __future__ import annotations

import re

from sec_10k_agent.retrieval.models import RetrievedChunk

SYSTEM_PROMPT = """You are a financial analyst assistant answering questions about SEC 10-K filings.

Rules:
- Answer ONLY using the numbered SOURCES provided. Do not use outside knowledge.
- Cite every claim with the source number(s) in square brackets, e.g. [1] or [2][3].
- If the sources do not contain the answer, say exactly: "I don't have enough information in the provided filings to answer that." Do not guess.
- Be concise and specific. Prefer figures, dates, and direct language from the filings.
- When sources disagree or span multiple fiscal years, say so explicitly and cite each."""

_CITATION_RE = re.compile(r"\[(\d+)\]")


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered SOURCES block for the prompt."""
    if not chunks:
        return "(no sources retrieved)"
    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] {c.citation()}"
        blocks.append(f"{header}\n{c.text.strip()}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble the user turn: the numbered sources followed by the question."""
    context = format_context(chunks)
    return f"SOURCES:\n{context}\n\nQUESTION: {question}\n\nAnswer with citations:"


def parse_citations(text: str, n_sources: int) -> list[int]:
    """Extract the 1-based source numbers cited in `text`, de-duplicated in order.

    Numbers outside 1..n_sources (hallucinated citations) are dropped.
    """
    seen: list[int] = []
    for match in _CITATION_RE.finditer(text):
        idx = int(match.group(1))
        if 1 <= idx <= n_sources and idx not in seen:
            seen.append(idx)
    return seen
