"""Synthesize node: one LLM call producing a list of claims, each tagged with
the source id(s) it relies on (docs/architecture.md §4, `synthesize` node).

Differs from Phase 3's single-hop pipeline (rag/prompts.py), which asks for
free-text with `[n]` markers parsed by regex. Multi-hop retrievals span several
subqueries, so claims are tagged with the sources' own stable ids (a
`chunk_id`, or a synthetic `fact:...` key for XBRL facts -- see citations.py)
instead of a flat numbering. That per-claim, per-source tagging is what lets
`verify_citations` check each claim against exactly the evidence it named.
"""

from __future__ import annotations

from sec_10k_agent.agent.citations import build_source_map, format_sources_block
from sec_10k_agent.agent.json_utils import extract_json_object
from sec_10k_agent.agent.state import Claim, Retrieval
from sec_10k_agent.rag.llm import Generator

NO_INFO_TEXT = "I don't have enough information in the provided filings to answer that."

SYNTHESIZE_SYSTEM_PROMPT = f"""You are a financial analyst assistant answering a question about \
SEC 10-K filings using ONLY the numbered SOURCES provided (filing text excerpts and/or \
structured financial facts).

Produce a list of claims that together answer the question. Each claim must be a single,
specific, checkable statement, tagged with the exact source id(s) in square brackets (e.g.
"AAPL__Item 1A__0003" or "fact:AAPL:2024:us-gaap:Revenues") it relies on. Do not combine
unrelated facts into one claim -- one checkable assertion per claim.

If the sources do not contain enough information to answer the question, output exactly one
claim with text "{NO_INFO_TEXT}" and an empty source_ids list.

Output ONLY a JSON object:
{{"claims": [{{"text": "...", "source_ids": ["...", ...]}}, ...]}}"""


def synthesize(question: str, retrievals: list[Retrieval], generator: Generator) -> list[Claim]:
    """Produce claims answering `question` from `retrievals`' chunks/facts.

    Hallucinated source ids (not present in the retrieved set) are dropped
    from a claim rather than trusted -- a claim citing nothing real is
    equivalent to an unsupported claim, which `verify_citations` will reject.
    """
    sources = build_source_map(retrievals)
    block = format_sources_block(sources)
    user_prompt = f"SOURCES:\n{block}\n\nQUESTION: {question}\n\nOutput the claims JSON:"
    response = generator.complete(SYNTHESIZE_SYSTEM_PROMPT, user_prompt, temperature=0.0)
    obj = extract_json_object(response.text)

    claims: list[Claim] = []
    for raw in obj.get("claims") or []:
        source_ids = [sid for sid in (raw.get("source_ids") or []) if sid in sources]
        text = str(raw.get("text", "")).strip()
        if text:
            claims.append(Claim(text=text, chunk_ids=source_ids))

    if not claims:
        claims = [Claim(text=NO_INFO_TEXT, chunk_ids=[])]
    return claims
