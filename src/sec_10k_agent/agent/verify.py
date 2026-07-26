"""Verify_citations node: per-claim entailment check
(docs/architecture.md §4, `verify_citations` node).

Resolves the architecture doc's open question ("NLI model vs. LLM judge for
citation verification") in favor of an LLM judge: an NLI model would be a new
model class this project has no other use for, while the strict-prompt LLM
pattern is already proven here (eval/judge.py). This check is intentionally
tighter than that one -- eval's faithfulness grades a whole answer against its
full context; this grades one claim against only the sources it named.

A claim citing nothing is only "supported" if it's the explicit
not-enough-information abstention -- an unsupported assertion with no sources
is exactly what this node exists to catch.
"""

from __future__ import annotations

from sec_10k_agent.agent.citations import Source, build_source_map, format_sources_block
from sec_10k_agent.agent.json_utils import extract_json_object
from sec_10k_agent.agent.state import Claim, Retrieval
from sec_10k_agent.agent.synthesize import NO_INFO_TEXT
from sec_10k_agent.rag.llm import Generator

VERIFY_SYSTEM_PROMPT = """You are a strict fact-checker for a SEC 10-K question-answering system.

Given a CLAIM and the SOURCE(S) it cites, decide whether the sources actually support the
claim. The claim must be directly supported -- do not give credit for plausible-sounding
statements the sources don't actually say.

Respond with ONLY a JSON object: {"supported": true|false, "reasoning": "<one sentence>"}"""


def verify_claim(claim: Claim, sources: dict[str, Source], generator: Generator) -> bool:
    """True if `claim` is supported by the sources it cites."""
    if not claim.chunk_ids:
        # No sources cited: only the honest "not enough information" abstention counts.
        return claim.text.strip() == NO_INFO_TEXT

    cited = {cid: sources[cid] for cid in claim.chunk_ids if cid in sources}
    if not cited:
        return False  # every cited id was hallucinated/missing -- can't verify

    block = format_sources_block(cited)
    user_prompt = f"CLAIM: {claim.text}\n\nSOURCES:\n{block}"
    response = generator.complete(VERIFY_SYSTEM_PROMPT, user_prompt, temperature=0.0)
    obj = extract_json_object(response.text)
    return bool(obj.get("supported", False))


def verify_citations(
    claims: list[Claim], retrievals: list[Retrieval], generator: Generator
) -> list[Claim]:
    """Return `claims` with `verified` set on each, via one entailment check per claim."""
    sources = build_source_map(retrievals)
    return [
        claim.model_copy(update={"verified": verify_claim(claim, sources, generator)})
        for claim in claims
    ]
