"""Single-hop RAG pipeline (Phase 3).

retrieve -> build a numbered-sources prompt -> generate a cited answer. This is
deliberately one hop: one retrieval, one generation. Multi-hop decomposition
and routing arrive with the agent in Phase 6; this is the honest MVP the eval
harness (Phase 4) will measure a baseline against.
"""

from __future__ import annotations

from sec_10k_agent.rag.llm import Generator, build_generator
from sec_10k_agent.rag.models import Answer
from sec_10k_agent.rag.prompts import SYSTEM_PROMPT, build_user_prompt, parse_citations
from sec_10k_agent.retrieval.models import SearchRetriever
from sec_10k_agent.retrieval.retriever import DEFAULT_K, Retriever


class RAGPipeline:
    """Answer questions over the 10-K corpus with cited, retrieval-grounded output."""

    def __init__(
        self,
        retriever: SearchRetriever | None = None,
        generator: Generator | None = None,
        k: int = DEFAULT_K,
    ) -> None:
        self._retriever = retriever or Retriever()
        self._generator = generator or build_generator()
        self._k = k

    def answer(
        self,
        question: str,
        *,
        ticker: str | None = None,
        fiscal_year: int | None = None,
        section: str | None = None,
        k: int | None = None,
        temperature: float = 0.0,
    ) -> Answer:
        """Retrieve, generate, and return a cited Answer.

        Optional filters narrow retrieval to a ticker / fiscal year / section.
        If retrieval returns nothing, the pipeline short-circuits with an honest
        "no sources" answer rather than prompting the model with empty context.
        """
        chunks = self._retriever.search(
            question, ticker=ticker, fiscal_year=fiscal_year, section=section, k=k or self._k
        )
        if not chunks:
            return Answer(
                question=question,
                text="I don't have enough information in the provided filings to answer that.",
                sources=[],
            )

        user_prompt = build_user_prompt(question, chunks)
        response = self._generator.complete(SYSTEM_PROMPT, user_prompt, temperature=temperature)
        cited = parse_citations(response.text, len(chunks))
        return Answer(
            question=question,
            text=response.text,
            sources=chunks,
            cited_indices=cited,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
