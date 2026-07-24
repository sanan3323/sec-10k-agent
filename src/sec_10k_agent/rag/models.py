"""Types for the single-hop RAG pipeline (Phase 3)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from sec_10k_agent.retrieval.models import RetrievedChunk


class LLMResponse(BaseModel):
    """A raw completion from the generator, with token accounting."""

    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


class Answer(BaseModel):
    """The result of answering one question over the corpus.

    `sources` is the ordered list of retrieved chunks; a `[n]` marker in `text`
    refers to `sources[n-1]`. `cited_indices` are the 1-based markers the model
    actually used, so the UI can highlight (and a later phase can verify) them.
    """

    question: str
    text: str
    sources: list[RetrievedChunk] = Field(default_factory=list)
    cited_indices: list[int] = Field(default_factory=list)
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def cited_sources(self) -> list[RetrievedChunk]:
        """The subset of `sources` the answer actually cited, in citation order."""
        return [self.sources[i - 1] for i in self.cited_indices if 1 <= i <= len(self.sources)]
