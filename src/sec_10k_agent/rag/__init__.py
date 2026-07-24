"""sec_10k_agent.rag — single-hop retrieval-augmented generation (Phase 3).

See docs/architecture.md. The agent layer (Phase 6) orchestrates on top of this.
"""

from __future__ import annotations

from sec_10k_agent.rag.llm import (
    Generator,
    OpenAICompatGenerator,
    build_generator,
)
from sec_10k_agent.rag.models import Answer, LLMResponse
from sec_10k_agent.rag.pipeline import RAGPipeline
from sec_10k_agent.rag.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    format_context,
    parse_citations,
)

__all__ = [
    "SYSTEM_PROMPT",
    "Answer",
    "Generator",
    "LLMResponse",
    "OpenAICompatGenerator",
    "RAGPipeline",
    "build_generator",
    "build_user_prompt",
    "format_context",
    "parse_citations",
]
