"""Token counting for chunk sizing.

Chunk size is measured in BGE tokens because BGE-large is the embedding
model. BGE has a hard 512-token context window, and anything beyond that is
silently truncated at embed time — so the chunker's hard cap is set in BGE
tokens, not OpenAI tokens, not characters, not words.

Two implementations:

- BgeTokenCounter: real, lazy-loads `BAAI/bge-large-en-v1.5`'s tokenizer once
  and reuses it. The ~50 MB tokenizer download (not the ~1.3 GB model) is
  the only network cost.

- WordCountTokenCounter: deterministic test stand-in. Word count is a rough
  approximation of BGE token count (typically 0.7-1.5x). Tests use it so CI
  doesn't pull HuggingFace on every run.

The Protocol means callers stay model-agnostic. Swapping to a different
embedding model in Phase 5 means writing a new TokenCounter, not touching
the chunker.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for counting tokens in a string."""

    def count(self, text: str) -> int: ...


class WordCountTokenCounter:
    """Deterministic stand-in for tests. Counts whitespace-separated words."""

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split())


class BgeTokenCounter:
    """Counts BGE tokens. Lazy-loads the tokenizer on first use."""

    _MODEL_NAME = "BAAI/bge-large-en-v1.5"

    def __init__(self) -> None:
        # We use Any to avoid complex type stubs for the transformers library
        self._tokenizer: Any = None

    def count(self, text: str) -> int:
        if not text:
            return 0

        if self._tokenizer is None:
            # Lazy import to keep the initial load fast
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._MODEL_NAME)

        # Local reference for type narrowing
        tok = self._tokenizer
        if tok is None:
            raise RuntimeError("Failed to load BGE tokenizer")

        return len(tok.encode(text, add_special_tokens=False))
