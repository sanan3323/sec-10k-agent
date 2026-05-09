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

from typing import Protocol


class TokenCounter(Protocol):
    """Counts tokens in a string."""

    def count(self, text: str) -> int: ...


class WordCountTokenCounter:
    """Deterministic stand-in for tests. Counts whitespace-separated words.

    Not used in production. The chunker's behavior is identical with either
    counter, but absolute token thresholds need to be tuned per counter.
    """

    def count(self, text: str) -> int:
        return len(text.split())


class BgeTokenCounter:
    """Counts BGE tokens. Lazy-loads the tokenizer on first use.

    The tokenizer is loaded once per process and shared across calls. We use
    `transformers.AutoTokenizer` directly rather than the full
    `sentence_transformers.SentenceTransformer` so we don't pay for model
    weights we don't need at chunking time.
    """

    _MODEL_NAME = "BAAI/bge-large-en-v1.5"

    def __init__(self) -> None:
        self._tokenizer = None  # lazy

    def count(self, text: str) -> int:
        if self._tokenizer is None:
            # Imported lazily so package import doesn't pull transformers.
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._MODEL_NAME)
        # add_special_tokens=False so we count content tokens only. The
        # caller's max-tokens budget should already account for [CLS]/[SEP].
        return len(self._tokenizer.encode(text, add_special_tokens=False))
