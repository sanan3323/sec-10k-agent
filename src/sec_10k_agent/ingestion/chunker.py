"""Chunker.

Turns a `ParsedFiling` into a list of `Chunk`s ready to embed.

Rules (no token-level overlap; context expansion happens at retrieval time
via prev/next pointers):

1. Default: one chunk per paragraph.
2. A paragraph longer than `max_tokens` is split at sentence boundaries
   into pieces, each under `max_tokens`. The pieces stay in order with
   prev/next pointers, so a retrieved piece can pull its neighbors back at
   synthesis time.
3. A paragraph shorter than `min_tokens` is merged forward into the next
   paragraph in the SAME section. Merges never cross an Item boundary —
   a short trailing paragraph in Item 1 does not glue to the start of
   Item 1A.
4. Hard cap: no chunk exceeds `max_tokens`. Asserted at the end.

Token thresholds default to BGE-large limits:
- max_tokens=480 (leaves headroom under BGE's 512 hard limit for special
  tokens [CLS]/[SEP] added at embed time)
- target_tokens=400 (the soft size we aim for when splitting big paragraphs)
- min_tokens=80 (below this, merge forward)
"""

from __future__ import annotations

import re

from sec_10k_agent.ingestion.models import Chunk
from sec_10k_agent.ingestion.parsed_filing import ParsedFiling, Section
from sec_10k_agent.ingestion.tokenizer import TokenCounter

# Sentence boundary heuristic: split after `.?!` followed by whitespace and
# a capital letter. Imperfect (e.g. "U.S. Securities" splits) but our goal
# here is to break a too-long paragraph into pieces under a token budget,
# not to do linguistic analysis. The occasional weird boundary is fine.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


class Chunker:
    def __init__(
        self,
        token_counter: TokenCounter,
        max_tokens: int = 480,
        target_tokens: int = 400,
        min_tokens: int = 80,
    ) -> None:
        if not (0 < min_tokens < target_tokens <= max_tokens):
            raise ValueError(
                "Need: 0 < min_tokens < target_tokens <= max_tokens. "
                f"Got min={min_tokens}, target={target_tokens}, max={max_tokens}."
            )
        self._tok = token_counter
        self._max = max_tokens
        self._target = target_tokens
        self._min = min_tokens

    def chunk(self, parsed: ParsedFiling) -> list[Chunk]:
        """Produce chunks for every section, in document order, with prev/
        next pointers wired up across the whole filing."""
        all_chunks: list[Chunk] = []
        for section in parsed.sections:
            section_chunks = self._chunk_section(parsed, section)
            all_chunks.extend(section_chunks)

        _wire_prev_next(all_chunks)
        self._assert_size_invariants(all_chunks)
        return all_chunks

    # Per-section chunking

    def _chunk_section(self, parsed: ParsedFiling, section: Section) -> list[Chunk]:
        # Step 1: convert paragraphs into "atoms" — pieces small enough to
        # be a chunk on their own. Big paragraphs get split here.
        atoms: list[str] = []
        for para in section.paragraphs:
            if self._tok.count(para) <= self._max:
                atoms.append(para)
            else:
                atoms.extend(self._split_oversized(para))

        # Step 2: merge tiny adjacent atoms forward, but never across the
        # section boundary (which is implicit — we're inside one section).
        merged = self._merge_tiny_forward(atoms)

        # Step 3: build Chunk objects.
        chunks: list[Chunk] = []
        for idx, text in enumerate(merged):
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(
                        parsed.filing.accession_number, section.display_section, idx
                    ),
                    cik=parsed.filing.cik,
                    ticker=parsed.filing.ticker,
                    fiscal_year=parsed.filing.fiscal_year,
                    accession_number=parsed.filing.accession_number,
                    section=section.display_section,
                    section_title=section.title,
                    text=text,
                    token_count=self._tok.count(text),
                )
            )
        return chunks

    # Oversized paragraph splitting

    def _split_oversized(self, paragraph: str) -> list[str]:
        """Split a too-long paragraph into pieces, each <= max_tokens."""
        sentences = _split_sentences(paragraph)
        pieces: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._tok.count(sentence)

            # If a single sentence is bigger than the hard limit, we must
            # hard-split it by words, otherwise it will fail the assertion.
            if sentence_tokens > self._max:
                # Flush the current buffer first
                if current:
                    pieces.append(" ".join(current))
                    current = []
                    current_tokens = 0

                # Hard-split the mega-sentence into pieces of ~target_tokens
                words = sentence.split()
                for i in range(0, len(words), self._target):
                    chunk_text = " ".join(words[i : i + self._target])
                    pieces.append(chunk_text)
                continue

            if current and current_tokens + sentence_tokens > self._target:
                pieces.append(" ".join(current))
                current = [sentence]
                current_tokens = sentence_tokens
            else:
                current.append(sentence)
                current_tokens += sentence_tokens

        if current:
            pieces.append(" ".join(current))

        return pieces

    # Tiny-paragraph merging

    def _merge_tiny_forward(self, atoms: list[str]) -> list[str]:
        """Merge each below-min atom into the next one. Merges that would
        push a chunk past max_tokens are skipped — the tiny atom stays as
        its own chunk in that case (rare: a tiny atom plus a near-full
        next atom).
        """
        if not atoms:
            return atoms

        out: list[str] = []
        i = 0
        while i < len(atoms):
            current = atoms[i]
            current_tokens = self._tok.count(current)

            # If this atom is below min and there's a next one, try to merge.
            while current_tokens < self._min and i + 1 < len(atoms):
                candidate = current + " " + atoms[i + 1]
                candidate_tokens = self._tok.count(candidate)
                if candidate_tokens > self._max:
                    break
                current = candidate
                current_tokens = candidate_tokens
                i += 1

            out.append(current)
            i += 1

        return out

    # Final invariant check

    def _assert_size_invariants(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            assert c.token_count <= self._max, (
                f"chunk {c.chunk_id} exceeds max ({c.token_count} > {self._max})"
            )


# helpers


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_id(accession_number: str, display_section: str, idx: int) -> str:
    """Format: '{accession}__{section_slug}__{idx:04d}'.

    Section slug replaces spaces with underscores so chunk IDs are
    URL-safe and grep-friendly. The display form ('Item 1A') is preserved
    in the Chunk.section field for citations.
    """
    section_slug = display_section.replace(" ", "_")
    return f"{accession_number}__{section_slug}__{idx:04d}"


def _wire_prev_next(chunks: list[Chunk]) -> None:
    """Set prev_chunk_id / next_chunk_id across the full chunk list.

    The chain spans Item boundaries: chunk N+1 is the next document chunk,
    even if it's in a different section. That's deliberate — at retrieval
    time we want context expansion to follow document order, including
    across Items, since a question may span sections.
    """
    for i, c in enumerate(chunks):
        c.prev_chunk_id = chunks[i - 1].chunk_id if i > 0 else None
        c.next_chunk_id = chunks[i + 1].chunk_id if i + 1 < len(chunks) else None
