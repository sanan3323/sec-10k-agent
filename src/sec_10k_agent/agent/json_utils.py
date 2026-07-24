"""Shared helper for extracting a JSON object from an LLM reply.

Every agent node that asks the generator for structured output (router,
decomposer, synthesizer, verifier) hits the same problem: models wrap JSON in
prose or code fences even when told not to. One tolerant extractor, reused
everywhere, instead of four slightly different regexes.
"""

from __future__ import annotations

import json
import re

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> dict:
    """Find and parse the first `{...}` block in `text`. Raises ValueError if
    none is found or it doesn't parse as JSON."""
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError(f"no JSON object in reply: {text!r}")
    return dict(json.loads(match.group(0)))
