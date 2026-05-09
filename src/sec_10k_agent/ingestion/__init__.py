"""Ingestion: download, parse, chunk, and extract XBRL from 10-K filings.

Public API:
    Filing, Chunk, XBRLFact   — data models passed between stages
    EdgarClient               — throttled, cached, retried EDGAR client
    EdgarError, FilingNotFound

Internals (rate limiter, edgartools adapter) live in submodules and are not
re-exported.
"""

from sec_10k_agent.ingestion.chunker import Chunker
from sec_10k_agent.ingestion.edgar_client import (
    EdgarClient,
    EdgarError,
    FilingNotFound,
)
from sec_10k_agent.ingestion.models import Chunk, Filing, XBRLFact
from sec_10k_agent.ingestion.parsed_filing import (
    ParsedFiling,
    Section,
    read_from_cache,
    write_to_cache,
)
from sec_10k_agent.ingestion.parser import PARSER_VERSION, FilingParser
from sec_10k_agent.ingestion.tokenizer import (
    BgeTokenCounter,
    TokenCounter,
    WordCountTokenCounter,
)

__all__ = [
    "PARSER_VERSION",
    "BgeTokenCounter",
    "Chunk",
    "Chunker",
    "EdgarClient",
    "EdgarError",
    "Filing",
    "FilingNotFound",
    "FilingParser",
    "ParsedFiling",
    "Section",
    "TokenCounter",
    "WordCountTokenCounter",
    "XBRLFact",
    "read_from_cache",
    "write_to_cache",
]
