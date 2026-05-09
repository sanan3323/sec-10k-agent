"""Project scope — tickers and fiscal years in scope for v1.

10 tickers × 5 fiscal years = ~50 10-K filings. Diverse on purpose:
- Tech hardware/software/semis: AAPL, NVDA
- Financials: JPM (very different 10-K structure)

Five fiscal years (2021–2025), the most recent five FYs as of project start
in May 2026. Almost every issuer in scope has filed a 10-K covering FY2025
by now (AAPL, NVDA, etc. file early in the calendar year following their FY
end). This window gives us:
- Supply-chain crisis era (2021–2022)
- Post-pandemic normalization and macro tightening (2022–2023)
- The AI-disclosure inflection (2023–2024)
- Mature AI-era risk-factor language (2024–2025), which is the most
  interesting linguistic regime to test temporal reasoning on.

We deliberately avoid 2020 because including a single COVID-shock year
distorts comparisons without adding a temporal regime we couldn't get from
later years; the trade is not worth the staleness on the recent end.
"""
# ruff: noqa: RUF002

from __future__ import annotations

from typing import Final

TICKERS: Final[tuple[str, ...]] = ("AAPL", "NVDA", "JPM")

FISCAL_YEARS: Final[tuple[int, ...]] = (2021, 2022, 2023, 2024, 2025)

# Convenience: the full (ticker, fy) matrix the project covers.
SCOPE: Final[tuple[tuple[str, int], ...]] = tuple((t, y) for t in TICKERS for y in FISCAL_YEARS)
