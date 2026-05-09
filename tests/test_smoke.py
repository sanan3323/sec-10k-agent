"""Smoke tests — verify package imports cleanly and config validates correctly.

Real component tests arrive in Phase 1 alongside the ingestion code they cover.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sec_10k_agent import __version__
from sec_10k_agent.config import Settings, get_settings
from sec_10k_agent.scope import FISCAL_YEARS, SCOPE, TICKERS


def test_package_version_is_a_string() -> None:
    # We don't assert a specific value — version comes from package metadata
    # (or "0.0.0+local" when running uninstalled), so pinning a literal would
    # be brittle. We only care that the import path resolves to a string.
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_scope_is_well_formed() -> None:
    assert len(TICKERS) == 3
    assert len(FISCAL_YEARS) == 5
    assert len(SCOPE) == 15
    assert all(2021 <= y <= 2025 for _, y in SCOPE)


def test_settings_load_with_valid_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    # Clear the lru_cache so the cached singleton from a prior test
    # (or production startup) doesn't shadow our patched env.
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.sec_user_agent == "Test User test@example.com"
    assert settings.sec_rate_limit_per_sec == 5.0


def test_settings_reject_user_agent_without_email(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "no-email-here")
    get_settings.cache_clear()
    # Instantiating directly (not through get_settings) so the validation error
    # surfaces synchronously and we can assert on its content cleanly.
    with pytest.raises(ValidationError) as exc_info:
        Settings()  # type: ignore[call-arg]
    assert "SEC_USER_AGENT" in str(exc_info.value) or "email" in str(exc_info.value).lower()
