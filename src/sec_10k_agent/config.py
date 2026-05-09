"""Application configuration.

Loaded lazily via `get_settings()`. Importing this module never reads or
validates env vars — callers must invoke `get_settings()` to materialize the
config. This keeps `import sec_10k_agent` cheap and side-effect-free, so
`sec10k --help`, test discovery, and tooling that imports the package work
even when the environment isn't fully configured.

Usage:
    from sec_10k_agent.config import get_settings
    settings = get_settings()
    print(settings.sec_user_agent)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root, resolved once. data/, docs/, etc. are relative to this.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Global application settings.

    SEC EDGAR requires a User-Agent header on every request that identifies
    the requester. Without one, requests are rejected; with a misleading or
    missing one, IPs are banned. Validation here is fail-loud on first call
    to `get_settings()`.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # SEC
    sec_user_agent: str = Field(
        ...,
        description="REQUIRED. Format: 'Your Name your@email.com'.",
    )
    sec_rate_limit_per_sec: float = 5.0  # SEC ceiling is 10/s; we leave headroom.

    # LLMs
    # xAI (Grok) is OpenAI-SDK-compatible.
    xai_api_key: str | None = None
    xai_base_url: str = "https://api.x.ai/v1"
    xai_model: str = "grok-4.3"  # update when needed

    # LLM-as-judge for Phase 4 evals. Different model family from the generator
    # to avoid self-judge bias. Gemini Flash has a free tier sized for our
    # eval workloads. Swap to Ollama by leaving this blank and setting the
    # ollama_* fields below.
    gemini_api_key: str | None = None
    gemini_judge_model: str = "gemini-2.5-flash"

    # Optional Ollama fallback for the judge (e.g. running llama-3 locally).
    ollama_base_url: str | None = None
    ollama_judge_model: str = "llama3.1:8b"

    # Postgres + pgvector
    # One database for both vectors and metadata. See ADR-001.
    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/sec10k"

    # Observability
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"

    # Cache
    redis_url: str = "redis://localhost:6379/0"

    # App
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    env: Literal["dev", "staging", "prod"] = "dev"
    cost_alert_usd_per_day: float = 5.0

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    @field_validator("sec_user_agent")
    @classmethod
    def _validate_user_agent(cls, v: str) -> str:
        # SEC explicitly requires an email-like contact in the User-Agent.
        if "@" not in v or len(v.split()) < 2:
            raise ValueError(
                "SEC_USER_AGENT must include a name and email, e.g. 'Jane Doe jane@example.com'."
            )
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance.

    Cached so repeated calls are free. Tests that need a different config
    should call `get_settings.cache_clear()` after monkeypatching env vars.
    """
    return Settings()  # type: ignore[call-arg]
