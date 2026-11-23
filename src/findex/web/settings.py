"""Environment-driven web service configuration via pydantic-settings (Lab 7)."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the findex HTTP service loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    index_path: Path | None = Field(
        default=None,
        description="Path to the persisted findex index file (env: INDEX_PATH).",
    )
    index_url: str | None = Field(
        default=None,
        description=(
            "Optional URL to download the index file on startup when INDEX_PATH is absent "
            "(env: INDEX_URL)."
        ),
    )
    host: str = Field(
        default="127.0.0.1",
        description="Bind host address (env: HOST).",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Listening TCP port (env: PORT).",
    )
    workers: int = Field(
        default=1,
        ge=1,
        description="Number of uvicorn worker processes (env: WORKERS).",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level for the web service (env: LOG_LEVEL).",
    )
