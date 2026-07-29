from functools import cached_property
from pathlib import Path
from typing import Final
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class settings(BaseSettings):
    """
    Centralized, validated application configuration.
    Reads values from environment variables / a .env file at startup,
    so a missing or malformed credential fails immediately and loudly
    instead of causing a confusing error deep inside a webhook handler.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # don't error if .env has keys we haven't declared yet
    )

    # --- GitHub App credentials (Phase 3 / 4) ---
    GITHUB_APP_ID: int
    GITHUB_PRIVATE_KEY_PATH: Path
    GITHUB_WEBHOOK_SECRET: str

    # --- ML / Vector DB config (owned by the Architect side, DO NOT rename) ---
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHROMA_PATH: str = "./data/chroma"
    COLLECTION_NAME: str = "openlake_issues"
    DIRECTORY_FALLBACK_WEIGHT: Final[float] = 0.8

    # --- Duplicate detection tuning ---
    SIMILARITY_THRESHOLD: float = Field(default=0.85, ge=0.0, le=1.0)

    @field_validator("GITHUB_PRIVATE_KEY_PATH")
    @classmethod
    def _validate_key_exists(cls, path: Path) -> Path:
        """Fail at startup, not mid-request, if the key file is missing."""
        if not path.exists():
            raise ValueError(f"GitHub private key not found at: {path}")
        return path

    @cached_property
    def GITHUB_PRIVATE_KEY(self) -> str:
        """
        Lazily reads the PEM file contents once, then caches it in memory
        for the lifetime of the app. Avoids re-reading disk on every request.
        """
        return self.GITHUB_PRIVATE_KEY_PATH.read_text()


Setting = settings()  # type: ignore
