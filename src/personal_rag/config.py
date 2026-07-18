"""Typed configuration and immutable embedding-profile contracts."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingProfile(BaseModel):
    """All inputs that make an existing vector collection compatible."""

    provider: Literal["openai", "voyage"]
    model: str
    dimensions: int = Field(ge=1, le=4096)
    distance_metric: Literal["cosine"] = "cosine"
    parser_version: str = "1"
    chunker: str = "llamaindex-sentence-splitter"
    chunk_size: int = Field(ge=128, le=4096)
    chunk_overlap: int = Field(ge=0, le=1024)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a local `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Personal Knowledge Studio"
    environment: Literal["development", "test", "production"] = "development"
    host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    ui_port: int = Field(default=8501, ge=1, le=65535)
    api_url: str = "http://127.0.0.1:8000"

    auth_enabled: bool = False
    api_key: SecretStr | None = None
    cors_origins: list[str] = Field(default_factory=lambda: ["http://127.0.0.1:8501"])

    data_dir: Path = Path(".data")
    database_filename: str = "personal_rag.sqlite3"

    qdrant_mode: Literal["http", "persistent", "memory"] = "http"
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = Field(default=6333, ge=1, le=65535)
    qdrant_https: bool = False
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "personal_knowledge"

    embedding_provider: Literal["openai", "voyage"] = "openai"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = Field(default=3072, ge=1, le=4096)
    openai_api_key: SecretStr | None = None
    voyage_api_key: SecretStr | None = None

    chat_provider: Literal["openai"] = "openai"
    chat_model: str = "gpt-4.1-mini"
    chat_temperature: float = Field(default=0.1, ge=0.0, le=1.0)

    chunk_size: int = Field(default=768, ge=128, le=4096)
    chunk_overlap: int = Field(default=96, ge=0, le=1024)
    parser_version: str = "1"
    upload_max_bytes: int = Field(
        default=25 * 1024 * 1024,
        ge=1024,
        le=25 * 1024 * 1024,
    )
    upload_read_bytes: int = Field(default=1024 * 1024, ge=4096)
    max_pdf_pages: int = Field(default=500, ge=1, le=5000)
    max_extracted_characters: int = Field(default=2_000_000, ge=1000)

    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_max_top_k: int = Field(default=10, ge=1, le=50)
    retrieval_min_score: float = Field(default=0.1, ge=-1.0, le=1.0)
    max_query_characters: int = Field(default=4000, ge=1, le=4000)
    max_history_messages: int = Field(default=12, ge=0, le=100)
    citation_snippet_characters: int = Field(default=420, ge=80, le=2000)

    job_max_attempts: int = Field(default=3, ge=1, le=10)
    job_lease_seconds: int = Field(default=300, ge=10, le=3600)
    worker_poll_seconds: float = Field(default=1.0, ge=0.1, le=60.0)
    worker_stale_seconds: int = Field(default=90, ge=5, le=3600)

    provider_timeout_seconds: float = Field(default=60.0, ge=1.0, le=600.0)
    provider_max_retries: int = Field(default=3, ge=0, le=10)
    ui_poll_seconds: float = Field(default=1.0, ge=0.2, le=10.0)
    ui_poll_timeout_seconds: float = Field(default=180.0, ge=5.0, le=3600.0)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_logs: bool = True
    metrics_enabled: bool = True

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> object:
        if isinstance(value, str) and not value.lstrip().startswith("["):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator(
        "api_key",
        "qdrant_api_key",
        "openai_api_key",
        "voyage_api_key",
        mode="before",
    )
    @classmethod
    def normalize_optional_secrets(cls, value: object) -> object:
        if value is None:
            return None
        raw = value.get_secret_value() if isinstance(value, SecretStr) else str(value)
        cleaned = raw.strip()
        if not cleaned:
            return None
        if cleaned.casefold().startswith(("change-me", "replace-me")):
            raise ValueError("placeholder secret values must be replaced")
        return cleaned

    @field_validator("qdrant_collection")
    @classmethod
    def validate_collection_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not 3 <= len(cleaned) <= 63:
            raise ValueError("qdrant_collection must contain 3 to 63 characters")
        if not cleaned.replace("_", "").replace("-", "").isalnum():
            raise ValueError("qdrant_collection may contain letters, numbers, _ and -")
        return cleaned

    @model_validator(mode="after")
    def validate_cross_field_contracts(self) -> Settings:
        if self.environment == "production" and not self.auth_enabled:
            raise ValueError("authentication must be enabled in production")
        if self.environment == "production" and self.qdrant_mode != "http":
            raise ValueError("production requires RAG_QDRANT_MODE=http")
        if (
            self.environment == "production"
            and self.qdrant_mode == "http"
            and self.qdrant_api_key is None
        ):
            raise ValueError("RAG_QDRANT_API_KEY is required for production Qdrant HTTP mode")
        if (
            self.environment == "production"
            and self.qdrant_mode == "http"
            and self.qdrant_api_key is not None
            and len(self.qdrant_api_key.get_secret_value()) < 24
        ):
            raise ValueError("RAG_QDRANT_API_KEY must contain at least 24 characters")
        if self.auth_enabled and self.api_key is None:
            raise ValueError("RAG_API_KEY is required when authentication is enabled")
        if (
            self.auth_enabled
            and self.api_key is not None
            and len(self.api_key.get_secret_value()) < 24
        ):
            raise ValueError("RAG_API_KEY must contain at least 24 characters")
        if (
            self.environment == "production"
            and self.api_key is not None
            and self.qdrant_api_key is not None
            and self.api_key.get_secret_value() == self.qdrant_api_key.get_secret_value()
        ):
            raise ValueError("RAG_API_KEY and RAG_QDRANT_API_KEY must be different values")
        for name, secret in (
            ("RAG_OPENAI_API_KEY", self.openai_api_key),
            ("RAG_VOYAGE_API_KEY", self.voyage_api_key),
        ):
            if secret is not None and len(secret.get_secret_value()) < 12:
                raise ValueError(f"{name} must contain at least 12 characters")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.retrieval_top_k > self.retrieval_max_top_k:
            raise ValueError("retrieval_top_k cannot exceed retrieval_max_top_k")
        if self.embedding_provider == "openai" and not self.embedding_model.startswith(
            "text-embedding-"
        ):
            raise ValueError("OpenAI embeddings require a text-embedding-* model")
        if self.embedding_provider == "voyage" and not self.embedding_model.startswith("voyage-"):
            raise ValueError("Voyage embeddings require a voyage-* model")
        if self.environment == "production" and self.provider_configuration_errors:
            raise ValueError("; ".join(self.provider_configuration_errors))
        return self

    @property
    def database_path(self) -> Path:
        return self.data_dir / self.database_filename

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def staging_dir(self) -> Path:
        return self.data_dir / "staging"

    @property
    def qdrant_persist_dir(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def embedding_profile(self) -> EmbeddingProfile:
        return EmbeddingProfile(
            provider=self.embedding_provider,
            model=self.embedding_model,
            dimensions=self.embedding_dimensions,
            parser_version=self.parser_version,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    @property
    def provider_configuration_errors(self) -> list[str]:
        errors: list[str] = []
        if self.embedding_provider == "openai" and self.openai_api_key is None:
            errors.append("OpenAI embedding key is not configured")
        if self.embedding_provider == "voyage" and self.voyage_api_key is None:
            errors.append("Voyage embedding key is not configured")
        if self.chat_provider == "openai" and self.openai_api_key is None:
            errors.append("OpenAI answer-model key is not configured")
        return errors

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.uploads_dir, self.staging_dir):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)

    def sanitized_summary(self) -> dict[str, object]:
        return {
            "environment": self.environment,
            "auth_enabled": self.auth_enabled,
            "qdrant_mode": self.qdrant_mode,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "chat_provider": self.chat_provider,
            "chat_model": self.chat_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "collection": self.qdrant_collection,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one immutable settings object per process."""

    return Settings()
