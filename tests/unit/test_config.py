from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from personal_rag.config import EmbeddingProfile, Settings


def test_embedding_profile_fingerprint_is_stable_and_complete() -> None:
    profile = EmbeddingProfile(
        provider="openai",
        model="text-embedding-3-large",
        dimensions=3072,
        chunk_size=768,
        chunk_overlap=96,
    )

    assert profile.fingerprint == profile.model_copy().fingerprint
    assert len(profile.fingerprint) == 64
    assert profile.model_copy(update={"chunk_size": 1024}).fingerprint != profile.fingerprint


def test_settings_enforces_auth_and_embedding_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="RAG_API_KEY"):
        Settings(data_dir=tmp_path, auth_enabled=True, api_key=None, _env_file=None)

    with pytest.raises(ValidationError, match="text-embedding"):
        Settings(
            auth_enabled=False,
            data_dir=tmp_path,
            embedding_provider="openai",
            embedding_model="voyage-3-large",
        )

    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings(
            auth_enabled=False,
            data_dir=tmp_path,
            chunk_size=128,
            chunk_overlap=128,
        )


def test_settings_builds_private_directories_and_sanitized_summary(tmp_path: Path) -> None:
    settings = Settings(
        auth_enabled=False,
        data_dir=tmp_path / "private-data",
        openai_api_key="not-for-output-value",
    )

    settings.ensure_directories()

    assert settings.database_path == settings.data_dir / "personal_rag.sqlite3"
    assert settings.uploads_dir.is_dir()
    assert settings.staging_dir.is_dir()
    summary_text = repr(settings.sanitized_summary())
    assert "not-for-output" not in summary_text
    assert "openai_api_key" not in summary_text


def test_production_http_qdrant_requires_a_separate_api_key(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="RAG_QDRANT_API_KEY"):
        Settings(
            environment="production",
            auth_enabled=True,
            api_key="application-token-at-least-24-characters",
            qdrant_mode="http",
            qdrant_api_key=None,
            openai_api_key="offline-openai-provider-key",
            data_dir=tmp_path,
            _env_file=None,
        )

    settings = Settings(
        environment="production",
        auth_enabled=True,
        api_key="application-token-at-least-24-characters",
        qdrant_mode="http",
        qdrant_api_key="different-vector-token-at-least-24-characters",
        openai_api_key="offline-openai-provider-key",
        data_dir=tmp_path,
        _env_file=None,
    )

    assert settings.qdrant_api_key is not None
    assert "different-vector-token" not in repr(settings.sanitized_summary())


def test_blank_provider_keys_are_not_reported_as_configured(tmp_path: Path) -> None:
    settings = Settings(
        auth_enabled=False,
        data_dir=tmp_path,
        openai_api_key="   ",
        voyage_api_key="",
        _env_file=None,
    )

    assert settings.openai_api_key is None
    assert settings.voyage_api_key is None
    assert settings.provider_configuration_errors


def test_placeholder_and_short_enabled_auth_tokens_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="placeholder"):
        Settings(
            auth_enabled=True,
            api_key="change-me-generate-a-long-random-value",
            data_dir=tmp_path,
            _env_file=None,
        )

    with pytest.raises(ValidationError, match="at least 24"):
        Settings(
            auth_enabled=True,
            api_key="too-short",
            data_dir=tmp_path,
            _env_file=None,
        )


def test_production_rejects_reused_application_and_qdrant_tokens(tmp_path: Path) -> None:
    repeated = "one-token-must-not-protect-two-services"
    with pytest.raises(ValidationError, match="must be different"):
        Settings(
            environment="production",
            auth_enabled=True,
            api_key=repeated,
            qdrant_mode="http",
            qdrant_api_key=repeated,
            openai_api_key="offline-openai-provider-key",
            data_dir=tmp_path,
            _env_file=None,
        )


@pytest.mark.parametrize("qdrant_mode", ["memory", "persistent"])
def test_production_requires_shared_http_qdrant(tmp_path: Path, qdrant_mode: str) -> None:
    with pytest.raises(ValidationError, match="RAG_QDRANT_MODE=http"):
        Settings(
            environment="production",
            auth_enabled=True,
            api_key="application-token-at-least-24-characters",
            qdrant_mode=qdrant_mode,  # type: ignore[arg-type]
            openai_api_key="offline-openai-provider-key",
            data_dir=tmp_path,
            _env_file=None,
        )


def test_production_requires_embedding_and_answer_provider_keys(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="OpenAI embedding key"):
        Settings(
            environment="production",
            auth_enabled=True,
            api_key="application-token-at-least-24-characters",
            qdrant_mode="http",
            qdrant_api_key="different-vector-token-at-least-24-characters",
            openai_api_key=None,
            data_dir=tmp_path,
            _env_file=None,
        )

    with pytest.raises(ValidationError, match="OpenAI answer-model key"):
        Settings(
            environment="production",
            auth_enabled=True,
            api_key="application-token-at-least-24-characters",
            qdrant_mode="http",
            qdrant_api_key="different-vector-token-at-least-24-characters",
            embedding_provider="voyage",
            embedding_model="voyage-3-large",
            embedding_dimensions=1024,
            voyage_api_key="offline-voyage-provider-key",
            openai_api_key=None,
            data_dir=tmp_path,
            _env_file=None,
        )


def test_cors_origins_accepts_a_comma_separated_value(tmp_path: Path) -> None:
    settings = Settings(
        auth_enabled=False,
        data_dir=tmp_path,
        cors_origins="https://one.example, https://two.example",  # type: ignore[arg-type]
    )

    assert settings.cors_origins == ["https://one.example", "https://two.example"]


def test_ui_and_api_hard_limits_cannot_be_configured_past_schema_limits(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="less than or equal to 4000"):
        Settings(
            auth_enabled=False,
            data_dir=tmp_path,
            max_query_characters=4001,
            _env_file=None,
        )

    with pytest.raises(ValidationError, match="less than or equal to 26214400"):
        Settings(
            auth_enabled=False,
            data_dir=tmp_path,
            upload_max_bytes=(25 * 1024 * 1024) + 1,
            _env_file=None,
        )
