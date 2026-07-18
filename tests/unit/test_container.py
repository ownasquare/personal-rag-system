from __future__ import annotations

from pathlib import Path

import pytest

import personal_rag.container as container_module
from personal_rag.config import Settings
from personal_rag.errors import ConfigurationError


class FakeVectorStore:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_development_container_validates_providers_before_vector_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vector_store_called = False

    def unexpected_vector_store(_settings: Settings) -> None:
        nonlocal vector_store_called
        vector_store_called = True

    monkeypatch.setattr(container_module, "VectorStore", unexpected_vector_store)
    settings = Settings(
        environment="development",
        auth_enabled=False,
        qdrant_mode="memory",
        data_dir=tmp_path,
        _env_file=None,
    )

    container = container_module.build_container(settings)

    assert vector_store_called is False
    assert container.vector_store is None
    assert container.rag_service is None
    assert container.startup_errors == settings.provider_configuration_errors


def test_production_container_fails_fast_when_vector_storage_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable_vector_store(_settings: Settings) -> None:
        raise ConnectionError

    monkeypatch.setattr(container_module, "VectorStore", unavailable_vector_store)
    settings = Settings(
        environment="production",
        auth_enabled=True,
        api_key="application-token-long-enough-for-production",
        qdrant_mode="http",
        qdrant_api_key="different-qdrant-token-long-enough-for-production",
        openai_api_key="offline-openai-provider-key",
        data_dir=tmp_path,
        _env_file=None,
    )

    with pytest.raises(ConfigurationError, match="Vector storage"):
        container_module.build_container(settings)


def test_container_composes_rag_only_after_storage_and_providers_are_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vector_store = FakeVectorStore()
    embedding = object()
    llm = object()
    rag_service = object()
    monkeypatch.setattr(container_module, "VectorStore", lambda _settings: vector_store)
    monkeypatch.setattr(container_module, "build_embedding", lambda _settings: embedding)
    monkeypatch.setattr(container_module, "build_llm", lambda _settings: llm)
    monkeypatch.setattr(container_module, "RAGService", lambda *_args: rag_service)
    settings = Settings(
        environment="test",
        auth_enabled=False,
        qdrant_mode="memory",
        openai_api_key="offline-openai-provider-key",
        data_dir=tmp_path,
        _env_file=None,
    )

    container = container_module.build_container(settings)

    assert container.vector_store is vector_store
    assert container.rag_service is rag_service
    assert container.startup_errors == []
    container.close()
    assert vector_store.closed is True
