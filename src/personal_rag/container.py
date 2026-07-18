"""Application composition root with provider-first startup validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from personal_rag.config import Settings
from personal_rag.database import Database
from personal_rag.errors import ConfigurationError, RagError
from personal_rag.observability import get_logger
from personal_rag.providers import build_embedding, build_llm
from personal_rag.rag_service import RAGService
from personal_rag.repository import Repository
from personal_rag.vector_store import VectorStore


@dataclass(slots=True)
class AppContainer:
    """Long-lived process resources shared by API requests."""

    settings: Settings
    database: Database
    repository: Repository
    vector_store: Any | None = None
    rag_service: Any | None = None
    startup_errors: list[str] = field(default_factory=list)

    def close(self) -> None:
        for resource in (self.vector_store, self.database):
            close = getattr(resource, "close", None)
            if callable(close):
                close()


def build_container(settings: Settings) -> AppContainer:
    """Build core resources without making paid provider health calls."""

    settings.ensure_directories()
    database = Database(settings.database_path)
    repository = Repository(
        database,
        lease_seconds=settings.job_lease_seconds,
        max_attempts=settings.job_max_attempts,
    )
    repository.initialize()
    container = AppContainer(settings=settings, database=database, repository=repository)

    if settings.provider_configuration_errors:
        if settings.environment == "production":
            container.close()
            raise ConfigurationError("Provider configuration is incomplete")
        container.startup_errors.extend(settings.provider_configuration_errors)
        return container

    try:
        embedding = build_embedding(settings)
        llm = build_llm(settings)
    except RagError as error:
        if settings.environment == "production":
            container.close()
            raise
        container.startup_errors.append(error.message)
        get_logger("startup").warning("rag_dependencies_degraded", code=error.code)
    except Exception as error:
        if settings.environment == "production":
            container.close()
            raise ConfigurationError("RAG dependencies could not be initialized") from error
        container.startup_errors.append("RAG dependencies could not be initialized")
        get_logger("startup").warning("rag_dependencies_degraded", error_type=type(error).__name__)
        return container

    try:
        vector_store = VectorStore(settings)
        container.vector_store = vector_store
    except RagError as error:
        if settings.environment == "production":
            container.close()
            raise
        container.startup_errors.append(error.message)
        get_logger("startup").warning("vector_store_degraded", code=error.code)
        return container
    except Exception as error:
        if settings.environment == "production":
            container.close()
            raise ConfigurationError("Vector storage could not be initialized") from error
        container.startup_errors.append("Vector storage could not be initialized")
        get_logger("startup").warning("vector_store_degraded", error_type=type(error).__name__)
        return container

    try:
        container.rag_service = RAGService(
            settings,
            container.vector_store,
            embedding,
            llm,
            repository,
        )
    except RagError as error:
        if settings.environment == "production":
            container.close()
            raise
        container.startup_errors.append(error.message)
        get_logger("startup").warning("rag_dependencies_degraded", code=error.code)
    except Exception as error:
        if settings.environment == "production":
            container.close()
            raise ConfigurationError("RAG dependencies could not be initialized") from error
        container.startup_errors.append("RAG dependencies could not be initialized")
        get_logger("startup").warning("rag_dependencies_degraded", error_type=type(error).__name__)
    return container
