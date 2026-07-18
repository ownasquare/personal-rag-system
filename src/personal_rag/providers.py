"""Explicit paid-provider construction without process-global LlamaIndex state."""

from __future__ import annotations

from typing import cast

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.openai import OpenAI
from voyageai.client import Client as VoyageClient
from voyageai.client_async import AsyncClient as AsyncVoyageClient

from personal_rag.config import Settings
from personal_rag.errors import ConfigurationError


def build_embedding(settings: Settings) -> BaseEmbedding:
    """Build the configured embedding client with credentials passed explicitly.

    Keeping this factory explicit prevents LlamaIndex ``Settings`` mutations from
    leaking between API requests, worker processes, or tests.
    """

    if settings.embedding_provider == "openai":
        if settings.openai_api_key is None:
            raise ConfigurationError("OpenAI embedding key is not configured")
        return cast(
            BaseEmbedding,
            OpenAIEmbedding(
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
                api_key=settings.openai_api_key.get_secret_value(),
                timeout=settings.provider_timeout_seconds,
                max_retries=settings.provider_max_retries,
            ),
        )

    if settings.voyage_api_key is None:
        raise ConfigurationError("Voyage embedding key is not configured")
    api_key = settings.voyage_api_key.get_secret_value()
    embedding = VoyageEmbedding(
        model_name=settings.embedding_model,
        voyage_api_key=api_key,
        output_dimension=settings.embedding_dimensions,
        # Never let the provider silently discard input. The application chunker
        # owns the context-window boundary and an oversize request must be visible.
        truncation=False,
    )
    # LlamaIndex's Voyage integration does not expose the SDK's timeout/retry
    # arguments, so replace its default clients with explicitly bounded ones.
    embedding._client = VoyageClient(
        api_key=api_key,
        timeout=settings.provider_timeout_seconds,
        max_retries=settings.provider_max_retries,
    )
    embedding._aclient = AsyncVoyageClient(
        api_key=api_key,
        timeout=settings.provider_timeout_seconds,
        max_retries=settings.provider_max_retries,
    )
    return cast(BaseEmbedding, embedding)


def build_llm(settings: Settings) -> LLM:
    """Build the answer model. Retrieval embeddings and answers stay separate."""

    if settings.chat_provider != "openai":  # defensive if the settings contract widens
        raise ConfigurationError(f"Unsupported chat provider: {settings.chat_provider}")
    if settings.openai_api_key is None:
        raise ConfigurationError("OpenAI answer-model key is not configured")
    return OpenAI(
        model=settings.chat_model,
        temperature=settings.chat_temperature,
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.provider_timeout_seconds,
        max_retries=settings.provider_max_retries,
    )
