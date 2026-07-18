"""Explicitly opted-in live connectivity checks for paid model providers."""

from __future__ import annotations

import math
import os

import pytest

from personal_rag.config import Settings
from personal_rag.providers import build_embedding, build_llm

pytestmark = pytest.mark.live

_LIVE_TESTS_ENABLED = os.getenv("RAG_RUN_LIVE_TESTS") == "1"


@pytest.mark.skipif(
    not _LIVE_TESTS_ENABLED,
    reason="set RAG_RUN_LIVE_TESTS=1 to allow paid live-provider requests",
)
def test_selected_embedding_and_openai_answer_provider_are_reachable() -> None:
    """Make one bounded embedding call and one bounded OpenAI answer call."""

    settings = Settings(
        environment="test",
        auth_enabled=False,
        qdrant_mode="memory",
        provider_max_retries=0,
    )
    configuration_errors = settings.provider_configuration_errors
    if configuration_errors:
        pytest.fail(
            "Live provider tests were enabled, but required credentials are missing: "
            + "; ".join(configuration_errors),
            pytrace=False,
        )

    embedding = build_embedding(settings)
    llm = build_llm(settings)

    try:
        vector = embedding.get_text_embedding("Personal RAG live-provider connectivity check.")
    except Exception as error:
        pytest.fail(
            "The selected embedding provider request failed "
            f"({type(error).__name__}); provider response content was suppressed.",
            pytrace=False,
        )

    if len(vector) != settings.embedding_dimensions or not all(
        math.isfinite(value) for value in vector
    ):
        pytest.fail(
            "The selected embedding provider returned an invalid vector shape or values.",
            pytrace=False,
        )

    try:
        completion = llm.complete(
            "This is a live connectivity check. Reply with the single word READY."
        )
    except Exception as error:
        pytest.fail(
            "The OpenAI answer-provider request failed "
            f"({type(error).__name__}); provider response content was suppressed.",
            pytrace=False,
        )

    response_text = getattr(completion, "text", None)
    if not isinstance(response_text, str) or not response_text.strip():
        pytest.fail(
            "The OpenAI answer provider returned an empty response.",
            pytrace=False,
        )
