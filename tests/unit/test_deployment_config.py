from __future__ import annotations

import tomllib
from pathlib import Path

from personal_rag.config import Settings

ROOT = Path(__file__).resolve().parents[2]


def test_streamlit_hardening_is_copied_into_the_runtime_image() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    streamlit_config = tomllib.loads(
        (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    )

    assert "COPY --chown=rag:rag .streamlit ./.streamlit" in dockerfile
    assert streamlit_config["server"]["enableXsrfProtection"] is True
    assert streamlit_config["client"]["showErrorDetails"] is False
    assert (
        streamlit_config["server"]["maxUploadSize"] * 1024 * 1024
        == Settings(_env_file=None).upload_max_bytes
    )


def test_compose_maps_security_and_runtime_tuning_to_the_right_services() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "qdrant/qdrant:v1.18.3-unprivileged" in compose
    assert 'RAG_QDRANT_PORT: "6333"' in compose
    assert "127.0.0.1:${RAG_QDRANT_HOST_PORT:-6333}:6333" in compose
    assert "QDRANT__SERVICE__API_KEY" in compose
    for setting_name in (
        "RAG_CHUNK_SIZE",
        "RAG_UPLOAD_MAX_BYTES",
        "RAG_RETRIEVAL_MAX_TOP_K",
        "RAG_JOB_LEASE_SECONDS",
        "RAG_PROVIDER_TIMEOUT_SECONDS",
        "RAG_METRICS_ENABLED",
    ):
        assert setting_name in compose

    ui_section = compose.split("  ui:", maxsplit=1)[1]
    assert "RAG_OPENAI_API_KEY" not in ui_section
    assert "RAG_VOYAGE_API_KEY" not in ui_section
