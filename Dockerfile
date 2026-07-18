# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.8.17 AS uv

FROM python:3.14.6-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    RAG_DATA_DIR=/data

RUN groupadd --gid 10001 rag \
    && useradd --uid 10001 --gid rag --create-home --shell /usr/sbin/nologin rag \
    && mkdir -p /app /data \
    && chown -R rag:rag /app /data

COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /app

COPY --chown=rag:rag pyproject.toml uv.lock README.md LICENSE ./
COPY --chown=rag:rag .streamlit ./.streamlit
COPY --chown=rag:rag src ./src
RUN uv sync --frozen --no-dev --no-editable --no-cache

USER rag
EXPOSE 8000 8501

HEALTHCHECK --interval=20s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3).read()"]

CMD ["uvicorn", "personal_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
