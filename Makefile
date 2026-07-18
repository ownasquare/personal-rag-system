.DEFAULT_GOAL := help

.PHONY: help sync lock test test-live coverage lint format typecheck security check api worker ui qdrant compose-up compose-down

help:
	@echo "sync          Install the locked runtime and development environment"
	@echo "test          Run deterministic tests without paid provider calls"
	@echo "test-live     Explicitly run the paid provider smoke test"
	@echo "check         Run lint, formatting, typing, security, and tests"
	@echo "api           Start FastAPI on the configured loopback address"
	@echo "worker        Start the durable ingestion worker"
	@echo "ui            Start Streamlit"
	@echo "qdrant        Start a local Qdrant server on loopback port 6333"
	@echo "compose-up    Start the production-shaped single-host stack"

sync:
	uv sync --all-groups --frozen

lock:
	uv lock

test:
	uv run pytest -q -m "not live" --disable-socket --allow-unix-socket

test-live:
	RAG_RUN_LIVE_TESTS=1 uv run pytest -q -m live tests/live/test_providers.py

coverage:
	uv run pytest -q -m "not live" --disable-socket --allow-unix-socket --cov=personal_rag --cov-report=term-missing

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy src

security:
	uv run bandit -q -r src scripts
	uv run pip-audit

check: lint typecheck security coverage

api:
	uv run uvicorn personal_rag.api.app:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log

worker:
	uv run python -m personal_rag.worker

ui:
	uv run streamlit run src/personal_rag/ui/app.py --server.address 127.0.0.1 --server.port 8501

qdrant:
	docker compose up -d qdrant

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down
