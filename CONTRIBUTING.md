# Contributing

Thanks for helping improve Personal Library. The project is intentionally focused: one person,
one private host, and a dependable path from documents to answers with inspectable sources.

## Start without provider credentials

```bash
uv sync --all-groups --frozen
uv run python scripts/demo.py
```

The demo uses deterministic sample data and makes no provider calls. Open
`http://127.0.0.1:8512`, then stop it with `Ctrl+C`.

## Development setup

Install Python 3.12, `uv`, Docker, and Docker Compose v2, then:

```bash
python3 scripts/setup.py
uv sync --all-groups --frozen
make qdrant
make api
make worker
make ui
```

The setup assistant creates local service tokens and collects the OpenAI key required by the real
development stack. The last three commands are long-running and use separate terminals. Default
tests are offline and deterministic; they do not need an OpenAI or Voyage key.

## Repository map

- `src/personal_rag/api/` — versioned FastAPI routes and request boundaries
- `src/personal_rag/ui/` — Streamlit workspace and API client
- `src/personal_rag/document_types.py` — supported-file contract
- `src/personal_rag/parsers.py` — bounded document extraction
- `src/personal_rag/providers.py` — embedding and answer-provider factories
- `src/personal_rag/rag_service.py` — retrieval, grounding, and citation validation
- `src/personal_rag/job_service.py` — durable ingestion/reindex/delete work
- `src/personal_rag/database.py` and `repository.py` — SQLite schema and persistence
- `tests/` — unit, API, integration, UI, browser-fixture, and opt-in live tests

See [Extending Personal Library](docs/extending.md) before adding a file type, provider, route, or
UI surface.

## Make a change

1. Keep the change small and preserve the single-user/single-host boundary unless the proposal
   explicitly changes it.
2. Add a failing test first. Use Streamlit AppTest for component behavior and the deterministic
   browser fixture for rendered flows.
3. Never rewrite a released SQLite migration. Append a new migration and add an upgrade test from
   the previous schema.
4. Keep provider calls behind explicit factories and keep default tests provider-free.
5. Do not log questions, answers, document content, source snippets, embeddings, or credentials.
6. Update the nearest public documentation and `CHANGELOG.md` when behavior changes.

## Validate

```bash
make check
uv lock --check
uv build
docker compose --env-file .env.example config --quiet
```

`make check` runs formatting, lint, strict typing, security/dependency checks, public-repository
hygiene, and the deterministic coverage suite. Run `make test-live` only when you deliberately
want a paid provider smoke test.

## Pull requests

Explain the user-visible outcome, tests run, security/privacy impact, migration or embedding-profile
impact, and what remains untested. Screenshots are useful for interface changes, but tests and
rendered interaction proof are still required.

By participating, you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
