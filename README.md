# Personal Knowledge Studio

Personal Knowledge Studio is a private, production-grade personal RAG system. Upload PDFs, Word
documents, Markdown, or text; let a durable worker index them with LlamaIndex and Qdrant; then ask
questions in Streamlit and inspect the exact sources behind each answer.

This release is deliberately engineered for one user on one host. It has durable jobs, strict
upload bounds, content-hash deduplication, crash recovery, backend-built citations, verified
deletion, authentication, readiness/metrics, deterministic no-cost tests, locked dependencies,
backup/restore tools, and a production-shaped Compose topology.

## What it includes

- FastAPI as the only public system-of-record API
- Streamlit Chat, Library, and Settings & Status areas
- OpenAI `text-embedding-3-large` or Voyage `voyage-3-large` embeddings
- OpenAI answer-model generation, configured independently from embeddings
- Persistent Qdrant vector storage with immutable embedding-profile checks
- SQLite WAL manifest and durable ingest/reindex/delete jobs
- Separate worker with atomic leases, heartbeat, bounded retries, and crash reclamation
- PDF, DOCX, Markdown, and text parsing with page/section citation metadata
- SHA-256 deduplication and optional request idempotency keys
- Source-grounded answers, low-support abstention, and citation-marker validation
- Constant-time bearer authentication, strict CORS, safe errors, security headers, and
  content-free structured logs/metrics
- Non-root, capability-dropped containers and a private, API-key-protected Qdrant service

## Architecture

```text
Browser -> Streamlit -> FastAPI -> SQLite manifest + private upload storage
                         |   \
                         |    -> LlamaIndex retrieval -> Qdrant -> OpenAI answer model
                         |
                         -> durable jobs <- worker -> parser -> embedding provider -> Qdrant
```

FastAPI never performs long document ingestion in an in-process background task. The worker
leases persisted jobs, so a process restart does not silently lose accepted work. Qdrant is a
separate service in Compose; embedded persistence is restricted to local development and tests.

Read the full [architecture](docs/architecture.md) and [security model](docs/security.md).

## Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) 0.8.17 or compatible
- Docker Compose for the production-shaped deployment
- An OpenAI API key for answers
- Either OpenAI or Voyage credentials for embeddings

## Quick start with Docker Compose

```bash
cp .env.example .env
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

Generate two different values: put one in `RAG_API_KEY` and one in `RAG_QDRANT_API_KEY`. Add
provider keys to the non-committed `.env`, then:

```bash
docker compose config
docker compose up --build -d
docker compose ps
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501). The API is on loopback at
[http://127.0.0.1:8000](http://127.0.0.1:8000); authenticated Qdrant is bound only to host
loopback on port `6333`.

## Local development

Install the exact lock and start one process per terminal:

```bash
uv sync --all-groups --frozen
make qdrant
make api
make worker
make ui
```

The `api`, `worker`, and `ui` commands are long-running and should use separate terminals;
`qdrant` starts the detached Compose service and returns.
Development OpenAPI documentation is at `http://127.0.0.1:8000/docs`.

## Choose an embedding provider

The default `.env.example` profile is:

```dotenv
RAG_EMBEDDING_PROVIDER=openai
RAG_EMBEDDING_MODEL=text-embedding-3-large
RAG_EMBEDDING_DIMENSIONS=3072
```

Voyage configuration is:

```dotenv
RAG_EMBEDDING_PROVIDER=voyage
RAG_EMBEDDING_MODEL=voyage-3-large
RAG_EMBEDDING_DIMENSIONS=1024
```

Voyage supplies embeddings only. `RAG_OPENAI_API_KEY` is still required for the answer model.
The locked Voyage SDK is intentionally `0.3.7`, the compatible version required by the current
LlamaIndex Voyage integration.

Provider, embedding model, dimensions, parser version, chunk size, and overlap form an immutable
profile. Changing any of them requires a new collection plus a deliberate document-by-document
reindex; vectors with different dimensions are never mixed in one collection. Follow the exact
[embedding-profile migration workflow](docs/operations.md#embedding-profile-migration).

## Use the app

1. Open **Library**, select one or more supported files, and click **Add to library**.
2. Watch durable job state progress from queued through ready. A refresh re-reads state from the
   API rather than losing it.
3. Open **Chat**, ask a question, and expand citation cards for filename, page/section, snippet,
   stable chunk ID, and raw retrieval score.
4. Reindex or delete from **Library**. Deletion is not reported complete until Qdrant readback is
   zero and the retained source file is removed.
5. Use **Settings & Status** for sanitized models, counts, and dependency health. Keys and storage
   paths are never displayed.

## API

All `/api/v1/*` routes use `Authorization: Bearer <RAG_API_KEY>` when auth is enabled.

| Capability | Route |
|---|---|
| Liveness / readiness | `GET /health/live`, `GET /health/ready` |
| Sanitized status | `GET /api/v1/status` |
| Upload / list | `POST`, `GET /api/v1/documents` |
| Document detail | `GET /api/v1/documents/{id}` |
| Reindex | `POST /api/v1/documents/{id}/reindex` |
| Verified deletion | `DELETE /api/v1/documents/{id}` |
| Durable job readback | `GET /api/v1/jobs/{id}` |
| Grounded chat | `POST /api/v1/chat` |
| Prometheus metrics | `GET /metrics` |

See the complete [API contract](docs/api.md).

## Quality gates

Default tests never make paid provider calls. They use deterministic embeddings/answers and
temporary SQLite/Qdrant stores.

```bash
make test
make coverage
make lint
make typecheck
make security
make check
```

`make check` covers Ruff lint/format, strict mypy, Bandit, dependency audit, and branch coverage.
Live-provider smoke tests are separate, explicit, and non-gating.

Run `make test-live` only after setting `RAG_RUN_LIVE_TESTS=1` and the required provider
credentials. It makes one paid embedding request and one paid answer request.

## Privacy boundary

Provider-backed RAG sends chunks to the selected embedding provider and sends the question plus
retrieved context to OpenAI for answer generation. Review current provider retention, residency,
and processing policies before indexing regulated or highly sensitive material. Logs and metrics
exclude questions, answers, document text, snippets, embeddings, provider responses, and secret
values.

The bearer token is a loopback/private-network control. Internet exposure requires TLS and an
identity-aware access layer such as OIDC. Do not publish Qdrant.

## Operations

- [Operations, health, upgrade, backup, and restore](docs/operations.md)
- [Security and privacy](docs/security.md)
- [Architecture and scaling boundary](docs/architecture.md)
- [HTTP API](docs/api.md)
- [Validation evidence and proof boundaries](docs/validation.md)
- [Implementation completion record](docs/personal-rag/2026-07-17-production-rag-implementation.md)
- [Implementation plan](docs/superpowers/plans/2026-07-17-personal-rag-system.md)

## Honest limitations

- This release is single-user and single-host, not multi-tenant or highly available.
- Scanned image-only PDFs require an external OCR stage and are rejected instead of silently
  indexing empty content.
- Streamlit chat history is presentation-session state; documents and jobs are durable.
- Compose backup requires both application and Qdrant volumes to be snapshotted in one stopped
  window. The local backup script cannot snapshot a separate container volume.
- Live provider behavior, cost, quota, and privacy remain account/provider concerns and are kept
  separate from deterministic local validation.
