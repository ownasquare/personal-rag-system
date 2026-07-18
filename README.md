# Personal Library

Personal Library is a private, production-grade personal RAG system with a deliberately ordinary
document-workspace interface. Add PDFs, Word documents, Markdown, or text; let a durable worker
index them with LlamaIndex and Qdrant; then keep saved conversations and inspect the exact passages
behind each answer.

This release is deliberately engineered for one user on one host. It has durable jobs, strict
upload bounds, content-hash deduplication, crash recovery, backend-built citations, durable
conversation turns, refresh-safe activity, verified deletion, authentication, readiness/metrics,
deterministic no-cost tests, locked dependencies, backup/restore tools, and a production-shaped
Compose topology.

## What it includes

- FastAPI as the only public system-of-record API
- Calm Streamlit Ask, Documents, Activity, and secondary System views
- OpenAI `text-embedding-3-large` or Voyage `voyage-3-large` embeddings
- OpenAI answer-model generation, configured independently from embeddings
- Persistent Qdrant vector storage with immutable embedding-profile checks
- SQLite WAL manifest with durable ingest/reindex/delete jobs and saved cited conversations
- Separate worker with atomic leases, heartbeat, bounded retries, and crash reclamation
- Server-side literal filename/type search, OR status filters, deterministic sorting, and
  paginated single-document management
- PDF, DOCX, Markdown, and text parsing with page/section citation metadata
- SHA-256 deduplication and optional request idempotency keys
- Source-grounded answers, low-support abstention, and citation-marker validation
- Constant-time bearer authentication, strict CORS, safe errors, security headers, and
  content-free structured logs/metrics
- Non-root, capability-dropped containers and a private, API-key-protected Qdrant service

## Architecture

```text
Browser -> Streamlit -> FastAPI -> SQLite manifest, conversations + private upload storage
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

1. Open **Ask**. If the library is empty, the first-run surface explains the three-step flow and
   accepts one or more supported files.
2. Follow durable progress in **Activity**. Jobs are restored from the API after a refresh or a
   closed browser; newly accepted work appears there immediately. Choose **Refresh** when you want
   the latest state—terminal activity does not create a permanent background polling loop.
3. Ask from the question-first composer, optionally narrow **Where to look**, then open **Sources**
   to read the filename, page or section, and exact supporting passage. Suggestions follow the
   primary action, while retrieval scoring stays out of the normal reading experience.
4. Reopen or remove saved conversations from the secondary **Saved conversations** disclosure.
   Completed, pending, and failed turn truth is stored by the API. Saved pending and retryable
   turns expose a one-click retry, while an edited question receives a new client turn ID so
   idempotency remains correct.
5. Use **Documents** to search filenames and file types on the server, combine user-facing status
   groups, choose a deterministic sort, and move through ten-item pages. Select one document to
   inspect or act on it; adding more files stays secondary when the library is non-empty. A
   successful source deletion also removes saved turns that cite it.
6. Open **System** only for sanitized setup and dependency details. Keys and storage paths are
   never displayed.

## API

All `/api/v1/*` routes use `Authorization: Bearer <RAG_API_KEY>` when auth is enabled.

| Capability | Route |
|---|---|
| Liveness / readiness | `GET /health/live`, `GET /health/ready` |
| Sanitized status | `GET /api/v1/status` |
| Upload / searchable, filterable, sortable list | `POST`, `GET /api/v1/documents` |
| Document detail | `GET /api/v1/documents/{id}` |
| Reindex | `POST /api/v1/documents/{id}/reindex` |
| Verified deletion | `DELETE /api/v1/documents/{id}` |
| Recent activity / job readback | `GET /api/v1/jobs`, `GET /api/v1/jobs/{id}` |
| Durable conversations | `POST`, `GET`, `DELETE /api/v1/conversations...` |
| Stateless grounded chat compatibility | `POST /api/v1/chat` |
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
- [Phase 2 completion record](docs/personal-rag/2026-07-17-product-experience-phase-2.md)
- [Library findability Phase 2 completion record](docs/personal-rag/2026-07-18-library-findability-phase-2.md)
- [Implementation plan](docs/superpowers/plans/2026-07-17-personal-rag-system.md)
- [Phase 2 product-experience plan](docs/superpowers/plans/2026-07-17-personal-rag-product-experience-phase-2.md)
- [Library findability Phase 2 plan](docs/superpowers/plans/2026-07-18-personal-library-findability-phase-2.md)

## Honest limitations

- This release is single-user and single-host, not multi-tenant or highly available.
- Scanned image-only PDFs require an external OCR stage and are rejected instead of silently
  indexing empty content.
- Completed conversation turns are durable. An unsubmitted text-area draft remains browser-session
  state; retryable submitted turns retain their server reservation and safe failure state.
- Library search covers display name and extension metadata only. It does not search document
  bodies, snippets, or vectors; tags, collections, and renaming are deferred.
- The optional Ask document picker still loads at most 2,000 ready-document choices. Library
  browsing itself uses truthful server-side pagination and is not limited by that picker.
- Compose backup requires both application and Qdrant volumes to be snapshotted in one stopped
  window. The local backup script cannot snapshot a separate container volume.
- Live provider behavior, cost, quota, and privacy remain account/provider concerns and are kept
  separate from deterministic local validation.
