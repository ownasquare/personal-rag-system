# Operations

## Prerequisites

- Python 3.12
- `uv` 0.8.17 or a compatible release that honors the lockfile
- Docker Compose for the production-shaped four-service stack
- An OpenAI API key for answer generation
- Either the same OpenAI key for embeddings or a Voyage key for `voyage-3-large`

The checked-in lockfile is the deployment dependency contract. Default tests use deterministic
providers and do not spend provider credits.

## Configure

```bash
cp .env.example .env
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

Generate two different values and paste them into `RAG_API_KEY` and `RAG_QDRANT_API_KEY`,
configure provider keys, and keep `.env` private.
For Voyage embeddings, set provider/model/dimensions to `voyage`, `voyage-3-large`, and `1024`.
OpenAI still supplies the answer model.

RAG_UPLOAD_MAX_BYTES may be lowered but cannot exceed the Streamlit and API hard ceiling of
25 MiB. RAG_MAX_QUERY_CHARACTERS may be lowered but cannot exceed the request-schema ceiling of
4,000 characters. Startup rejects values above either shared ceiling.

Changing provider, embedding model, dimensions, chunk size, overlap, or parser version changes
the embedding fingerprint. Existing vectors must be rebuilt into a compatible collection; the
system does not mix profiles.

### Embedding-profile migration

Use a new Qdrant collection for every incompatible profile:

1. Stop writers and create a tested offline backup.
2. Choose a new, unused RAG_QDRANT_COLLECTION value and update the provider, model, dimensions,
   parser, and chunk settings together. Never point new dimensions at the old collection.
3. Start Qdrant, API, worker, and UI. Readiness is expected to remain degraded while the new
   collection is missing vectors for still-ready manifest rows.
4. For every active document, call POST /api/v1/documents/{document_id}/reindex and poll the
   returned job through GET /api/v1/jobs/{job_id}. The retained source file makes this operation
   independent of the old vector collection.
5. Wait until every job succeeds, readiness reports matching expected and observed chunk counts,
   and several known questions return the expected citations. Successful reindex atomically
   updates each document's embedding fingerprint for correct future deduplication.
6. Keep the old collection until backup and retrieval verification are complete. Retire it only
   through an authenticated Qdrant administrative workflow with an explicit collection name.

There is intentionally no in-place mixed-profile mode. If any document fails, leave the old
collection intact, correct the failure, and retry that document before retiring old vectors.

## Install and validate

```bash
uv sync --all-groups --frozen
make check
```

`make check` runs formatting and lint checks, strict typing, Bandit, dependency audit, and the
network-disabled deterministic coverage suite. `make test-live` is separate, opt-in, and makes
one paid embedding request plus one paid answer request.

## Local development

Run each long-lived process in its own terminal:

```bash
make qdrant
make api
make worker
make ui
```

Open `http://127.0.0.1:8501`. FastAPI documentation is available at
`http://127.0.0.1:8000/docs` in development only.

### Conversations and activity

Personal Library stores conversations, questions, grounded answers, and their normalized
citations in the same SQLite database as document and job metadata. Conversation writes are
idempotent by client turn ID: reloading or retrying an already completed question returns the
persisted result without repeating a provider call. An interrupted pending reservation can be
recovered after its lease expires, and retryable failures can be submitted again with the same
client turn ID. Live requests renew an internal reservation lease, and ownership fencing prevents
a late provider result from overwriting a newer recovery attempt.

The Activity view is durable rather than browser-session state. It reads the bounded recent-jobs
endpoint, so an upload, refresh, or removal remains visible after a page reload. The worker is
still authoritative for completion; leaving the Activity view does not cancel queued work. Use
the visible Refresh action for current progress. The UI intentionally stops automatic background
requests rather than polling forever after work becomes terminal.

Schema v2 is a forward-only startup migration. It adds conversation, turn, and citation tables
without rewriting schema-v1 document or job rows. Before upgrading an existing installation,
stop writers and create the coordinated offline backup described below. A v2 database cannot be
downgraded in place; rollback restores the pre-upgrade SQLite database and matching vector/source
data together.

## Single-host Compose deployment

```bash
docker compose config
docker compose up --build -d
docker compose ps
```

The API is bound to `127.0.0.1:8000`, Streamlit to `127.0.0.1:8501`, and Qdrant to
`127.0.0.1:6333`. Qdrant is API-key protected and also available on the private Compose network.
Application and Qdrant state live in separate named volumes. Containers drop Linux capabilities,
use `no-new-privileges`, and mount only explicit writable data/tmp locations.

RAG_QDRANT_HOST_PORT changes only the optional host-loopback mapping. API and worker services
always connect to Qdrant's internal port 6333.

If remote access is required, add a reviewed TLS reverse proxy and identity-aware authentication.
Do not change the Compose bindings to `0.0.0.0` and call the bearer token an internet security
boundary.

## Health and status

```bash
curl --fail http://127.0.0.1:8000/health/live
curl --fail http://127.0.0.1:8000/health/ready
curl --fail --header "Authorization: Bearer $RAG_API_KEY" http://127.0.0.1:8000/api/v1/status
```

- Liveness proves only that the process can respond.
- Readiness checks metadata, Qdrant, provider configuration, and worker freshness without a paid
  call. It also reconciles the exact expected and observed vector inventory.
- Protected status explains each degraded dependency without returning secrets or storage paths.

Prometheus metrics are at `/metrics`. Place metrics behind the same private network boundary as
the API.

## Backup

Backups must be offline because a live Qdrant directory is not a coordinated snapshot.

1. Stop API, worker, UI, and Qdrant.
2. For local persistent mode, run:

```bash
uv run python scripts/backup.py \
  --data-dir .data \
  --output backups/personal-rag-2026-07-17.tar.gz \
  --acknowledge-services-stopped
```

3. Store the mode-`0600` archive in encrypted storage with a defined retention period.
4. Restart services and confirm readiness.

The application-data archive includes the SQLite conversation history. Treat it as document
content: answers and citation snippets can reveal source material even when the original upload
is not opened. A restore test should therefore verify both a known citation and a saved
conversation, then delete a fixture document and confirm its cited turns no longer appear.

Compose named volumes require an infrastructure-level offline snapshot of both `app_data` and
`qdrant_data` in the same stopped window. The local script cannot reach a separate Qdrant volume
and does not claim to back it up.

## Restore test

Restore into a new empty directory first:

```bash
uv run python scripts/restore.py \
  --archive backups/personal-rag-2026-07-17.tar.gz \
  --target-dir .restore-test \
  --confirm RESTORE
```

The restore rejects path traversal, links, device members, duplicate members, excessive members,
oversized expansion, unknown files, nonempty targets, malformed or unsafe manifest paths, and hash
mismatches. It requires the archive inventory to match the recorded manifest exactly. Point an
isolated test stack at the restored directories, verify document and chunk counts, retrieve a
known citation, and verify one deletion before replacing active data.

Restored manifests use portable upload keys. The restore tool also normalizes legacy absolute
rows so reindex and deletion resolve only inside the new target's uploads directory; the original
tree is never treated as an authorized path.

## Upgrade

1. Create and test an offline backup.
2. Review Qdrant migration notes and keep the client/integration/server compatibility matrix.
3. Update compatible ranges, run `uv lock`, and inspect the lock diff.
4. Run `make check`, build the image, and validate an isolated restored dataset.
5. Stop the stack, deploy the image, start Qdrant before worker/API/UI, and confirm readiness.
6. Roll back image and restored data together if schema/profile validation fails.

## Troubleshooting

- **Needs setup:** configure the embedding key and OpenAI answer key; readiness never tests them by
  making paid calls.
- **Worker unavailable:** start `make worker`; queued documents remain durable and recover after
  lease expiry.
- **Qdrant unavailable:** confirm the Qdrant version/host/port, API key, and private service health.
- **Embedding profile mismatch:** restore the previous configuration or perform a deliberate full
  reindex. Do not point a new dimension at an existing collection.
- **Document failed:** inspect its safe error code in Library; encrypted, image-only, corrupt,
  unsupported, empty, and oversized files require different corrective action.
- **Question appears pending:** wait for the active reservation lease before retrying. The same
  client turn ID prevents a duplicate paid call; a retryable failed turn can safely be retried.
- **Activity is empty after an action:** refresh the page once and verify the API/worker are using
  the same SQLite data directory. Recent activity comes from persisted jobs, not browser memory.
- **Deletion failed:** keep the document blocked from retrieval, restore Qdrant/file access, and
  retry. Never manually mark it deleted without zero readback.
