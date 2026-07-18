# HTTP API

The API prefix is `/api/v1`. Protected requests use:

```http
Authorization: Bearer <RAG_API_KEY>
```

Errors use one stable envelope:

```json
{
  "error": {
    "code": "unsupported_file_type",
    "message": "Use a PDF, DOCX, Markdown, or text file.",
    "retryable": false,
    "request_id": "5f09ecfb43bd45f7810e122c5e1735ab"
  }
}
```

The response header `X-Request-ID` carries the same correlation ID. Request and response bodies
are not logged.

The API rejects request bodies above the configured upload limit before multipart parsing,
including streaming requests without Content-Length. Stateless chat history and reconstructed
conversation history are capped by both schema and runtime configuration; the absolute schema
ceiling is 100 messages. The configurable upload
limit can be lowered from the shared 25 MiB UI/API ceiling, and the query limit can be lowered
from the 4,000-character RAG-question schema ceiling. Document-metadata search has a separate,
fixed 200-character ceiling.

## Unprotected operational endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health/live` | Process liveness only |
| `GET` | `/health/ready` | Metadata, Qdrant, provider configuration, and worker freshness |
| `GET` | `/version` | Package name and version |
| `GET` | `/metrics` | Content-safe Prometheus metrics when enabled |

Readiness intentionally makes no paid provider request. It also compares the SQLite manifest's
expected ready chunk count with Qdrant's exact collection count so partial vector-volume loss is
reported as degraded rather than hidden behind a successful heartbeat.

## Status

`GET /api/v1/status` returns sanitized provider/model names, dimensions, collection name,
document/chunk/job counts, dependency states, and worker heartbeat time. It never returns keys or
filesystem paths.

## Documents

### Upload

`POST /api/v1/documents` accepts one multipart field named `file` and returns `202` only after the
document and durable job are stored. The optional `Idempotency-Key` header must contain 8-128
URL-safe characters.

```json
{
  "document": {
    "id": "2d9da2e2cc5d4dd58e5c0083615a9b26",
    "display_name": "notes.md",
    "content_type": "text/markdown",
    "extension": ".md",
    "size_bytes": 1402,
    "status": "queued",
    "active_version": 0,
    "chunk_count": 0,
    "error_code": null,
    "error_message": null,
    "created_at": "2026-07-17T20:00:00Z",
    "updated_at": "2026-07-17T20:00:00Z"
  },
  "job": {
    "id": "0aa15b1ba03e41e38d92347d531dc758",
    "document_id": "2d9da2e2cc5d4dd58e5c0083615a9b26",
    "kind": "ingest",
    "status": "queued",
    "stage": "queued",
    "progress": 0.0,
    "attempts": 0,
    "max_attempts": 3,
    "lease_owner": null,
    "lease_expires_at": null,
    "error_code": null,
    "error_message": null,
    "created_at": "2026-07-17T20:00:00Z",
    "updated_at": "2026-07-17T20:00:00Z",
    "finished_at": null
  },
  "duplicate": false
}
```

### Library routes

| Method | Path | Result |
|---|---|---|
| `GET` | `/api/v1/documents?limit=50&offset=0&status=ready` | Paginated active documents |
| `GET` | `/api/v1/documents/{document_id}` | One sanitized document record |
| `POST` | `/api/v1/documents/{document_id}/reindex` | `202` durable reindex job |
| `DELETE` | `/api/v1/documents/{document_id}` | `202` durable verified-deletion job |
| `GET` | `/api/v1/jobs?limit=50&offset=0&status=running&document_id=...` | Paginated durable activity |
| `GET` | `/api/v1/jobs/{job_id}` | Durable stage/progress/error readback |

#### Document listing contract

`GET /api/v1/documents` accepts:

- `q`: optional literal search text, at most 200 characters. It is trimmed and matched with a
  deterministic Unicode normalization/casefold against `display_name` and `extension` only.
  Percent and underscore characters have no wildcard meaning. Unsupported control, format, and
  other Unicode control-category characters return a sanitized `invalid_document_query` `422`.
- `status`: optional and repeatable. Multiple values use OR semantics; the existing single
  `status=ready` form remains compatible. Values are validated `DocumentStatus` enums.
- `sort`: `created`, `updated`, or `name`; default `created`.
- `order`: `asc` or `desc`; default `desc`.
- `limit`: 1-100; default 50. `offset` is non-negative; default 0.

For example:

```http
GET /api/v1/documents?q=plan%25_&status=failed&status=deletion_failed&sort=name&order=asc&limit=10&offset=0
```

`total` applies the same query and status predicates before paging. Sort ties use document ID, so
page boundaries are stable while the underlying library is unchanged. List items and totals are
separate reads and may briefly reflect a concurrent add or removal; the UI safely clamps or
refreshes an empty boundary page. Deleted records never appear. Metadata search reads SQLite only:
it does not inspect file bodies, stored paths, snippets, chunks, embeddings, Qdrant, or a model
provider.

Deletion is idempotent. A document in `deleting` state is excluded from ordinary retrieval; it is
not reported `deleted` until Qdrant and retained-file readback succeed.

## Durable conversations

The UI uses server-side conversation truth rather than browser-local chat history.

| Method | Path | Result |
|---|---|---|
| `POST` | `/api/v1/conversations` | Create an empty saved conversation |
| `GET` | `/api/v1/conversations?limit=50&offset=0` | Recency-ordered summaries |
| `GET` | `/api/v1/conversations/{conversation_id}` | One summary and completed-turn count |
| `DELETE` | `/api/v1/conversations/{conversation_id}` | `204`, hard-delete retained turn content |
| `GET` | `/api/v1/conversations/{conversation_id}/turns` | Completed, pending, and failed turn truth |
| `POST` | `/api/v1/conversations/{conversation_id}/turns` | Reserve, answer, and persist one turn |

Create a turn with a stable client-generated ID:

```json
{
  "client_turn_id": "a96ee96a24c24a8290bdc123e704a05f",
  "message": "What was the Atlas launch key color?",
  "top_k": 5,
  "document_ids": null
}
```

The API reserves `(conversation_id, client_turn_id)` before provider work. A completed duplicate
returns the persisted result without a second provider call. An active duplicate returns a
retryable `409`; an expired or retryable failed reservation can be reclaimed only with the same
request fingerprint. Every live provider call renews an internal ownership token; completion and
failure writes are fenced by that token so a stale attempt cannot overwrite a newer recovery.
Conversation history is rebuilt exclusively from complete user/assistant pairs in completed
turns.

Each response is a `ConversationTurn` with `pending`, `completed`, or `failed` status, the original
question and document scope, typed citations, and safe failure metadata. Provider exception text
is never stored. The first completed question gives an untitled conversation a deterministic
72-character title. Immediately before answer/citation commit, SQLite revalidates every cited
document as retrieval-ready in the same write transaction. A source removed or reindexed while a
provider call is in flight therefore produces a retryable `source_changed` response instead of
re-persisting stale content.

When verified source deletion completes, every whole conversation turn citing that document is
deleted in the same SQLite transaction. Empty affected conversations are removed, and non-empty
affected conversations are retitled from their first remaining completed question so a purged
question cannot survive in the title. Offline backups remain separate retained copies.

## Chat

`POST /api/v1/chat` accepts:

```json
{
  "message": "What was the Atlas launch key color?",
  "history": [
    {"role": "user", "content": "Summarize the launch notes."},
    {"role": "assistant", "content": "The notes describe the Atlas launch."}
  ],
  "top_k": 5,
  "document_ids": null
}
```

The response separates answer prose from authoritative citation records:

```json
{
  "answer": "The Atlas launch key was cobalt blue [S1].",
  "citations": [
    {
      "label": "S1",
      "document_id": "2d9da2e2cc5d4dd58e5c0083615a9b26",
      "chunk_id": "d8ad91fd3ff3d0b9fef64083f427450d",
      "document_name": "notes.md",
      "page_number": null,
      "section": "Launch checklist",
      "snippet": "The Atlas launch key is cobalt blue.",
      "score": 0.83
    }
  ],
  "no_answer": false,
  "request_id": "5f09ecfb43bd45f7810e122c5e1735ab"
}
```

`score` is retrieval relevance, not a calibrated probability. When sources do not support an
answer, `no_answer` is true and citations are empty.

`POST /api/v1/chat` remains a stateless compatibility endpoint. New first-party UI work should use
conversation turns so idempotency, history, failures, citations, and refresh recovery are durable.
