# Extending Personal Library

Personal Library favors explicit, typed extension points over a dynamic plugin system. That makes
security boundaries and migrations reviewable, but it means an extension usually touches a small,
documented set of files.

Start with `uv sync --all-groups --frozen`. Keep default tests deterministic and inject fakes at
the existing protocol/factory boundaries.

## Add a document type or parser

1. Add one `DocumentTypeSpec` in `src/personal_rag/document_types.py`. Keep extensions lowercase,
   include the leading dot, and use one canonical stored content type.
2. Add the parser dispatch in `src/personal_rag/parsers.py`. Preserve path, byte, page, character,
   decompression, encoding, and signature limits.
3. Return LlamaIndex `Document` objects whose scalar metadata includes `source_name`,
   `source_extension`, `parser_version`, `unit_index`, and an applicable `page_number` or
   `section`.
4. Add fixtures and tests in `tests/unit/test_parsers.py` for valid input, extension spoofing,
   malformed/encrypted input, empty content, and the relevant resource ceiling.
5. Update `tests/unit/test_document_types.py`, `docs/security.md`, and the supported-file copy.

Run:

```bash
uv run pytest -q tests/unit/test_document_types.py tests/unit/test_parsers.py tests/api/test_api.py
```

The API and UI derive their public allowlists from the registry. Do not create a second extension
list.

## Add an embedding provider

1. Extend `Settings.embedding_provider` and provider-specific secret/config fields in
   `src/personal_rag/config.py`.
2. Add validation for model naming, dimensions, missing credentials, and sanitized setup status.
3. Add a branch to `build_embedding()` in `src/personal_rag/providers.py`. Pass credentials,
   timeout, and retry limits explicitly; do not mutate process-global LlamaIndex settings.
4. Wire the same settings through `docker-compose.yml` and `.env.advanced.example`.
5. Add a deterministic fake for ordinary tests and one opt-in paid smoke case under `tests/live/`.
6. Document provider data flow and the immutable embedding-profile consequence.

Changing provider, model, dimensions, parser version, chunk size, or overlap changes the embedding
fingerprint. Existing vectors must be reindexed into a new compatible collection; never mix vector
dimensions in one collection.

## Add an answer provider

1. Extend `Settings.chat_provider` and its explicit credential validation.
2. Implement the LlamaIndex `LLM` factory branch in `build_llm()`.
3. Preserve the RAG service contract: source text is untrusted data, the model receives no tools,
   citations are backend-built, unsupported citation markers fail closed, and low-support answers
   abstain.
4. Add deterministic completion fakes and explicit opt-in live coverage.
5. Update the privacy boundary in `README.md` and `docs/security.md`.

## Add an API capability

1. Define or extend Pydantic contracts in `src/personal_rag/models.py`.
2. Add a route under `src/personal_rag/api/routes/` and register it in
   `src/personal_rag/api/app.py`.
3. Put persistence in `repository.py`, long-running work in durable jobs, and provider/vector work
   behind the container factories. Do not use in-process FastAPI background tasks for durable work.
4. Return the standard safe error envelope from `errors.py`; never return provider exceptions.
5. Add API tests in `tests/api/test_api.py` and persistence/integration tests as needed.
6. Update `docs/api.md`.

## Build another frontend

The Streamlit app is intentionally a thin server-side client. Another frontend should use only the
versioned `/api/v1` routes documented in [api.md](api.md):

- send `Authorization: Bearer <RAG_API_KEY>` from a trusted server boundary;
- upload through `POST /api/v1/documents` and poll the returned durable job;
- list only ready documents when scoping a question;
- create conversations and idempotent turns through the conversation routes; and
- render citation snippets as untrusted text, not HTML.

Do not put provider keys or the application bearer token in public browser code. If a browser-only
client is required, add a real identity/session backend first.

## Add a Streamlit view

1. Add client behavior in `src/personal_rag/ui/client.py` and its fake in
   `tests/ui/conftest.py`.
2. Keep authoritative state in the API. Session state may own only draft/presentation choices.
3. Add the smallest primary navigation entry; diagnostics and rare maintenance belong in secondary
   disclosures.
4. Add AppTest coverage in `tests/ui/test_app.py`.
5. Follow the [manual rendered interface check](../CONTRIBUTING.md#validate)
   against `tests/browser/fake_api.py`, including desktop, tablet, phone, and one real interaction.

## Add a database migration

Append a new numbered migration in `src/personal_rag/database.py`; never edit a migration already
released. Add a frozen prior-schema fixture and prove forward upgrade, repeat startup, rollback by
restoring a backup, and the API behavior that depends on the new schema.
