# Security and Privacy

## Protected assets

The system protects document bytes, extracted text, vector embeddings, answers, citations,
provider keys, the local bearer token, job metadata, backups, and internal storage paths. The
default Compose configuration binds FastAPI, Streamlit, and authenticated Qdrant only to host
loopback. Qdrant also stays on the private Compose network and requires a separate API key.

## Authentication and network exposure

- Every `/api/v1/*` route requires a constant-time compared bearer token when authentication is
  enabled.
- Production configuration refuses to start with authentication disabled or without a token.
- CORS is an explicit allowlist and never uses credentials or a wildcard origin.
- Liveness, readiness, version, and Prometheus metrics contain no document content or secrets.
- The bearer token is suitable for loopback or a trusted private network. Internet exposure
  requires a TLS reverse proxy and OIDC or an equivalent identity-aware access layer. Do not
  expose Qdrant directly.

## Secrets

- `.env` and all provider-bearing variants are ignored by Git; `.env.example` contains key names
  and non-secret placeholders only.
- Streamlit reads the API token server-side. Provider keys are consumed only by API/worker
  processes and are never returned by status endpoints.
- Qdrant uses a distinct admin API key in Compose; it is never sent to Streamlit or returned by
  application endpoints.
- Secrets are injected at runtime rather than copied into the container image.
- Logs, metrics, error envelopes, backups manifests, and completion documents must never include
  key values.

## Upload controls

- Allowed extensions are `.pdf`, `.docx`, `.md`, and `.txt`.
- Backend signature checks are authoritative; Streamlit's file filter is only a convenience.
- File bytes are streamed into a UUID path, bounded during the stream, hashed with SHA-256, and
  stored with mode `0600`.
- Request-body middleware rejects an oversized Content-Length and stops over-limit chunked bodies
  before multipart parsing, while the streaming upload path independently enforces the same limit.
- Original names are sanitized display metadata and never joined into a filesystem path.
- PDF page count and total extracted characters are bounded. Encrypted, corrupt, empty,
  image-only, archive, executable, macro-enabled extensions, and unsupported formats are rejected
  with safe error codes. Accepted DOCX content is parsed as data and is never executed.
- Partial staging files are removed on every failure path.

## RAG-specific controls

- Retrieved source text is untrusted data, enclosed in source delimiters, and excluded from the
  instruction hierarchy. This mitigates prompt injection; it does not make model behavior
  formally immune to adversarial document text.
- The answer model receives no tools and cannot mutate files, jobs, or the vector database.
- Citation records are constructed exclusively from retrieved node metadata. Generated markers
  that do not map to retrieved nodes cause the answer to fail closed, but citation presence is not
  a mathematical proof that every sentence is semantically faithful.
- Query size, history depth, retrieval count, context size, provider attempts, and job attempts
  are bounded.
- Retrieval is authorized against SQLite ready-version truth before the vector query, after
  retrieval, and again before citations are returned. Stale vectors cannot make a non-ready
  document visible.
- Low-support questions abstain. The service does not present a plausible unsupported answer as
  grounded.

## Logging and errors

Structured HTTP logs include service, route template, status, duration, and request ID. Worker
events include service, worker ID, job ID, document ID, job kind, stage, and exception type when
relevant. They intentionally exclude document text, snippets, embeddings, questions, answers,
original filenames, bearer tokens, provider keys, and raw exception responses from providers.
API errors return a stable code, safe message, retryability, and request ID; production responses
never expose stack traces or filesystem paths.

## Provider privacy

Embedding text is sent to the configured OpenAI or Voyage API, and retrieved context plus the
question is sent to the configured OpenAI answer model. Users must review the selected provider's
current data-processing, retention, residency, and account policies before indexing sensitive or
regulated material. This project does not claim that an external provider is appropriate for
every privacy or compliance regime.

## Deletion and backups

Primary deletion is not reported complete until vector readback is zero and the retained source
file is unlinked. This is logical deletion, not guaranteed secure erasure from flash media,
filesystem snapshots, or provider systems. Document and job lifecycle metadata remains in SQLite
for operational history after content deletion.

Live application volumes contain plaintext source files, SQLite metadata, and vector data unless
the host or volume layer provides encryption at rest. Offline backups are immutable copies:
deleting primary data does not erase an older backup. Operators must choose and enforce a backup
retention policy, protect live volumes and archives with filesystem permissions and host/volume
encryption appropriate to the threat model, and destroy expired backups.

Backup creation refuses any output path inside the application data directory, preventing the
archive's temporary tree from recursively entering the upload or vector copy source.

## Security validation

The deterministic quality gate includes Ruff security rules, Bandit, dependency audit, hostile
filename/content tests, request-limit and auth tests, prompt-injection fixtures, sanitized error
tests, deletion readback, and network-disabled default tests. Live-provider tests are opt-in and
must not print provider responses containing user content.

For multi-host Qdrant, add TLS or use Qdrant Cloud/private-cloud controls. The included plaintext
Qdrant connection is limited to the private single-host Compose bridge and is not a public-network
transport design.
