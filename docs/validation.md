# Validation Evidence

## 2026-07-18 library-findability Phase 2 release result

The complete local working tree passed the deterministic release gate using generated fixtures,
temporary SQLite state, local vector stores, and a stateful no-network FastAPI browser fixture. No
personal document was used and no paid model-provider request was made.

### Current automated gates

| Gate | Result |
|---|---|
| Dependency lock | Passed; `uv lock --check` resolved the locked 173-package graph |
| Python package build | Passed; source distribution and wheel built from the locked project |
| Ruff format | Passed after formatting the two changed browser-fixture files; 54 files are in canonical format |
| Ruff lint | Passed; all checks passed |
| Strict mypy | Passed; no issues in 30 source files |
| Bandit | Passed; no findings in `src` or `scripts` |
| Dependency audit | Passed; no known vulnerabilities; only the local non-PyPI project package was skipped as expected |
| Pytest | Passed; 200 deterministic tests; one opt-in live-provider test deselected |
| Branch coverage | Passed; 82.46 percent against the 80 percent gate |
| Compose configuration | Passed with the example environment contract |
| Container image | Passed; the application image rebuilt successfully from the current source |
| Repository hygiene | Passed; `git diff --check` reported no whitespace errors |

One visible test warning remains: FastAPI's compatibility `TestClient` emits an upstream Starlette
deprecation warning about the httpx adapter. It does not affect runtime code and is not suppressed.

### Current regression coverage

- Repository tests seed 30 documents and prove literal Unicode matching, literal percent and
  underscore handling, extension matching, OR status groups, deleted exclusion, exact filtered
  counts, and complete stable page boundaries for all six sort/order pairs.
- API tests prove default and single-status compatibility, repeatable statuses, filtered totals,
  sanitized invalid query/enum handling, and rejection of leading controls, format characters,
  private-use characters, and overlong input.
- Client tests assert exact encoding of repeated status parameters and fixed sort/order values
  while preserving the legacy default request shape.
- Streamlit AppTests prove one-page requests, one selected action panel, one-request filter submit,
  page/filter reset, concurrent empty-boundary-page recovery, typed deletion, hostile-name
  escaping, and non-empty-library upload de-emphasis.
- Ask AppTests prove composer-before-suggestions ordering, widget-safe suggestion seeding through a
  new draft key, secondary saved-conversation management, and durable retry/delete behavior.
- The browser fixture contains 16 documents and mirrors literal search, repeated statuses, Unicode
  normalization, fixed sort, exact totals, paging, and separate-field matching.

### Current deterministic browser proof

The Streamlit interface was exercised in the in-app browser against the stateful local fixture.

| Viewport | Observed result |
|---|---|
| Desktop, 1440 by 1000 | Library filters and first-page selection were visible together; the two-column master/detail remained calm and readable |
| Tablet, 768 by 1024 | Filters, selection list, and selected detail remained readable with measured document width exactly 768 pixels |
| Phone, 390 by 844 | Search controls and a complete first document row fit in the first viewport; Ask showed the question field and primary action in the first viewport; measured width stayed exactly 390 pixels |

The rendered flow applied the literal `Résumé 100%_` query with **Needs attention** and **Name
A-Z**, verified the exact one-result total, moved to page two and selected another document, opened
the secondary upload control, seeded a suggested question without widget-state failure, and opened
saved-conversation controls. The document heading was verified after eliminating a stale generated
anchor. Semantic H1/H2/H3 structure and visible text status labels remained present, horizontal
overflow was absent at all three widths, and a fresh final tab reported an empty browser log. The
temporary browser, API, and UI processes were stopped. The fixed light theme remains intentional;
dark-mode proof is not claimed.

### Current isolated Compose runtime proof

The rebuilt image was started as a disposable four-service Compose project on isolated loopback
ports with proof-only credentials and a nonfunctional placeholder provider key.

- API, worker, Streamlit, and pinned `qdrant/qdrant:v1.18.3-unprivileged` containers all ran.
- API and UI health checks reported healthy.
- `GET /health/ready` reported metadata, Qdrant, vector inventory, provider configuration, and
  worker heartbeat ready without making a paid call.
- Streamlit `/_stcore/health` returned `ok`.
- An authenticated document request with literal query, repeated statuses, name sort, order,
  limit, and offset returned the correct empty `DocumentList` contract.
- The exact proof containers, network, volumes, and ignored environment file were removed.

This proves the current local container topology and readback boundary. It does not prove a real
OpenAI/Voyage request, provider quota or cost, hosted development, production deployment, or
multi-user/high-availability behavior.

## Historical evidence — 2026-07-17 product-experience Phase 2

The 2026-07-17 Phase 2 local release gate passed for the complete working tree. Tests used
generated fixtures, deterministic embedding and answer providers, temporary SQLite state, and
local vector stores. Browser proof used a stateful local FastAPI fixture. No personal document was
used and no paid model-provider request was made.

### Historical automated gates

| Gate | Result |
|---|---|
| Dependency lock | Passed; `uv lock --check` resolved the locked 173-package graph |
| Python package build | Passed; source distribution and wheel built from the locked project |
| Ruff format | Passed; 54 files already formatted |
| Ruff lint | Passed; all checks passed |
| Strict mypy | Passed; no issues in 30 source files |
| Bandit | Passed; no findings in `src` or `scripts` |
| Dependency audit | Passed; no known vulnerabilities; only the local non-PyPI project package was skipped as expected |
| Pytest | Passed; 160 deterministic tests; the opt-in live-provider test was deselected |
| Branch coverage | Passed; 81.85 percent against the 80 percent gate |
| Compose configuration | Passed with the example environment contract |
| Container image | Passed; the final Phase 2 application image built successfully |
| Repository hygiene | Passed; `git diff --check` reported no whitespace errors |

One warning remains in the test environment: FastAPI's TestClient compatibility module emits an
upstream Starlette deprecation warning about the httpx adapter. It does not affect runtime code
and remains visible rather than being suppressed.

### Historical Phase 2 regression coverage

The release suite includes focused proof for the new product and safety contracts:

- SQLite schema v1 migrates forward to v2 without changing existing document or job records.
- Conversations, turns, and normalized citations survive repository restart and paginate without
  splitting user/assistant history pairs.
- A client turn ID is idempotent: a completed duplicate returns persisted truth, an active
  duplicate is rejected, and edited retry input receives a new identifier.
- Turn reservations use renewable ownership tokens. A stale provider attempt cannot complete or
  fail a turn after a newer attempt has recovered the reservation.
- Completion revalidates cited source state inside the write transaction. Deleted or non-ready
  sources produce a retryable `source_changed` failure instead of retained stale citations.
- Source deletion purges cited turns and removes or retitles affected empty conversations in the
  same metadata transaction.
- Persisted retryable and expired-pending turns can be retried after a browser refresh.
- Recent durable jobs are paginated and presented as In progress, Needs attention, or Completed.
- The deterministic browser fixture advances queued work through running to succeeded, mirrors
  real active-job conflicts, and purges fixture citation history after completed deletion.
- Hostile document names remain text in Documents and Activity instead of being interpreted as
  trusted HTML or Markdown.

### Historical deterministic browser proof

The Streamlit interface was exercised in the in-app browser against a temporary local FastAPI
fixture containing sanitized library, job, conversation, and citation data.

| Viewport | Observed result |
|---|---|
| Desktop, 1440 by 1000 | Ask-first workspace, saved conversation controls, scoped composer, suggestions, and source-backed history rendered without horizontal overflow |
| Tablet, 768 by 1024 | Ask, Documents, Activity, and System navigation remained reachable and the cited workflow stayed readable |
| Phone, 390 by 844 | Header compacted to a computed 20.48-pixel title, subtitle collapsed, navigation remained usable, and measured document width stayed exactly 390 pixels |

The rendered flow also exercised saved conversation reload, cited answer display, failed-document
reindex handoff, deterministic job progression, and the explicit Activity refresh control. The
Activity view separated active, failed, and completed work. The semantic heading sequence was
H1, H2, then H3; no framework exception was present. A new final proof tab reported an empty
browser log on desktop and mobile. Temporary browser/API/UI proof processes were stopped.

The checked-in Streamlit theme is intentionally fixed light. Dark-mode proof is not claimed.

### Historical isolated Compose runtime proof

An isolated Compose project was started on disposable loopback ports with generated proof-only
credentials and a nonfunctional placeholder model key. The final application image and pinned
`qdrant/qdrant:v1.18.3-unprivileged` service were used.

- API, worker, UI, and Qdrant containers all started.
- API and UI containers reported healthy.
- `GET /health/live` returned `alive`.
- `GET /health/ready` reported metadata, Qdrant, vector inventory, provider configuration, and
  worker heartbeat ready.
- Authenticated `GET /api/v1/status` returned an empty ready library and a current worker
  heartbeat.
- Streamlit `/_stcore/health` returned `ok`.

This proves local container topology, configuration validation, authentication, storage
connectivity, worker heartbeat, and health readback. Provider readiness here means configuration
is present; it does not prove that a real OpenAI or Voyage request succeeds. No document was
uploaded and no model endpoint was contacted. After proof, the exact containers, network,
volumes, and temporary environment file were removed, and empty post-teardown inventories were
verified.

### Historical proof boundaries

- Local source, deterministic test, deterministic browser, final image build, and isolated local
  Compose runtime proof: complete.
- Live OpenAI or Voyage semantics, quota, latency, privacy, and cost proof: not run. No usable
  credential was supplied and no provider credit was spent.
- Hosted-development proof: not run.
- Production deployment proof: not run.
- Multi-user, multi-host, and high-availability proof: outside the documented single-user product
  boundary.

## Reproduce the deterministic gate

```bash
uv sync --all-groups --frozen
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run bandit -q -r src scripts
uv run pip-audit
uv run pytest -m "not live" --disable-socket --allow-unix-socket \
  --cov=personal_rag --cov-report=term-missing --cov-branch -q
uv build
docker compose --env-file .env.example config --quiet
docker compose --env-file .env.example build
git diff --check
```

For runtime smoke proof, configure a private non-committed environment file, start Compose, read
back API/UI health and authenticated status, then stop the project with volumes removed. Use
disposable data for the first deployment smoke test.
