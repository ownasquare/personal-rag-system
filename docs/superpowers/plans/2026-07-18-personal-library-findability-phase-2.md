# Personal Library Findability Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a growing Personal Library feel like a familiar files workspace: searchable and
pageable on the server, easy to scan one document at a time, and fast to ask from on a phone.

**Architecture:** Keep SQLite document metadata as the only source used by library search. Extend
the existing paginated document endpoint with validated literal query, status, and fixed sort
contracts; do not touch embeddings, Qdrant, worker jobs, or conversation persistence. Rebuild the
Streamlit Documents surface as a compact library-first master/detail view and reorder Ask so the
composer precedes suggestions and secondary saved-conversation controls.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, SQLite/WAL, httpx, Streamlit 1.59, pytest,
Streamlit AppTest, deterministic FastAPI browser fixture, Ruff, mypy, Bandit, pip-audit, Docker
Compose, in-app Browser.

---

## Product decisions

- This is the user-requested new Phase 2, built on clean local `main` at `4755d19`.
- Search covers safe document metadata only: display name and extension. It does not search body
  text, snippets, embeddings, or stored paths.
- Existing `GET /documents` callers retain newest-first behavior and a single `status=ready` query
  remains valid.
- Status filters use OR semantics. Deleted records never appear through this public endpoint.
- Sort values are enums mapped to fixed SQL fragments; user input never enters `ORDER BY`.
- The Documents surface requests one page at a time and renders actions for one selected document.
- Upload remains one obvious action but is collapsed when the library is non-empty.
- Ask keeps the existing four-section navigation and durable conversation API. The question box and
  primary action move ahead of suggestions and conversation-management detail.
- Collections, tags, rename, OCR, hosting, and conversation schema changes are deliberately out of
  scope. Rename remains deferred until SQLite, Qdrant payloads, and saved citation labels have one
  authoritative naming contract.

## File structure and ownership

- `src/personal_rag/models.py` — validated `DocumentSort` and `SortOrder` public enums.
- `src/personal_rag/database.py` — register one deterministic Unicode casefold SQLite function;
  schema version remains 2.
- `src/personal_rag/repository.py` — shared list/count filter builder and fixed deterministic sort
  mapping.
- `src/personal_rag/api/routes/documents.py` — bounded query/status/sort/order request contract.
- `src/personal_rag/ui/client.py` — typed page request with repeated status query parameters.
- `src/personal_rag/ui/app.py` — applied filter state, compact library master/detail, collapsed
  upload, and composer-first Ask layout.
- `src/personal_rag/ui/presentation.py` — selected-row and compact-library styling only if needed.
- `tests/unit/test_repository.py` — literal/Unicode search, multi-status filters, sort stability,
  truthful totals, deleted exclusion.
- `tests/api/test_api.py` — authenticated query contract, compatibility, and sanitized validation.
- `tests/ui/test_client.py` — exact request encoding and response parsing.
- `tests/ui/conftest.py`, `tests/ui/test_app.py` — one-page UI requests, selected detail, filter/page
  recovery, composer ordering.
- `tests/browser/fake_api.py`, `tests/browser/test_fake_api.py` — deterministic multi-page parity.
- `README.md`, `docs/api.md`, `docs/architecture.md`, `docs/operations.md`, `docs/security.md`,
  `docs/validation.md` — behavior and proof boundaries.
- `docs/personal-rag/2026-07-18-library-findability-phase-2.md` — completion record.

### Task 1: Literal metadata search and deterministic pagination

**Files:**
- Modify: `src/personal_rag/models.py`
- Modify: `src/personal_rag/database.py`
- Modify: `src/personal_rag/repository.py`
- Test: `tests/unit/test_repository.py`

- [x] **Step 1: Write failing repository tests**

Add at least 30 documents spanning ready, processing, failed, deletion-failed, and deleted states.
Assert Unicode casefold matching, literal `%`, `_`, quote, slash, and Markdown characters, OR status
filters, deleted exclusion, exact filtered counts, and stable page boundaries for every sort.

```python
def test_document_search_is_literal_unicode_and_counted(repository: Repository) -> None:
    create_named_document(repository, "Résumé 100%_plan.md", document_id="doc-special")
    assert [item.id for item in repository.list_documents(query="RÉSUMÉ 100%_")] == [
        "doc-special"
    ]
    assert repository.count_documents(query="RÉSUMÉ 100%_") == 1


def test_document_status_filters_use_or_semantics(repository: Repository) -> None:
    items = repository.list_documents(
        statuses=[DocumentStatus.FAILED, DocumentStatus.DELETION_FAILED]
    )
    assert {item.status for item in items} == {
        DocumentStatus.FAILED,
        DocumentStatus.DELETION_FAILED,
    }
```

- [x] **Step 2: Run the focused tests and confirm contract failures**

Run: `uv run pytest tests/unit/test_repository.py -q`

Expected: new tests fail because `query`, `statuses`, `sort`, and `order` are not implemented.

- [x] **Step 3: Add bounded public sort enums**

```python
class DocumentSort(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    NAME = "name"


class SortOrder(StrEnum):
    ASC = "asc"
    DESC = "desc"
```

- [x] **Step 4: Register deterministic Unicode casefold per SQLite connection**

Register `unicode_casefold(value)` through `sqlite3.Connection.create_function` so parameterized
`instr(unicode_casefold(display_name), ?)` handles Unicode without SQL wildcards. Keep schema
version 2 and backup format unchanged.

- [x] **Step 5: Implement one shared filter contract for list and count**

```python
def _document_filter_sql(
    self,
    *,
    status: DocumentStatus | None,
    statuses: Sequence[DocumentStatus] | None,
    query: str | None,
    include_deleted: bool,
) -> tuple[str, list[Any]]:
    """Return one parameterized predicate reused by list and count."""
```

Validate pagination and normalize a query to at most 200 visible characters. Map enum pairs to
hardcoded SQL such as `created_at DESC, id DESC` and
`unicode_casefold(display_name) ASC, id ASC`.

- [x] **Step 6: Run focused repository tests**

Run: `uv run pytest tests/unit/test_repository.py -q`

Expected: all repository tests pass with no duplicate/skipped stable rows.

### Task 2: Authenticated document query API

**Files:**
- Modify: `src/personal_rag/api/routes/documents.py`
- Modify: `tests/api/test_api.py`

- [x] **Step 1: Write failing API tests**

Cover default compatibility, repeated `status` parameters, `q`, every sort/order enum, filtered
totals, auth, overlong/control-character queries, and invalid enum values.

```python
response = client.get(
    "/api/v1/documents",
    params=[("status", "failed"), ("status", "deletion_failed"), ("q", "plan%_")],
    headers=auth_headers,
)
assert response.status_code == 200
assert response.json()["total"] == len(response.json()["items"])
```

- [x] **Step 2: Run focused API tests and confirm expected failures**

Run: `uv run pytest tests/api/test_api.py -q`

- [x] **Step 3: Extend the route without changing existing defaults**

Use bounded FastAPI query parameters. Normalize `q` in a small helper and return a sanitized 422
domain error for control characters. Pass the exact same filter values to `list_documents` and
`count_documents`.

- [x] **Step 4: Run API tests**

Run: `uv run pytest tests/api/test_api.py -q`

Expected: new and existing authenticated document contracts pass.

### Task 3: Typed page client

**Files:**
- Modify: `src/personal_rag/ui/client.py`
- Modify: `tests/ui/test_client.py`

- [x] **Step 1: Write failing exact-request tests**

Assert one page call encodes `q`, repeated `status`, `sort`, `order`, `limit`, and `offset`, while a
default call remains identical to the existing contract.

- [x] **Step 2: Implement the typed request**

```python
def list_documents(
    self,
    *,
    limit: int = 100,
    offset: int = 0,
    query: str | None = None,
    statuses: Sequence[DocumentStatus] | None = None,
    sort: DocumentSort = DocumentSort.CREATED,
    order: SortOrder = SortOrder.DESC,
) -> DocumentList:
    ...
```

Preserve bearer authentication, timeout handling, safe error envelopes, and server-only secrets.

- [x] **Step 3: Run client tests**

Run: `uv run pytest tests/ui/test_client.py -q`

### Task 4: Library-first master/detail Documents UI

**Files:**
- Modify: `src/personal_rag/ui/app.py`
- Modify: `src/personal_rag/ui/presentation.py`
- Modify: `tests/ui/conftest.py`
- Modify: `tests/ui/test_app.py`

- [x] **Step 1: Write failing AppTest behavior tests**

Assert that non-empty Documents renders Your library before Add documents, makes upload available
inside a collapsed labeled disclosure, calls `list_documents` for one page instead of
`list_all_documents`, exposes Search/status/sort controls, resets pagination when filters apply,
keeps selection within visible results, and renders one destructive confirmation panel.

- [x] **Step 2: Add explicit applied filter state**

Initialize `document_query`, `document_status_filter`, `document_sort`, `document_offset`, and
`selected_document_id`. Put search/status/sort in a form so typing does not request on every rerun.
Map human labels to explicit status lists:

```python
DOCUMENT_STATUS_GROUPS = {
    "All": None,
    "Ready": [DocumentStatus.READY],
    "Needs attention": [DocumentStatus.FAILED, DocumentStatus.DELETION_FAILED],
    "Processing": [
        DocumentStatus.QUEUED,
        DocumentStatus.VALIDATING,
        DocumentStatus.EXTRACTING,
        DocumentStatus.CHUNKING,
        DocumentStatus.EMBEDDING,
        DocumentStatus.INDEXING,
        DocumentStatus.REINDEXING,
        DocumentStatus.DELETING,
    ],
}
```

- [x] **Step 3: Render one page and one selected detail**

Use a compact, text-labeled selection list plus one detail panel. Keep status text, size/type/date,
technical details, Refresh, and typed permanent-removal confirmation. Clamp the page/selection when
filters or deletion remove the current record.

- [x] **Step 4: Collapse upload for non-empty libraries**

Render `st.expander("Add documents", expanded=False)` above the library toolbar. Empty-library
onboarding remains expanded and unchanged.

- [x] **Step 5: Run AppTest tests**

Run: `uv run pytest tests/ui/test_app.py -q`

Expected: one-page calls, state recovery, document actions, and hostile-name rendering pass.

### Task 5: Composer-first Ask workflow

**Files:**
- Modify: `src/personal_rag/ui/app.py`
- Modify: `tests/ui/test_app.py`

- [x] **Step 1: Write failing ordering and suggestion tests**

Assert the question text area and Ask library appear before suggestion buttons, and saved
conversation management remains reachable in one collapsed disclosure. Retain durable retry tests.

- [x] **Step 2: Move question and scope into one form**

Render the text area first. Put Look in and passage count inside `Where to look`, then render the
primary submit action. Place suggestions after the form.

- [x] **Step 3: Make post-form suggestions widget-safe**

When a suggestion is chosen, increment `question_draft_version`, seed a new not-yet-instantiated
draft key, and rerun. Never mutate the state of an already instantiated text area.

- [x] **Step 4: Collapse saved-conversation management**

Keep the currently selected conversation title visible, with previous/open/delete controls inside
a `Saved conversations` expander. Do not change the durable API or delete confirmation.

- [x] **Step 5: Run the complete UI suite**

Run: `uv run pytest tests/ui -q`

### Task 6: Deterministic browser parity and rendered proof

**Files:**
- Modify: `tests/browser/fake_api.py`
- Modify: `tests/browser/test_fake_api.py`

- [x] **Step 1: Add more than one deterministic result page**

Teach the fixture to apply the same literal query, repeated statuses, fixed sort, exact total, and
page slicing as the repository. Include hostile names and multiple status groups.

- [x] **Step 2: Add fixture contract tests**

Verify filter parity, stable ordering, pagination, and deleted exclusion.

- [x] **Step 3: Run rendered desktop/tablet/mobile proof**

Use the in-app Browser at 1440x1000, 768x1024, and 390x844. Exercise Search, status filter, sort,
pagination, selecting a document, expanding upload, and asking a suggested question. Verify page
identity, meaningful DOM, no framework overlay, fresh-tab console health, screenshot evidence,
interaction state, semantic headings, keyboard labels, and no horizontal overflow.

Phone acceptance:

- Search plus at least one library selection is visible in the first 844 pixels.
- Your question and Ask library are visible in the first 844 pixels for an existing conversation.
- Touch targets are plainly labeled and status is never conveyed by color alone.

### Task 7: Documentation, full gates, and completion handoff

**Files:**
- Modify: `README.md`
- Modify: `docs/api.md`
- Modify: `docs/architecture.md`
- Modify: `docs/operations.md`
- Modify: `docs/security.md`
- Modify: `docs/validation.md`
- Create: `docs/personal-rag/2026-07-18-library-findability-phase-2.md`
- Create: `docs/handoffs/2026-07-18-codex-personal-library-findability-phase-2.handoff.mdc`
- Create canonical home-scope handoff under
  `/Users/fortunevieyra/Documents/Github/beladed.com/docs/handoffs/`.

- [x] **Step 1: Update current behavior and limitations**

Remove the bounded local-search limitation from README and document literal metadata search,
multi-status filters, sort, pagination, master/detail actions, and the composer-first workflow.
Keep body search, rename, collections, live providers, hosted, and production proof explicitly out
of scope.

- [x] **Step 2: Run final deterministic gates**

```bash
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

- [ ] **Step 3: Write completion records and handoff**

Read the post-chat handoff rule, perform the complete next-item inventory, create the canonical
12-section `.mdc` plus repo-local pointer, and record exact commit/proof boundaries.

- [ ] **Step 4: Commit locally**

Commit the validated implementation and a documentation-only closeout. No remote is configured;
report committed locally and not pushed.

## Self-review

- Spec coverage: server-side findability, exact totals, stable pagination, library-first UI,
  one selected action panel, collapsed upload, composer-first mobile flow, deterministic fixture,
  browser proof, docs, security, and handoff are all assigned to tasks.
- Scope control: no schema migration, vector mutation, provider call, rename, collections, tags,
  OCR, hosting, tenancy, or dashboard work is included.
- Placeholder scan: the plan contains exact method signatures, queries, commands, expected results,
  and viewport acceptance criteria; no TBD/TODO step remains.
- Type consistency: `DocumentSort`, `SortOrder`, `query`, `statuses`, `limit`, and `offset` names are
  consistent across repository, route, client, fixture, and UI tasks.
