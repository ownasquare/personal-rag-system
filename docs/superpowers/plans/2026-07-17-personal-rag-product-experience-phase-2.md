# Personal RAG Product Experience Phase 2 Implementation Plan

**Implementation status (2026-07-17): Complete and validated.** The adversarial review added
reservation ownership fencing, source-state revalidation, latest-window pagination,
persisted-turn retry controls, and deterministic browser-fixture progression beyond the original
plan. The final gate passed 160 deterministic tests with 81.85 percent branch coverage, clean
static/security/package checks, rendered desktop/tablet/mobile proof, and isolated Compose health
readback.

The unchecked steps below are retained as the original implementation recipe, not as the current
tracker. Authoritative completion evidence is in
`docs/personal-rag/2026-07-17-product-experience-phase-2.md` and `docs/validation.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the production-safe Phase 1 RAG service into a calm personal library with durable conversations, refresh-safe activity, clear document lifecycle controls, and a responsive interface that does not resemble a generic AI demo.

**Architecture:** Add a forward-only SQLite v2 conversation model and authenticated conversation-turn API while retaining the stateless `/chat` contract for compatibility. Expose the repository's existing durable job history through a paginated API, then rebuild the Streamlit shell around conditionally rendered Ask, Documents, and Activity views with secondary system details. Browser proof uses a stateful no-network FastAPI fixture and keeps fixture, live-provider, container, hosted, and production evidence explicitly separate.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, SQLite/WAL, LlamaIndex service boundary, httpx, Streamlit 1.59, pytest/AppTest, Ruff, mypy, Bandit, pip-audit, in-app Browser/Playwright.

---

## File Structure and Ownership

- `src/personal_rag/models.py` — public conversation, turn, citation, and job-list contracts.
- `src/personal_rag/database.py` — forward-only SQLite v2 migration for conversation truth.
- `src/personal_rag/repository.py` — transactional conversation reservations, persisted turns/citations, recent activity, and deletion privacy.
- `src/personal_rag/api/routes/conversations.py` — authenticated conversation and turn endpoints.
- `src/personal_rag/api/routes/jobs.py` — paginated recent-job endpoint in addition to ID readback.
- `src/personal_rag/api/app.py` — route registration only; stateless chat remains compatible.
- `src/personal_rag/ui/client.py` — typed conversation/activity client with existing safe error normalization.
- `src/personal_rag/ui/presentation.py` — trusted static CSS and human-facing format helpers; never accepts document-derived HTML.
- `src/personal_rag/ui/app.py` — conditional workspace navigation, durable thread orchestration, onboarding, documents, activity, and secondary system view.
- `tests/unit/test_repository.py` — v1-to-v2 migration, idempotency, citation persistence, ordering, privacy purge.
- `tests/api/test_api.py` — authenticated conversation/activity contracts and one-call provider behavior.
- `tests/ui/test_client.py` — typed client request/response coverage.
- `tests/ui/conftest.py` — durable fake client state used by AppTest.
- `tests/ui/test_app.py` — workspace navigation and behavior coverage.
- `tests/browser/fake_api.py` — stateful rendered-proof fixture for conversations, jobs, cited answers, recovery, and deletion.
- `.streamlit/config.toml` — warm neutral theme tokens with modest radii and a muted green accent.
- `README.md`, `docs/api.md`, `docs/architecture.md`, `docs/operations.md`, `docs/security.md`, `docs/validation.md` — Phase 2 behavior and proof boundaries.
- `docs/personal-rag/2026-07-17-product-experience-phase-2.md` — required completion record.

### Task 1: Durable Conversation Persistence and Privacy

**Files:**
- Modify: `src/personal_rag/models.py`
- Modify: `src/personal_rag/database.py`
- Modify: `src/personal_rag/repository.py`
- Modify: `src/personal_rag/job_service.py`
- Test: `tests/unit/test_repository.py`
- Test: `tests/unit/test_backup_restore.py`

- [ ] **Step 1: Write failing migration and repository tests**

Add tests that create a populated schema-v1 database, run `Database.initialize()`, and assert schema version 2 plus unchanged document/job rows. Add repository tests for conversation creation/list ordering, turn reservation, completed duplicate readback, active-reservation conflict, expired-reservation recovery, citation round-trip, bounded history, hard conversation deletion, and cited-turn purge during source deletion.

```python
def test_v1_database_migrates_conversations_without_changing_documents(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.sqlite3")
    database.initialize()
    receipt = Repository(database).create_document_with_job(...)
    database.connection().__enter__().execute("PRAGMA user_version = 1")
    Database(database.path).initialize()
    assert Database(database.path).schema_version() == 2
    assert Repository(database).get_document(receipt.document.id) is not None


def test_completed_turn_is_idempotent_and_round_trips_citations(repository: Repository) -> None:
    conversation = repository.create_conversation("Atlas notes")
    reservation = repository.reserve_conversation_turn(
        conversation.id,
        client_turn_id="turn-client-1",
        question="What is the key?",
        top_k=5,
        document_ids=["doc-1"],
        request_fingerprint="a" * 64,
    )
    completed = repository.complete_conversation_turn(
        reservation.turn.id,
        answer="Cobalt [S1].",
        citations=[citation],
        no_answer=False,
        request_id="request-1",
    )
    duplicate = repository.reserve_conversation_turn(...)
    assert duplicate.cached_turn == completed
    assert duplicate.created is False
```

- [ ] **Step 2: Run focused tests and verify the new contracts fail**

Run: `uv run pytest tests/unit/test_repository.py tests/unit/test_backup_restore.py -q`

Expected: failures for missing schema-v2 tables, models, and repository methods; all pre-existing tests remain collectable.

- [ ] **Step 3: Add explicit conversation and turn models**

Define `ConversationSummary`, `ConversationList`, `ConversationCreate`, `ConversationTurnStatus`, `ConversationTurn`, `ConversationTurnList`, `ConversationTurnCreate`, and `ConversationTurnReservation`. Enforce bounded title/question/client ID/document ID lengths with Pydantic fields. Store `Citation` values as typed children and keep safe failure fields separate from exception text.

```python
class ConversationTurnStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class ConversationTurnCreate(BaseModel):
    client_turn_id: str = Field(min_length=8, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[str] | None = Field(default=None, max_length=100)


class ConversationTurn(BaseModel):
    id: str
    conversation_id: str
    client_turn_id: str
    status: ConversationTurnStatus
    question: str
    answer: str | None
    citations: list[Citation] = Field(default_factory=list)
    no_answer: bool
    error_code: str | None
    retryable: bool
    created_at: datetime
    updated_at: datetime
```

- [ ] **Step 4: Add a forward-only SQLite v2 migration**

Set `SCHEMA_VERSION = 2`. Add `_MIGRATION_V2` with `conversations`, `conversation_turns`, and normalized `turn_citations`; include foreign keys, status checks, unique `(conversation_id, client_turn_id)`, request fingerprints, reservation expiry, and descending-list indices. Apply v1 then v2 in order within the existing file lock.

```sql
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE conversation_turns (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    client_turn_id TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'completed', 'failed')),
    question TEXT NOT NULL,
    answer TEXT,
    no_answer INTEGER NOT NULL DEFAULT 0 CHECK (no_answer IN (0, 1)),
    top_k INTEGER,
    document_ids_json TEXT,
    request_id TEXT,
    error_code TEXT,
    retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),
    reservation_expires_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (conversation_id, client_turn_id)
);
```

- [ ] **Step 5: Implement transactional repository methods**

Implement create/list/count/get/delete conversation; list/count turns; reserve/complete/fail turn; reconstruct completed history; and purge every whole turn whose normalized citation references a deleted document. Derive the first title deterministically from the sanitized first question and update conversation ordering only when a turn completes.

```python
def reserve_conversation_turn(
    self,
    conversation_id: str,
    *,
    client_turn_id: str,
    question: str,
    top_k: int | None,
    document_ids: Sequence[str] | None,
    request_fingerprint: str,
    reservation_seconds: int = 120,
) -> ConversationTurnReservation:
    """Atomically reserve one paid turn or return its persisted result."""


def purge_conversation_turns_for_document(
    self, connection: sqlite3.Connection, document_id: str
) -> int:
    """Delete whole turns that retain content from a document being deleted."""
```

- [ ] **Step 6: Join privacy purge to successful deletion completion**

Inside the same transaction that marks a delete job complete, purge cited turns before the document becomes `deleted`. If purge fails, leave the job retryable/deletion-failed rather than claiming a privacy-complete deletion. Vector/file removal is already idempotent, so retry remains safe.

- [ ] **Step 7: Run focused persistence tests**

Run: `uv run pytest tests/unit/test_repository.py tests/unit/test_backup_restore.py -q`

Expected: all repository, migration, deletion, backup, and restore tests pass.

- [ ] **Step 8: Commit persistence slice**

```bash
git add src/personal_rag/models.py src/personal_rag/database.py src/personal_rag/repository.py src/personal_rag/job_service.py tests/unit/test_repository.py tests/unit/test_backup_restore.py
git commit -m "feat: persist private conversation history"
```

### Task 2: Conversation Turns and Recent Activity API

**Files:**
- Create: `src/personal_rag/api/routes/conversations.py`
- Modify: `src/personal_rag/api/routes/jobs.py`
- Modify: `src/personal_rag/api/app.py`
- Modify: `src/personal_rag/models.py`
- Test: `tests/api/test_api.py`

- [ ] **Step 1: Write failing API contract tests**

Cover authentication, create/list/get/delete conversation, paginated completed turns, one provider call per client turn, completed duplicate caching, active duplicate 409, expired reservation recovery, safe provider failure, authoritative citation persistence, and paginated recent jobs.

```python
def test_duplicate_completed_turn_uses_persisted_result_once(client, auth_headers) -> None:
    conversation = client.post(
        "/api/v1/conversations", headers=auth_headers, json={"title": "Atlas"}
    ).json()
    body = {"client_turn_id": "client-turn-001", "message": "What is the key?"}
    first = client.post(
        f"/api/v1/conversations/{conversation['id']}/turns",
        headers=auth_headers,
        json=body,
    )
    second = client.post(
        f"/api/v1/conversations/{conversation['id']}/turns",
        headers=auth_headers,
        json=body,
    )
    assert first.json() == second.json()
    assert client.app.state.container.rag_service.call_count == 1
```

- [ ] **Step 2: Run API tests and confirm expected failures**

Run: `uv run pytest tests/api/test_api.py -q`

Expected: 404/missing-model failures for new routes while existing `/chat` tests remain green.

- [ ] **Step 3: Implement conversation endpoints around the existing RAG service**

Reserve the client turn before provider work, rebuild bounded history only from completed persisted turns, call `RAGService.chat()`, persist citations, and return typed server truth. A completed duplicate returns HTTP 200 with no provider call; an active reservation raises retryable `RagError(..., status_code=409)`; a failed provider call stores only its safe error code/retryability and preserves the client question for retry.

```python
@router.post("/{conversation_id}/turns", response_model=ConversationTurn)
async def create_turn(
    conversation_id: str,
    body: ConversationTurnCreate,
    request: Request,
) -> ConversationTurn:
    repository = cast(Repository, request.app.state.container.repository)
    reservation = repository.reserve_conversation_turn(...)
    if reservation.cached_turn is not None:
        return reservation.cached_turn
    history = repository.conversation_history(
        conversation_id,
        limit=request.app.state.container.settings.max_history_messages,
    )
    response = await anyio.to_thread.run_sync(
        partial(request.app.state.container.rag_service.chat, ChatRequest(...))
    )
    return repository.complete_conversation_turn(...)
```

- [ ] **Step 4: Expose recent durable jobs**

Add `JobList` and `GET /api/v1/jobs?limit=&offset=&status=&document_id=` by composing existing `Repository.list_jobs()` and `count_jobs()`. Keep `GET /jobs/{job_id}` unchanged.

- [ ] **Step 5: Register the router and run API tests**

Run: `uv run pytest tests/api/test_api.py -q`

Expected: all authenticated conversation/activity contracts and existing stateless chat contracts pass.

- [ ] **Step 6: Commit API slice**

```bash
git add src/personal_rag/api/routes/conversations.py src/personal_rag/api/routes/jobs.py src/personal_rag/api/app.py src/personal_rag/models.py tests/api/test_api.py
git commit -m "feat: add durable conversations and activity API"
```

### Task 3: Typed Workspace Client

**Files:**
- Modify: `src/personal_rag/ui/client.py`
- Modify: `tests/ui/test_client.py`

- [ ] **Step 1: Write failing client tests**

Add exact path/method/body assertions for create/list/get/delete conversations, list/create turns, and list jobs. Verify the bearer token remains server-side and typed error envelopes still hide raw provider responses.

- [ ] **Step 2: Run client tests and verify missing methods fail**

Run: `uv run pytest tests/ui/test_client.py -q`

Expected: attribute/model failures for the new client surface.

- [ ] **Step 3: Implement the narrow typed client methods**

```python
def create_conversation(self, title: str | None = None) -> ConversationSummary:
    data = self._request_json(
        "POST", "/api/v1/conversations", json_body=ConversationCreate(title=title).model_dump()
    )
    return self._parse_model(data, ConversationSummary)

def create_conversation_turn(
    self, conversation_id: str, turn: ConversationTurnCreate
) -> ConversationTurn:
    data = self._request_json(
        "POST",
        f"/api/v1/conversations/{self._path_identifier(conversation_id)}/turns",
        json_body=turn.model_dump(mode="json", exclude_none=True),
    )
    return self._parse_model(data, ConversationTurn)

def list_jobs(self, *, limit: int = 50, offset: int = 0) -> JobList:
    data = self._request_json("GET", "/api/v1/jobs", params={"limit": limit, "offset": offset})
    return self._parse_model(data, JobList)
```

- [ ] **Step 4: Run client tests**

Run: `uv run pytest tests/ui/test_client.py -q`

Expected: all typed client tests pass with bounded parameter validation and safe failures.

### Task 4: Calm Personal Library Workspace

**Files:**
- Create: `src/personal_rag/ui/presentation.py`
- Modify: `src/personal_rag/ui/app.py`
- Modify: `.streamlit/config.toml`
- Modify: `tests/ui/conftest.py`
- Modify: `tests/ui/test_app.py`

- [ ] **Step 1: Replace shape-dependent AppTests with behavior tests**

Cover default context-aware Ask/onboarding view, conditional navigation, no inactive-page health calls, explicit multi-upload, immediate activity tracker after enqueue, durable thread restoration, new/clear conversation, retryable question preservation, scoped document selection, advanced retrieval control hidden by default, citation detail without raw score, no-answer state, library search, reindex, deletion confirmation, API-down recovery, and hostile filename/snippet escaping.

```python
def test_ready_workspace_defaults_to_ask_without_rendering_system_calls(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    fake_client.documents = [make_document()]
    result = app_test.run()
    assert "Ask your library" in _visible_text(result)
    assert fake_client.health_live_calls == 0
    assert fake_client.health_ready_calls == 0


def test_enqueued_upload_reruns_into_visible_activity(
    app_test: AppTest, fake_client: FakeRagClient
) -> None:
    result = _navigate(app_test.run(), "Documents")
    result.file_uploader[0].set_value([("alpha.md", b"alpha", "text/markdown")]).run()
    _button(result, "Add to library").click().run()
    assert "Adding alpha.md" in _visible_text(result)
    assert fake_client.list_jobs_calls > 0
```

- [ ] **Step 2: Run AppTests and verify the old shell fails the new experience contract**

Run: `uv run pytest tests/ui/test_app.py -q`

Expected: failures for navigation, persistence, request counts, activity hydration, and copy changes.

- [ ] **Step 3: Add the static presentation module and warm theme**

Use warm off-white, charcoal, muted forest accent, 8–12px radii, thin borders, modest type sizes, and no gradients. Hide Streamlit's default robot/user avatars through trusted static selectors and render answers as plain reading panels. Never interpolate document/provider text into `unsafe_allow_html`.

```python
STATIC_STYLES = """
<style>
  :root { --paper: #f6f3ed; --ink: #252722; --line: #d9d5cb; --accent: #355f4b; }
  .stApp { background: var(--paper); color: var(--ink); }
  .block-container { max-width: 1120px; padding-top: 1.5rem; padding-bottom: 4rem; }
  div[data-testid="stChatMessageAvatarUser"],
  div[data-testid="stChatMessageAvatarAssistant"] { display: none; }
  div[data-testid="stChatMessage"] { border: 0; border-radius: 0; background: transparent; }
</style>
"""
```

- [ ] **Step 4: Replace eager tabs with conditional workspace navigation**

Render a compact wordmark and plain promise: `Personal Library` and `Your documents, ready when you need them.` Navigation exposes `Ask`, `Documents`, and `Activity`; `System` is secondary. Only the selected page calls its page-specific endpoints. Keep a recent-conversation sidebar on wide screens and a normal navigation affordance on mobile.

- [ ] **Step 5: Build context-aware onboarding and Ask page**

When there are no documents, show one upload surface and a three-step explanation. With ready documents, show the durable selected conversation, source scope, composer, and a collapsed `Search options` expander containing `top_k`. Render user questions without avatars and answers under `From your library`; source rows show filename/page/section while relevance scores remain only in `System diagnostics`.

- [ ] **Step 6: Build Documents and Activity views**

Documents use compact rows with human statuses (`Ready`, `Processing`, `Needs attention`) and secondary details for size/chunks/version. Destructive controls stay behind an explicit document details expander and exact-name confirmation. Activity merges server job truth with newly tracked IDs, reruns immediately after enqueue, and provides a visible refresh action without keeping an idle background polling loop alive.

- [ ] **Step 7: Make no-answer, failures, loading, and recovery first-class**

Differentiate `no_answer` from a normal response, preserve retryable drafts and client turn IDs, avoid duplicate paid calls, explain missing provider setup in user language, and keep detailed model/dependency metadata inside the secondary System view.

- [ ] **Step 8: Run and stabilize AppTests**

Run: `uv run pytest tests/ui/test_app.py tests/ui/test_client.py -q`

Expected: all workspace behavior and client tests pass; inactive screens have asserted zero unnecessary health calls.

- [ ] **Step 9: Commit workspace slice**

```bash
git add src/personal_rag/ui/presentation.py src/personal_rag/ui/app.py src/personal_rag/ui/client.py .streamlit/config.toml tests/ui/conftest.py tests/ui/test_app.py tests/ui/test_client.py
git commit -m "feat: redesign the personal library workspace"
```

### Task 5: Stateful Browser Fixture and Responsive Proof

**Files:**
- Modify: `tests/browser/fake_api.py`
- Modify: `docs/validation.md`

- [ ] **Step 1: Extend the no-network fixture**

Add in-memory conversations/turns/jobs plus authenticated endpoints matching the new contracts. Include one ready document, one needs-attention document, a recent completed thread, deterministic cited answers, job progression on reads, reindex, and delete behavior. Keep the fixture explicitly labeled and never call providers.

- [ ] **Step 2: Run the local fixture and Streamlit through the secret-safe launcher**

Run the fixture API on `127.0.0.1:8012` and Streamlit on `127.0.0.1:8512` using the fixture bearer token only inside the server process.

Expected: both health endpoints report ready and Streamlit loads with no configuration error.

- [ ] **Step 3: Prove the desktop flow at 1440×1000**

Using the in-app Browser first, verify URL/title, meaningful DOM, no blocking overlay, zero relevant console warnings/errors, and a screenshot. Exercise: open prior conversation, ask and receive one cited answer, inspect source detail, switch to Documents, reindex the needs-attention document, confirm immediate Activity visibility, and open System details.

- [ ] **Step 4: Prove tablet and mobile layouts**

Repeat layout and core navigation checks at 768×1024 and 390×844. Verify no horizontal overflow, readable header, accessible navigation, clear primary action, usable composer, long filenames, and destructive confirmation. Capture and visibly inspect screenshots.

- [ ] **Step 5: Record exact fixture proof boundary**

Document that this proves rendered local UI/API integration only. Do not claim live-provider, Docker runtime, hosted, or production proof unless those layers are separately executed.

### Task 6: Documentation, Full Gates, and Handoff

**Files:**
- Modify: `README.md`
- Modify: `docs/api.md`
- Modify: `docs/architecture.md`
- Modify: `docs/operations.md`
- Modify: `docs/security.md`
- Modify: `docs/validation.md`
- Create: `docs/personal-rag/2026-07-17-product-experience-phase-2.md`
- Create: `docs/handoffs/2026-07-17-codex-personal-rag-phase-2.handoff.mdc`
- Create: `/Users/fortunevieyra/Documents/Github/beladed.com/docs/handoffs/2026-07-17-codex-personal-rag-phase-2.handoff.mdc`

- [ ] **Step 1: Update product and operator documentation**

Document durable thread behavior, activity recovery, source-deletion privacy, new API routes, SQLite v2 migration, backup contents, UX workflow, and explicitly deferred tags/collections/server search. Remove the Phase 1 session-only chat limitation.

- [ ] **Step 2: Run formatting and static gates**

Run:

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run bandit -q -r src
uv run pip-audit
git diff --check
```

Expected: all commands exit 0 with no security findings.

- [ ] **Step 3: Run deterministic tests and coverage**

Run: `uv run pytest -m 'not live' --disable-socket --allow-unix-socket --cov=personal_rag --cov-report=term-missing --cov-branch -q`

Expected: all deterministic tests pass, one live test remains deselected, and branch coverage is at least 80%.

- [ ] **Step 4: Run package and deployment-shaped checks**

Run: `uv build`, `docker compose config`, and, only if Docker is available, `docker compose build && docker compose up -d` followed by API readiness, Streamlit health, and one browser load through Compose.

Expected: package and Compose parse/build pass. If Docker is unavailable, record that exact local blocker without turning it into a product failure or a runtime proof claim.

- [ ] **Step 5: Write completion documentation and final handoff**

Write the required repository completion record, update validation evidence, read the full post-chat handoff rule, and create the canonical 12-section `.mdc` plus repo-local mirror. Include commit hash, exact tests, screenshots inspected, blockers, proof boundaries, and next backlog.

- [ ] **Step 6: Commit final documentation and report delivery truth**

```bash
git add README.md docs .streamlit src tests
git commit -m "docs: close personal library phase two"
git status --short --branch
git log -3 --oneline --decorate
```

Expected: clean branch, validated Phase 2 commits present. With no remote configured, report `committed locally, not pushed` and do not imply CI/hosted deployment.

## Self-Review

- Spec coverage: the plan includes the non-AI-looking interface, ease of use, onboarding, durable conversations, document lifecycle recovery, responsive behavior, security preservation, validation, completion docs, and handoff.
- Deliberate deferrals: tags, collections, renaming, full-text document search, and provider/hosted deployment are not required for this productization slice; each adds cross-store or proof scope without fixing the observed primary workflow.
- Placeholder scan: implementation signatures, schemas, commands, expected outcomes, and proof boundaries are explicit; no open-ended code instruction remains.
- Type consistency: conversation/turn/job models, repository methods, API routes, client methods, fixture contracts, and UI tests use the same names throughout.
