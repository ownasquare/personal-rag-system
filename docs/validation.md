# Validation Evidence

## Release result

The 2026-07-17 local deterministic release gate passed for the complete working tree. All tests
used generated fixtures, fake embedding and answer providers, temporary SQLite state, and local
Qdrant persistence. No personal document or paid provider call was used.

## Automated gates

| Gate | Result |
|---|---|
| Dependency lock | Passed; uv resolved 173 packages and uv lock --check exited successfully |
| Python package build | Passed; source distribution and wheel built successfully from the locked project |
| Ruff lint | Passed; all checks passed |
| Ruff format | Passed; 51 files formatted |
| Strict mypy | Passed; no issues in 28 source files |
| Bandit | Passed; no findings in src or scripts |
| Dependency audit | Passed; no known vulnerabilities; the local non-PyPI project package was the only expected skip |
| Pytest | Passed; 124 deterministic tests; the opt-in live-provider test was deselected |
| Branch coverage | Passed; 81.19 percent against an 80 percent gate |
| Compose source parse | Passed; api, qdrant, ui, and worker services were present and the Qdrant host binding stayed on loopback |
| Repository hygiene | Passed; no unresolved placeholder markers in source, tests, scripts, or configuration; no whitespace errors |

One warning remains in the test environment: FastAPI's TestClient compatibility module emits an
upstream Starlette deprecation warning about the httpx test adapter. It does not affect runtime
code and is kept visible rather than suppressed.

The initial Chroma dependency was removed after the current advisory database reported
PYSEC-2026-311 with no fixed version. The user-approved Qdrant option was substituted, the lock was
regenerated, vector behavior was retested, and pip-audit then completed without a vulnerability.
The remaining Chroma references are historical decision records in documentation only.

## Final audit regressions

The release audit added focused proofs beyond the original suite:

- A restored database uses portable upload keys; reindex and delete operate only in the restored
  tree and leave the original source untouched.
- Production rejects embedded or in-memory Qdrant and missing provider credentials before any
  collection is created.
- A profile-changing reindex atomically updates the manifest fingerprint, and a later identical
  upload deduplicates against the migrated document.
- Any malformed or out-of-range model citation causes the complete answer to abstain.
- Qdrant query failures produce sanitized retry-aware 503 errors rather than generic 500s.
- Backup output inside the application data tree is refused before a recursive copy can start.
- The required document_id keyword index is created and verified after a simulated interrupted
  first attempt.
- Local default tests deselect the paid live test and disable network sockets.

## Deterministic browser proof

The Streamlit interface was exercised in the in-app browser against a temporary FastAPI fixture
that returned sanitized, deterministic library and citation data.

| Viewport | Observed result |
|---|---|
| Desktop, 1440 by 1000 | Chat, Library, and Settings and Status rendered and were navigable |
| Tablet, 768 by 1024 | Cited chat flow rendered with document controls and zero horizontal overflow |
| Phone, 390 by 844 | All three tabs remained reachable, the cited answer remained readable, and zero horizontal overflow was measured |

The chat interaction submitted a question, rendered a grounded answer containing marker S1, and
showed the matching atlas-launch-notes.md citation. Library showed the ready fixture document.
Settings and Status showed sanitized model and dependency information. The browser console
contained zero warnings or errors after the flow. Temporary proof services were stopped.

The checked-in Streamlit theme is intentionally a fixed light theme. Dark-mode proof is therefore
not claimed or required by this release.

## Proof boundaries

- Local source, deterministic test, and deterministic browser proof: complete.
- Docker image build and Compose runtime proof: not run locally because the Docker executable is
  unavailable. The checked-in CI container job builds the topology and waits for API, worker,
  vector-inventory, and UI health, but that remote job has not run because this repository has no
  remote.
- Live OpenAI or Voyage semantic, quota, latency, privacy, and cost proof: not run. No credential
  was supplied and no provider credit was spent.
- Hosted-development proof: not run.
- Production deployment proof: not run.
- Multi-user, multi-host, and high-availability proof: outside the documented product boundary.

## Reproduce the local gate

    uv sync --all-groups --frozen
    uv lock --check
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy src
    uv run bandit -q -r src scripts
    uv run pip-audit
    uv run pytest -m "not live" --disable-socket --allow-unix-socket --cov --cov-report=term-missing

For a Docker-capable host, configure a private env file and then run:

    docker compose config --quiet
    docker compose build
    docker compose up -d
    curl --fail http://127.0.0.1:8000/health/ready
    curl --fail http://127.0.0.1:8501/_stcore/health

Stop and remove the proof stack after validation. Do not use real personal data for the first
deployment smoke test.
