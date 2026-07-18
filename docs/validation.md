# Validation

## 0.2.0 — 2026-07-18

The public-beta release was validated with generated fixtures, deterministic local providers, an
isolated Compose stack, and the no-key product tour. No personal document or paid model request was
used.

| Check | Result |
|---|---|
| Quality gate | `make check` passed |
| Tests | 230 passed; one opt-in live-provider test deselected |
| Branch coverage | 82.25%, above the 80% gate |
| Formatting and lint | Ruff passed; 62 files in canonical format |
| Type safety | Strict mypy passed across 31 source files |
| Security scan | Bandit passed with no findings |
| Dependency audit | No known vulnerabilities; the unpublished local package was skipped |
| Public repository audit | Required community files, tracked links, and private-artifact checks passed |
| Dependency lock | `uv lock --check` passed for the locked 173-package graph |
| Package build | Source distribution and wheel built for version 0.2.0 |
| Compose | Example configuration rendered successfully |
| Container runtime | Fresh four-service start passed with zero restarts and healthy API/UI |
| Repository hygiene | `git diff --check` passed |

One upstream test-only warning remains: Starlette's compatibility `TestClient` reports that its
current httpx adapter is deprecated. It does not affect the running application and remains visible
instead of being suppressed.

### Rendered product proof

The deterministic tour was exercised in the in-app browser with simulated sample behavior clearly
labeled.

| Viewport | Result |
|---|---|
| Desktop, 1440 × 1000 | Ask, Library, Activity, source-backed history, scoped questions, and System status remained readable with no horizontal overflow |
| Tablet, 768 × 1024 | Question field and primary action remained visible; document width and scroll width both measured 768 pixels |
| Phone, 390 × 844 | Ask and the compact Library picker remained usable; document width and scroll width both measured 390 pixels |

The flow verified a bounded unsupported demo question, the known Atlas cited answer, document
selection, visible document-only scope and clearing, Activity state, System status, and return to
the primary workspace. The fixed light theme is intentional; dark-mode proof is not claimed.

### Container proof

A fresh no-cache image build from commit `87c52f7` produced
`sha256:e5dfd70f605c065228d9ab4c20d540b07ba26d76446ccba0cdda3e8866431c14`.
The Docker image listing fell from 2.04 GB to 1.26 GB after removing the package-manager cache. The
runtime user was the non-root `rag` account.

The isolated API, worker, UI, and Qdrant services started with zero restarts. API liveness,
readiness, version, authenticated status, and Streamlit health returned successful responses;
unauthenticated status returned 401. Readiness confirmed metadata, Qdrant, vector inventory,
provider configuration, and worker heartbeat. The first-start log scan found no Qdrant 409,
traceback, fatal startup, restart, OpenAI network, or Voyage network event. Disposable containers,
network, volumes, and proof configuration were removed afterward.

### Proof boundaries

- Complete: local source, deterministic tests, rendered local tour, package build, container image,
  isolated local runtime, and authenticated readback.
- Not run: live OpenAI or Voyage semantics, quota, latency, privacy, or cost proof.
- Not run: hosted-development or production deployment proof.
- Outside the product boundary: multi-user, multi-host, and high-availability behavior.

## Reproduce the deterministic gate

```bash
uv sync --all-groups --frozen
make check
uv lock --check
uv build
docker compose --env-file .env.example config --quiet
docker compose --env-file .env.example build
git diff --check
```

Run `uv run python scripts/demo.py` for the provider-free interface check. Use `make test-live` only
when you deliberately intend to make paid provider requests.
