# Open-source adoption readiness — 2026-07-18

## Outcome

Personal Library 0.2.0 is ready for a public-beta GitHub release. A new user can preview the full
interface without a provider key, configure a real local library through one setup assistant, and
follow a short Add → Process → Ask → Check the source workflow. Contributors have a repository map,
extension recipes, deterministic tests, community templates, and an explicit proof boundary.

## Why this release was needed

The earlier application was feature-complete but still read like an internal implementation. Setup
required too much prior knowledge, the interface exposed secondary controls too early, and the
repository did not yet explain how strangers should evaluate, operate, or extend it.

## What changed

- Added a no-key simulated tour and a secret-safe setup assistant.
- Rewrote the README around one first answer and added a current interface preview.
- Centered the product on Ask, Library, and Activity; moved diagnostics and maintenance behind
  secondary controls and help text.
- Made document-only question scope visible and easy to clear.
- Replaced long document radio lists with one compact picker and removed empty-library filter
  clutter.
- Added contribution, support, conduct, security-reporting, configuration, and extension guidance.
- Added issue and pull-request templates plus a tracked-public-repository audit.
- Centralized supported document types so API, parser, demo, security guidance, and UI stay aligned.
- Fixed the fresh-checkout tour command and the concurrent Qdrant collection-creation race.
- Removed the package-manager cache from the image, reducing the Docker listing from 2.04 GB to
  1.26 GB.

## Affected surfaces

- First run: `README.md`, `scripts/setup.py`, `scripts/demo.py`, and environment examples.
- Product interface: `src/personal_rag/ui/`.
- Runtime reliability: `src/personal_rag/vector_store.py` and `Dockerfile`.
- Extension contract: `src/personal_rag/document_types.py`, parser/API integration, and
  `docs/extending.md`.
- Project adoption: `CONTRIBUTING.md`, `SECURITY.md`, `SUPPORT.md`, `CODE_OF_CONDUCT.md`, `.github/`,
  and the public-repository audit.

## Commits

- `1899778` — initial public-adoption productization.
- `169dcba` — adversarial release corrections, clean startup, concise proof, and visual preview.
- `3014594` — public release record and adoption evidence.
- `87c52f7` — browser-compatible preview asset correction.
- `0ee02d9` — exact final container-image proof.

## Validation

- `make check`: passed; 230 tests passed, one live-provider test deselected, 82.25% branch coverage.
- Ruff, strict mypy, Bandit, dependency audit, public-repository audit, and lock validation: passed.
- Source distribution and wheel for 0.2.0: built successfully.
- Fresh isolated Compose start: four services, zero restarts, healthy API/UI, ready metadata/vector/
  provider/worker checks, protected status readback, and no provider network call.
- Browser proof: desktop 1440 × 1000, tablet 768 × 1024, and phone 390 × 844 with no horizontal
  overflow across the core Ask, Library, Activity, scoped-search, and System flows.

Detailed evidence and reproduction commands are in [Validation](../validation.md).

## Proof boundary

This release proves local deterministic behavior, packaging, a production-shaped isolated
container topology, and the public onboarding path. It does not claim a live OpenAI or Voyage
request, hosted deployment, production deployment, multi-user isolation, or high availability.
