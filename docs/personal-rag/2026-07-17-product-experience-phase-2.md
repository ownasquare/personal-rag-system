# Personal Library Phase 2 Product Experience Completion

## Completion status

Phase 2 is complete and locally validated as of 2026-07-17. The production-safe Phase 1 service is
now presented as **Personal Library**: a calm, familiar document workspace rather than a generic
AI chat demo. The release adds durable cited conversations and visible operational recovery while
preserving the original stateless chat API for compatibility.

Validated implementation commit: `de69b90254e5404e3d4a3abd90406d78d307d984` on local `main`.
No Git remote is configured, so this release is committed locally and was not pushed.

## Product outcome

The primary interface now has four plainly named areas:

- **Ask** keeps the question composer ahead of saved history, supports document scope, offers
  useful starter questions, and makes supporting passages easy to open.
- **Documents** provides upload, search, lifecycle state, reindex recovery, and confirmed permanent
  removal without exposing storage internals.
- **Activity** separates In progress, Needs attention, and Completed work and uses an explicit
  Refresh action instead of an endless background rerun.
- **System** keeps sanitized dependency and setup information available without giving operational
  details equal prominence to the everyday workflow.

The warm paper, charcoal, and muted-green visual system uses restrained borders and familiar form
controls. It avoids gradients, robot imagery, chat bubbles, novelty dashboards, and decorative
model terminology. Desktop, tablet, and phone layouts were reviewed in the rendered application.
The final phone header uses a compact 20.48-pixel title, hides secondary hero copy, preserves the
Ask/Documents/Activity/System navigation, and has no horizontal overflow.

## Durable conversation model

SQLite schema v2 adds conversations, turns, and normalized citations. The API supports creating,
listing, opening, and deleting conversations and submitting idempotent turns. Completed answers,
citations, pending state, retryable failure state, and request metadata survive browser and process
restart. The UI consumes pagination and loads the latest bounded turn window without separating
question/answer pairs.

The original `POST /api/v1/chat` route remains available for stateless clients.

## Concurrency and privacy hardening

- Each paid turn attempt receives a random reservation ownership token and a renewable 120-second
  lease. The API renews active work every 30 seconds while the provider call runs.
- Completion and failure updates are fenced by that token. An expired attempt cannot overwrite a
  newer retry or recovery attempt.
- The request fingerprint binds the client turn ID to question, scope, and retrieval settings.
  Editing a retry rotates the client turn ID instead of causing an idempotency conflict.
- Citation completion rechecks every cited source inside the same SQLite write transaction. A
  source deleted or made non-ready during provider execution becomes a safe retryable
  `source_changed` failure.
- Successful document deletion removes saved turns that cite the source and removes or retitles
  conversations left without retained content.
- Durable job history is exposed through an authenticated paginated endpoint. The browser fixture
  now follows real queued, running, succeeded, conflict, and deletion-purge behavior closely enough
  for deterministic workflow proof.

## Accessibility and interaction details

- The rendered heading order is H1, H2, then H3.
- Navigation has an accessible Workspace radiogroup.
- Source-derived names are rendered as text in document and activity surfaces.
- Retryable submitted turns expose one-click recovery after a page refresh.
- Upload error state clears after a later successful upload.
- Terminal job history does not keep the page polling forever.
- Destructive document and conversation actions require explicit confirmation.

## Validation record

The exact final tree passed:

- 160 deterministic tests, one opt-in live-provider test deselected;
- 81.85 percent branch coverage against an 80 percent gate;
- Ruff format and lint;
- strict mypy across 30 source files;
- Bandit and pip-audit with no known vulnerabilities;
- locked source/wheel package build;
- Compose configuration and final application image build;
- isolated local Compose startup with healthy API and UI, ready metadata/Qdrant/vector
  inventory/provider configuration/worker checks, authenticated status readback, and complete
  teardown verification;
- deterministic browser proof at 1440 by 1000, 768 by 1024, and 390 by 844 with no final console
  messages, framework errors, or horizontal overflow.

The full command record and evidence boundaries are in
[`docs/validation.md`](../validation.md).

## Proof boundaries

The release has local deterministic, rendered-browser, image-build, and isolated Compose-runtime
proof. Browser data was a running deterministic fixture; repository restart tests provide the
separate persistence proof. The Compose placeholder key proved configuration only and made no
model request.

No live OpenAI/Voyage request, hosted-development deployment, or production deployment was run.
Provider semantics, quota, latency, privacy policy, and cost therefore remain explicitly unproved.
The product remains intentionally single-user and single-host.

## Deferred opportunities

Useful later phases could add tags/collections, server-side full-library search, document renaming,
OCR for image-only PDFs, export/import for conversations, and an optional identity-aware hosted
deployment. None is required for this Phase 2 completion, and none should weaken the current
source-grounding, deletion, or single-user privacy contracts.
