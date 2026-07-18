# Changelog

This project follows semantic versioning. User-visible changes and important fixes are recorded
here.

## 0.2.0 — 2026-07-18

### Added

- guided secret-safe setup and configuration validation;
- an API-key-free deterministic product demo;
- public contribution, support, conduct, security-reporting, configuration, and extension guides;
- GitHub issue and pull-request templates;
- one typed supported-document registry shared by API, parser, and UI;
- public-repository hygiene checks in the normal quality gate.

### Changed

- reduced the first-run environment file to essential choices;
- simplified the interface around Add → Process → Ask → Sources;
- moved system diagnostics, filters, examples, completed jobs, and destructive document actions
  behind secondary disclosures;
- made document-scoped questions visibly scoped and easy to clear or reset with a new conversation;
- reduced long libraries to a compact document picker and removed empty-library filter clutter;
- made the secondary system view return cleanly to the main Ask, Library, and Activity workspace;
- labeled the no-key tour as simulated and made unsupported demo questions fail honestly;
- removed the package-manager cache from the runtime image to reduce first-pull size;
- rewrote the README around a five-minute first answer and honest beta/privacy/cost boundaries.

### Fixed

- made the documented demo command work from a fresh checkout;
- retried the harmless Qdrant conflict produced when API and worker create a new collection at the
  same time, while still validating the resulting collection profile.

## 0.1.0 — 2026-07-17

- initial single-user, single-host FastAPI/Streamlit RAG application;
- durable SQLite ingestion, conversation, reindex, and deletion jobs;
- OpenAI or Voyage embeddings, OpenAI answers, Qdrant storage, grounded citations, backup/restore,
  authenticated Compose deployment, and deterministic tests;
- server-side document search, status filtering, sorting, pagination, and responsive library UI.
