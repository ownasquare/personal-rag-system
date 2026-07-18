# Changelog

This project follows semantic versioning. User-visible changes are recorded here; internal agent
plans and machine-specific handoffs are intentionally excluded from the public repository.

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
- rewrote the README around a five-minute first answer and honest beta/privacy/cost boundaries.

## 0.1.0 — 2026-07-17

- initial single-user, single-host FastAPI/Streamlit RAG application;
- durable SQLite ingestion, conversation, reindex, and deletion jobs;
- OpenAI or Voyage embeddings, OpenAI answers, Qdrant storage, grounded citations, backup/restore,
  authenticated Compose deployment, and deterministic tests;
- server-side document search, status filtering, sorting, pagination, and responsive library UI.
