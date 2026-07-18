"""SQLite lifecycle, connection policy, and forward-only schema migrations."""

from __future__ import annotations

import sqlite3
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock

SCHEMA_VERSION = 2

_MIGRATION_V1 = """
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    content_type TEXT NOT NULL,
    extension TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    status TEXT NOT NULL CHECK (status IN (
        'queued', 'validating', 'extracting', 'chunking', 'embedding', 'indexing',
        'ready', 'failed', 'reindexing', 'deleting', 'deletion_failed', 'deleted'
    )),
    embedding_fingerprint TEXT NOT NULL,
    active_version INTEGER NOT NULL DEFAULT 0 CHECK (active_version >= 0),
    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
    error_code TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_active_content_profile
ON documents (content_sha256, embedding_fingerprint)
WHERE deleted_at IS NULL AND status <> 'deleted';

CREATE INDEX IF NOT EXISTS ix_documents_status_created
ON documents (status, created_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('ingest', 'reindex', 'delete')),
    status TEXT NOT NULL CHECK (status IN (
        'queued', 'running', 'retrying', 'succeeded', 'failed'
    )),
    stage TEXT NOT NULL CHECK (stage IN (
        'queued', 'validating', 'extracting', 'chunking', 'embedding', 'indexing',
        'verifying', 'deleting', 'complete', 'failed'
    )),
    progress REAL NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 1),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    max_attempts INTEGER NOT NULL CHECK (max_attempts >= 1),
    lease_owner TEXT,
    lease_expires_at TEXT,
    error_code TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    finished_at TEXT,
    CHECK (
        (status = 'running' AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
        OR status <> 'running'
    )
);

CREATE INDEX IF NOT EXISTS ix_jobs_queue
ON jobs (status, created_at, id);

CREATE INDEX IF NOT EXISTS ix_jobs_document_created
ON jobs (document_id, created_at DESC, id DESC);

CREATE UNIQUE INDEX IF NOT EXISTS uq_jobs_active_document
ON jobs (document_id)
WHERE status IN ('queued', 'running', 'retrying');

CREATE TABLE IF NOT EXISTS upload_idempotency (
    idempotency_key TEXT PRIMARY KEY,
    request_fingerprint TEXT NOT NULL,
    document_id TEXT NOT NULL REFERENCES documents(id),
    job_id TEXT NOT NULL REFERENCES jobs(id),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

PRAGMA user_version = 1;
COMMIT;
"""

_MIGRATION_V2 = """
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL CHECK (length(title) BETWEEN 1 AND 120),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_conversations_updated
ON conversations (updated_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    client_turn_id TEXT NOT NULL CHECK (length(client_turn_id) BETWEEN 8 AND 128),
    request_fingerprint TEXT NOT NULL CHECK (length(request_fingerprint) = 64),
    reservation_token TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending', 'completed', 'failed')),
    question TEXT NOT NULL CHECK (length(question) BETWEEN 1 AND 4000),
    answer TEXT,
    no_answer INTEGER NOT NULL DEFAULT 0 CHECK (no_answer IN (0, 1)),
    top_k INTEGER CHECK (top_k BETWEEN 1 AND 50),
    document_ids_json TEXT CHECK (
        document_ids_json IS NULL OR json_valid(document_ids_json)
    ),
    request_id TEXT,
    error_code TEXT,
    retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),
    reservation_expires_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (conversation_id, client_turn_id),
    CHECK (
        (
            status = 'pending' AND answer IS NULL
            AND reservation_expires_at IS NOT NULL AND reservation_token IS NOT NULL
        )
        OR (
            status = 'completed' AND answer IS NOT NULL
            AND reservation_expires_at IS NULL AND reservation_token IS NULL
        )
        OR (
            status = 'failed' AND answer IS NULL
            AND reservation_expires_at IS NULL AND reservation_token IS NULL
        )
    )
);

CREATE INDEX IF NOT EXISTS ix_conversation_turns_list
ON conversation_turns (conversation_id, created_at DESC, id DESC);

CREATE INDEX IF NOT EXISTS ix_conversation_turns_reservations
ON conversation_turns (status, reservation_expires_at);

CREATE TABLE IF NOT EXISTS turn_citations (
    turn_id TEXT NOT NULL REFERENCES conversation_turns(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    label TEXT NOT NULL,
    document_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    document_name TEXT NOT NULL,
    page_number INTEGER,
    section TEXT,
    snippet TEXT NOT NULL,
    score REAL,
    PRIMARY KEY (turn_id, ordinal)
);

CREATE INDEX IF NOT EXISTS ix_turn_citations_document
ON turn_citations (document_id, turn_id);

PRAGMA user_version = 2;
COMMIT;
"""


def unicode_casefold(value: object) -> str:
    """Return one deterministic Unicode search key for SQLite and Python callers."""

    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).casefold()


class Database:
    """Create short-lived SQLite connections with one consistent safety policy."""

    def __init__(self, path: Path, *, busy_timeout_ms: int = 5_000) -> None:
        if busy_timeout_ms < 1:
            raise ValueError("busy_timeout_ms must be positive")
        self.path = Path(path)
        self.busy_timeout_ms = busy_timeout_ms
        self._migration_lock = FileLock(
            f"{self.path}.migrate.lock", timeout=max(1.0, busy_timeout_ms / 1000)
        )

    def initialize(self) -> None:
        """Create the database and apply all known forward-only migrations."""

        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._migration_lock, self.connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    "Database uses a newer schema version "
                    f"({version}) than this application supports ({SCHEMA_VERSION})"
                )
            if version < 1:
                connection.executescript(_MIGRATION_V1)
            if version < 2:
                connection.executescript(_MIGRATION_V2)

    def schema_version(self) -> int:
        """Return the persisted schema version without changing the database."""

        with self.connection() as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured connection and always close it afterward."""

        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.create_function(
            "unicode_casefold",
            1,
            unicode_casefold,
            deterministic=True,
        )
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        connection.execute("PRAGMA synchronous = NORMAL")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """Run a rollback-safe transaction on a fresh connection."""

        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()
