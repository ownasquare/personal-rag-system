"""Transactional persistence for conversations, documents, jobs, and worker state."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from typing import Any, cast
from uuid import uuid4

from personal_rag.database import Database
from personal_rag.errors import RagError
from personal_rag.models import (
    ChatHistoryMessage,
    Citation,
    ConversationSummary,
    ConversationTurn,
    ConversationTurnReservation,
    ConversationTurnStatus,
    DocumentPublic,
    DocumentRecord,
    DocumentStatus,
    JobKind,
    JobRecord,
    JobStage,
    JobStatus,
    UploadReceipt,
    utc_now,
)

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_PAGE_SIZE = 200

_DOCUMENT_TRANSITIONS: dict[DocumentStatus, frozenset[DocumentStatus]] = {
    DocumentStatus.QUEUED: frozenset(
        {DocumentStatus.VALIDATING, DocumentStatus.FAILED, DocumentStatus.DELETING}
    ),
    DocumentStatus.VALIDATING: frozenset(
        {DocumentStatus.EXTRACTING, DocumentStatus.FAILED, DocumentStatus.DELETING}
    ),
    DocumentStatus.EXTRACTING: frozenset(
        {DocumentStatus.CHUNKING, DocumentStatus.FAILED, DocumentStatus.DELETING}
    ),
    DocumentStatus.CHUNKING: frozenset(
        {DocumentStatus.EMBEDDING, DocumentStatus.FAILED, DocumentStatus.DELETING}
    ),
    DocumentStatus.EMBEDDING: frozenset(
        {DocumentStatus.INDEXING, DocumentStatus.FAILED, DocumentStatus.DELETING}
    ),
    DocumentStatus.INDEXING: frozenset(
        {DocumentStatus.READY, DocumentStatus.FAILED, DocumentStatus.DELETING}
    ),
    DocumentStatus.READY: frozenset({DocumentStatus.REINDEXING, DocumentStatus.DELETING}),
    DocumentStatus.FAILED: frozenset(
        {DocumentStatus.QUEUED, DocumentStatus.REINDEXING, DocumentStatus.DELETING}
    ),
    DocumentStatus.REINDEXING: frozenset(
        {
            DocumentStatus.VALIDATING,
            DocumentStatus.EXTRACTING,
            DocumentStatus.FAILED,
            DocumentStatus.DELETING,
        }
    ),
    DocumentStatus.DELETING: frozenset({DocumentStatus.DELETED, DocumentStatus.DELETION_FAILED}),
    DocumentStatus.DELETION_FAILED: frozenset({DocumentStatus.DELETING}),
    DocumentStatus.DELETED: frozenset(),
}

_JOB_STAGE_TRANSITIONS: dict[JobStage, frozenset[JobStage]] = {
    JobStage.QUEUED: frozenset({JobStage.VALIDATING, JobStage.DELETING}),
    JobStage.VALIDATING: frozenset({JobStage.EXTRACTING}),
    JobStage.EXTRACTING: frozenset({JobStage.CHUNKING}),
    JobStage.CHUNKING: frozenset({JobStage.EMBEDDING}),
    JobStage.EMBEDDING: frozenset({JobStage.INDEXING}),
    JobStage.INDEXING: frozenset({JobStage.VERIFYING}),
    JobStage.VERIFYING: frozenset(),
    JobStage.DELETING: frozenset(),
    JobStage.COMPLETE: frozenset(),
    JobStage.FAILED: frozenset(),
}

_STAGE_DOCUMENT_STATUS = {
    JobStage.VALIDATING: DocumentStatus.VALIDATING,
    JobStage.EXTRACTING: DocumentStatus.EXTRACTING,
    JobStage.CHUNKING: DocumentStatus.CHUNKING,
    JobStage.EMBEDDING: DocumentStatus.EMBEDDING,
    JobStage.INDEXING: DocumentStatus.INDEXING,
    JobStage.DELETING: DocumentStatus.DELETING,
}


class Repository:
    """Expose small, explicit transactions instead of leaking SQL to services."""

    def __init__(
        self,
        database: Database,
        *,
        lease_seconds: int = 300,
        max_attempts: int = 3,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.database = database
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self._clock = clock

    def initialize(self) -> None:
        self.database.initialize()

    def create_conversation(
        self,
        title: str | None = None,
        *,
        conversation_id: str | None = None,
    ) -> ConversationSummary:
        """Create an empty durable conversation."""

        normalized_title = (
            "New conversation" if title is None else self._validated_conversation_title(title)
        )
        identifier = self._validated_identifier(conversation_id or uuid4().hex, "conversation_id")
        now_text = self._serialize_datetime(self._now())
        try:
            with self.database.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO conversations (id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (identifier, normalized_title, now_text, now_text),
                )
                return self._conversation_summary_from_row(
                    connection,
                    connection.execute(
                        "SELECT * FROM conversations WHERE id = ?", (identifier,)
                    ).fetchone(),
                )
        except sqlite3.IntegrityError as exc:
            raise RagError(
                "persistence_conflict",
                "The conversation identifier is already in use",
                status_code=409,
            ) from exc

    def get_conversation(self, conversation_id: str) -> ConversationSummary | None:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if row is None:
                return None
            return self._conversation_summary_from_row(connection, row)

    def list_conversations(self, *, limit: int = 50, offset: int = 0) -> list[ConversationSummary]:
        self._validate_pagination(limit, offset)
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM conversations
                ORDER BY updated_at DESC, rowid DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
            return [self._conversation_summary_from_row(connection, row) for row in rows]

    def count_conversations(self) -> int:
        with self.database.connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM conversations").fetchone()
        return int(row["total"])

    def delete_conversation(self, conversation_id: str) -> bool:
        """Hard-delete a conversation and all retained answer/source content."""

        with self.database.transaction(immediate=True) as connection:
            deleted = connection.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
        return deleted.rowcount == 1

    def get_conversation_turn(self, turn_id: str) -> ConversationTurn | None:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM conversation_turns WHERE id = ?", (turn_id,)
            ).fetchone()
            return self._conversation_turn_from_row(connection, row) if row is not None else None

    def list_conversation_turns(
        self,
        conversation_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        include_incomplete: bool = False,
    ) -> list[ConversationTurn]:
        self._validate_pagination(limit, offset)
        query = (
            """
            SELECT * FROM conversation_turns
            WHERE conversation_id = ?
            ORDER BY created_at, rowid
            LIMIT ? OFFSET ?
            """
            if include_incomplete
            else """
            SELECT * FROM conversation_turns
            WHERE conversation_id = ? AND status = 'completed'
            ORDER BY created_at, rowid
            LIMIT ? OFFSET ?
            """
        )
        with self.database.connection() as connection:
            rows = connection.execute(
                query,
                (conversation_id, limit, offset),
            ).fetchall()
            return [self._conversation_turn_from_row(connection, row) for row in rows]

    def count_conversation_turns(
        self, conversation_id: str, *, include_incomplete: bool = False
    ) -> int:
        query = (
            """
            SELECT COUNT(*) AS total
            FROM conversation_turns
            WHERE conversation_id = ?
            """
            if include_incomplete
            else """
            SELECT COUNT(*) AS total
            FROM conversation_turns
            WHERE conversation_id = ? AND status = 'completed'
            """
        )
        with self.database.connection() as connection:
            row = connection.execute(
                query,
                (conversation_id,),
            ).fetchone()
        return int(row["total"])

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

        normalized_conversation_id = self._validated_identifier(conversation_id, "conversation_id")
        normalized_client_turn_id = self._validated_client_turn_id(client_turn_id)
        normalized_question = self._validated_conversation_text(question, "question", 4_000)
        normalized_top_k = self._validated_top_k(top_k)
        normalized_document_ids = self._validated_document_ids(document_ids)
        normalized_fingerprint = request_fingerprint.lower()
        if _HASH_PATTERN.fullmatch(normalized_fingerprint) is None:
            raise ValueError("request_fingerprint must be a lowercase SHA-256 hex digest")
        if reservation_seconds < 1 or reservation_seconds > 3_600:
            raise ValueError("reservation_seconds must be between 1 and 3600")

        now = self._now()
        now_text = self._serialize_datetime(now)
        expires_text = self._serialize_datetime(now + timedelta(seconds=reservation_seconds))
        with self.database.transaction(immediate=True) as connection:
            conversation = connection.execute(
                "SELECT id FROM conversations WHERE id = ?",
                (normalized_conversation_id,),
            ).fetchone()
            if conversation is None:
                raise RagError(
                    "conversation_not_found",
                    "The requested conversation does not exist.",
                    status_code=404,
                )

            existing = connection.execute(
                """
                SELECT * FROM conversation_turns
                WHERE conversation_id = ? AND client_turn_id = ?
                """,
                (normalized_conversation_id, normalized_client_turn_id),
            ).fetchone()
            if existing is not None:
                if existing["request_fingerprint"] != normalized_fingerprint:
                    raise RagError(
                        "idempotency_conflict",
                        "This client turn identifier was already used for a different question.",
                        status_code=409,
                    )
                turn = self._conversation_turn_from_row(connection, existing)
                if turn.status is ConversationTurnStatus.COMPLETED or (
                    turn.status is ConversationTurnStatus.FAILED and not turn.retryable
                ):
                    return ConversationTurnReservation(turn=turn, created=False, cached_turn=turn)
                if turn.status is ConversationTurnStatus.FAILED and turn.retryable:
                    reservation_token = uuid4().hex
                    connection.execute(
                        """
                        UPDATE conversation_turns
                        SET status = 'pending', error_code = NULL, retryable = 0,
                            reservation_expires_at = ?, reservation_token = ?, updated_at = ?
                        WHERE id = ? AND status = 'failed'
                        """,
                        (expires_text, reservation_token, now_text, existing["id"]),
                    )
                    retry = connection.execute(
                        "SELECT * FROM conversation_turns WHERE id = ?",
                        (existing["id"],),
                    ).fetchone()
                    return ConversationTurnReservation(
                        turn=self._conversation_turn_from_row(connection, retry),
                        created=True,
                        reservation_token=reservation_token,
                    )
                expiry = existing["reservation_expires_at"]
                if expiry is not None and self._parse_datetime(expiry) > now:
                    raise RagError(
                        "conversation_turn_in_progress",
                        "This question is already being answered.",
                        status_code=409,
                        retryable=True,
                    )
                reservation_token = uuid4().hex
                connection.execute(
                    """
                    UPDATE conversation_turns
                    SET reservation_expires_at = ?, reservation_token = ?, updated_at = ?
                    WHERE id = ? AND status = 'pending'
                    """,
                    (expires_text, reservation_token, now_text, existing["id"]),
                )
                recovered = connection.execute(
                    "SELECT * FROM conversation_turns WHERE id = ?",
                    (existing["id"],),
                ).fetchone()
                return ConversationTurnReservation(
                    turn=self._conversation_turn_from_row(connection, recovered),
                    created=True,
                    reservation_token=reservation_token,
                )

            turn_id = uuid4().hex
            reservation_token = uuid4().hex
            connection.execute(
                """
                INSERT INTO conversation_turns (
                    id, conversation_id, client_turn_id, request_fingerprint,
                    reservation_token, status, question, top_k, document_ids_json,
                    reservation_expires_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    normalized_conversation_id,
                    normalized_client_turn_id,
                    normalized_fingerprint,
                    reservation_token,
                    ConversationTurnStatus.PENDING.value,
                    normalized_question,
                    normalized_top_k,
                    (
                        json.dumps(normalized_document_ids, separators=(",", ":"))
                        if normalized_document_ids is not None
                        else None
                    ),
                    expires_text,
                    now_text,
                    now_text,
                ),
            )
            row = connection.execute(
                "SELECT * FROM conversation_turns WHERE id = ?", (turn_id,)
            ).fetchone()
            return ConversationTurnReservation(
                turn=self._conversation_turn_from_row(connection, row),
                created=True,
                reservation_token=reservation_token,
            )

    def renew_conversation_turn_reservation(
        self,
        turn_id: str,
        *,
        reservation_token: str,
        reservation_seconds: int = 120,
    ) -> bool:
        """Extend only the currently owned paid-call reservation."""

        normalized_turn_id = self._validated_identifier(turn_id, "turn_id")
        normalized_token = self._validated_identifier(reservation_token, "reservation_token")
        if reservation_seconds < 1 or reservation_seconds > 3_600:
            raise ValueError("reservation_seconds must be between 1 and 3600")
        now = self._now()
        with self.database.transaction(immediate=True) as connection:
            renewed = connection.execute(
                """
                UPDATE conversation_turns
                SET reservation_expires_at = ?, updated_at = ?
                WHERE id = ? AND status = 'pending' AND reservation_token = ?
                """,
                (
                    self._serialize_datetime(now + timedelta(seconds=reservation_seconds)),
                    self._serialize_datetime(now),
                    normalized_turn_id,
                    normalized_token,
                ),
            )
        return renewed.rowcount == 1

    def complete_conversation_turn(
        self,
        turn_id: str,
        *,
        reservation_token: str,
        answer: str,
        citations: Sequence[Citation],
        no_answer: bool,
        request_id: str | None,
    ) -> ConversationTurn:
        """Persist one grounded result and advance conversation recency atomically."""

        normalized_answer = self._validated_conversation_text(answer, "answer", 12_000)
        normalized_citations = [
            self._validated_citation(Citation.model_validate(citation)) for citation in citations
        ]
        if len(normalized_citations) > 50:
            raise ValueError("at most 50 citations may be persisted")
        normalized_request_id = (
            None if request_id is None else self._validated_identifier(request_id, "request_id")
        )
        normalized_token = self._validated_identifier(reservation_token, "reservation_token")
        now = self._now()
        now_text = self._serialize_datetime(now)
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM conversation_turns WHERE id = ?", (turn_id,)
            ).fetchone()
            if row is None:
                raise RagError(
                    "conversation_turn_not_found",
                    "The requested conversation turn does not exist.",
                    status_code=404,
                )
            status = ConversationTurnStatus(row["status"])
            if status is ConversationTurnStatus.COMPLETED:
                return self._conversation_turn_from_row(connection, row)
            if status is not ConversationTurnStatus.PENDING:
                raise RagError(
                    "invalid_conversation_turn_transition",
                    "Only a pending conversation turn can be completed.",
                    status_code=409,
                )
            if row["reservation_token"] != normalized_token:
                raise RagError(
                    "conversation_turn_lease_lost",
                    "This question is being completed by a newer request.",
                    status_code=409,
                    retryable=True,
                )

            cited_document_ids = list(
                dict.fromkeys(citation.document_id for citation in normalized_citations)
            )
            for document_id in cited_document_ids:
                document = connection.execute(
                    "SELECT status FROM documents WHERE id = ?",
                    (document_id,),
                ).fetchone()
                if (
                    document is None
                    or DocumentStatus(document["status"]) is not DocumentStatus.READY
                ):
                    raise RagError(
                        "source_changed",
                        "A source changed while the answer was being prepared. Please try again.",
                        status_code=409,
                        retryable=True,
                    )

            for ordinal, citation in enumerate(normalized_citations):
                connection.execute(
                    """
                    INSERT INTO turn_citations (
                        turn_id, ordinal, label, document_id, chunk_id,
                        document_name, page_number, section, snippet, score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        turn_id,
                        ordinal,
                        citation.label,
                        citation.document_id,
                        citation.chunk_id,
                        citation.document_name,
                        citation.page_number,
                        citation.section,
                        citation.snippet,
                        citation.score,
                    ),
                )
            connection.execute(
                """
                UPDATE conversation_turns
                SET status = ?, answer = ?, no_answer = ?, request_id = ?,
                    error_code = NULL, retryable = 0, reservation_expires_at = NULL,
                    reservation_token = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    ConversationTurnStatus.COMPLETED.value,
                    normalized_answer,
                    int(no_answer),
                    normalized_request_id,
                    now_text,
                    turn_id,
                ),
            )
            completed_before = connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM conversation_turns
                WHERE conversation_id = ? AND status = 'completed' AND id <> ?
                """,
                (row["conversation_id"], turn_id),
            ).fetchone()
            conversation = connection.execute(
                "SELECT title FROM conversations WHERE id = ?",
                (row["conversation_id"],),
            ).fetchone()
            title = str(conversation["title"])
            if int(completed_before["total"]) == 0 and title == "New conversation":
                title = self._derived_conversation_title(str(row["question"]))
            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, now_text, row["conversation_id"]),
            )
            completed = connection.execute(
                "SELECT * FROM conversation_turns WHERE id = ?", (turn_id,)
            ).fetchone()
            return self._conversation_turn_from_row(connection, completed)

    def fail_conversation_turn(
        self,
        turn_id: str,
        *,
        reservation_token: str,
        error_code: str,
        retryable: bool,
    ) -> ConversationTurn:
        """Store only safe failure metadata; never persist provider exception text."""

        normalized_code = self._validated_error_text(error_code, "error_code", 100)
        normalized_token = self._validated_identifier(reservation_token, "reservation_token")
        now_text = self._serialize_datetime(self._now())
        with self.database.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM conversation_turns WHERE id = ?", (turn_id,)
            ).fetchone()
            if row is None:
                raise RagError(
                    "conversation_turn_not_found",
                    "The requested conversation turn does not exist.",
                    status_code=404,
                )
            status = ConversationTurnStatus(row["status"])
            if status is ConversationTurnStatus.COMPLETED:
                return self._conversation_turn_from_row(connection, row)
            if (
                status is ConversationTurnStatus.PENDING
                and row["reservation_token"] == normalized_token
            ):
                connection.execute(
                    """
                    UPDATE conversation_turns
                    SET status = ?, error_code = ?, retryable = ?,
                        reservation_expires_at = NULL, reservation_token = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        ConversationTurnStatus.FAILED.value,
                        normalized_code,
                        int(retryable),
                        now_text,
                        turn_id,
                    ),
                )
            failed = connection.execute(
                "SELECT * FROM conversation_turns WHERE id = ?", (turn_id,)
            ).fetchone()
            return self._conversation_turn_from_row(connection, failed)

    def conversation_history(self, conversation_id: str, *, limit: int) -> list[ChatHistoryMessage]:
        """Reconstruct a bounded message window only from completed turns."""

        if limit < 0 or limit > 100:
            raise ValueError("limit must be between 0 and 100")
        if limit == 0:
            return []
        turn_limit = limit // 2
        if turn_limit == 0:
            return []
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT question, answer
                FROM conversation_turns
                WHERE conversation_id = ? AND status = 'completed'
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (conversation_id, turn_limit),
            ).fetchall()
        messages: list[ChatHistoryMessage] = []
        for row in reversed(rows):
            messages.append(ChatHistoryMessage(role="user", content=str(row["question"])))
            messages.append(ChatHistoryMessage(role="assistant", content=str(row["answer"])))
        return messages

    def purge_conversation_turns_for_document(
        self, connection: sqlite3.Connection, document_id: str
    ) -> int:
        """Delete whole turns that retain content from a document being deleted."""

        affected = connection.execute(
            """
            SELECT DISTINCT conversation_turns.conversation_id
            FROM conversation_turns
            JOIN turn_citations ON turn_citations.turn_id = conversation_turns.id
            WHERE turn_citations.document_id = ?
            """,
            (document_id,),
        ).fetchall()
        deleted = connection.execute(
            """
            DELETE FROM conversation_turns
            WHERE id IN (
                SELECT turn_id FROM turn_citations WHERE document_id = ?
            )
            """,
            (document_id,),
        )
        now_text = self._serialize_datetime(self._now())
        for row in affected:
            conversation_id = str(row["conversation_id"])
            remaining = connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM conversation_turns
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            if int(remaining["total"]) == 0:
                connection.execute(
                    "DELETE FROM conversations WHERE id = ?",
                    (conversation_id,),
                )
                continue
            first_completed = connection.execute(
                """
                SELECT question
                FROM conversation_turns
                WHERE conversation_id = ? AND status = 'completed'
                ORDER BY created_at, rowid
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            title = (
                self._derived_conversation_title(str(first_completed["question"]))
                if first_completed is not None
                else "New conversation"
            )
            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, now_text, conversation_id),
            )
        return deleted.rowcount

    def create_document_with_job(
        self,
        *,
        display_name: str,
        stored_path: str,
        content_type: str,
        extension: str,
        content_sha256: str,
        size_bytes: int,
        embedding_fingerprint: str,
        idempotency_key: str | None = None,
        document_id: str | None = None,
        job_id: str | None = None,
    ) -> UploadReceipt:
        """Atomically register one document and its initial ingestion job."""

        normalized = self._validated_upload_fields(
            display_name=display_name,
            stored_path=stored_path,
            content_type=content_type,
            extension=extension,
            content_sha256=content_sha256,
            size_bytes=size_bytes,
            embedding_fingerprint=embedding_fingerprint,
            idempotency_key=idempotency_key,
        )
        request_fingerprint = self._upload_request_fingerprint(normalized)
        now = self._now()
        now_text = self._serialize_datetime(now)

        with self.database.transaction(immediate=True) as connection:
            if idempotency_key is not None:
                previous = connection.execute(
                    """
                    SELECT request_fingerprint, document_id, job_id
                    FROM upload_idempotency
                    WHERE idempotency_key = ?
                    """,
                    (idempotency_key,),
                ).fetchone()
                if previous is not None:
                    if previous["request_fingerprint"] != request_fingerprint:
                        raise RagError(
                            "idempotency_conflict",
                            "This idempotency key was already used for a different upload",
                            status_code=409,
                        )
                    return self._upload_receipt(
                        connection,
                        previous["document_id"],
                        previous["job_id"],
                        duplicate=True,
                    )

            existing = connection.execute(
                """
                SELECT id
                FROM documents
                WHERE content_sha256 = ?
                  AND embedding_fingerprint = ?
                  AND deleted_at IS NULL
                  AND status <> 'deleted'
                LIMIT 1
                """,
                (normalized["content_sha256"], normalized["embedding_fingerprint"]),
            ).fetchone()
            if existing is not None:
                latest_job = connection.execute(
                    """
                    SELECT id FROM jobs
                    WHERE document_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """,
                    (existing["id"],),
                ).fetchone()
                if latest_job is None:
                    raise RuntimeError("A persisted document has no job history")
                if idempotency_key is not None:
                    connection.execute(
                        """
                        INSERT INTO upload_idempotency (
                            idempotency_key, request_fingerprint, document_id, job_id, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            idempotency_key,
                            request_fingerprint,
                            existing["id"],
                            latest_job["id"],
                            now_text,
                        ),
                    )
                return self._upload_receipt(
                    connection,
                    existing["id"],
                    latest_job["id"],
                    duplicate=True,
                )

            new_document_id = self._validated_identifier(document_id or uuid4().hex, "document_id")
            new_job_id = self._validated_identifier(job_id or uuid4().hex, "job_id")
            try:
                connection.execute(
                    """
                    INSERT INTO documents (
                        id, display_name, stored_path, content_type, extension,
                        content_sha256, size_bytes, status, embedding_fingerprint,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_document_id,
                        normalized["display_name"],
                        normalized["stored_path"],
                        normalized["content_type"],
                        normalized["extension"],
                        normalized["content_sha256"],
                        normalized["size_bytes"],
                        DocumentStatus.QUEUED.value,
                        normalized["embedding_fingerprint"],
                        now_text,
                        now_text,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO jobs (
                        id, document_id, kind, status, stage, max_attempts,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_job_id,
                        new_document_id,
                        JobKind.INGEST.value,
                        JobStatus.QUEUED.value,
                        JobStage.QUEUED.value,
                        self.max_attempts,
                        now_text,
                        now_text,
                    ),
                )
                if idempotency_key is not None:
                    connection.execute(
                        """
                        INSERT INTO upload_idempotency (
                            idempotency_key, request_fingerprint, document_id, job_id, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            idempotency_key,
                            request_fingerprint,
                            new_document_id,
                            new_job_id,
                            now_text,
                        ),
                    )
            except sqlite3.IntegrityError as exc:
                raise RagError(
                    "persistence_conflict",
                    "The document or job identifier is already in use",
                    status_code=409,
                ) from exc

            return self._upload_receipt(connection, new_document_id, new_job_id, duplicate=False)

    def get_document(
        self, document_id: str, *, include_deleted: bool = False
    ) -> DocumentRecord | None:
        with self.database.connection() as connection:
            query = (
                "SELECT * FROM documents WHERE id = ?"
                if include_deleted
                else "SELECT * FROM documents WHERE id = ? AND status <> 'deleted'"
            )
            row = connection.execute(query, (document_id,)).fetchone()
        return self._document_from_row(row) if row is not None else None

    def list_documents(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: DocumentStatus | None = None,
        include_deleted: bool = False,
    ) -> list[DocumentRecord]:
        self._validate_pagination(limit, offset)
        parameters: list[Any]
        if status is None:
            query = (
                "SELECT * FROM documents ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
                if include_deleted
                else "SELECT * FROM documents WHERE status <> 'deleted' "
                "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
            )
            parameters = [limit, offset]
        else:
            normalized_status = DocumentStatus(status).value
            query = (
                "SELECT * FROM documents WHERE status = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
                if include_deleted
                else "SELECT * FROM documents WHERE status = ? AND status <> 'deleted' "
                "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
            )
            parameters = [normalized_status, limit, offset]
        with self.database.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._document_from_row(row) for row in rows]

    def count_documents(
        self,
        *,
        status: DocumentStatus | None = None,
        include_deleted: bool = False,
    ) -> int:
        parameters: list[Any]
        if status is None:
            query = (
                "SELECT COUNT(*) AS total FROM documents"
                if include_deleted
                else "SELECT COUNT(*) AS total FROM documents WHERE status <> 'deleted'"
            )
            parameters = []
        else:
            query = (
                "SELECT COUNT(*) AS total FROM documents WHERE status = ?"
                if include_deleted
                else "SELECT COUNT(*) AS total FROM documents "
                "WHERE status = ? AND status <> 'deleted'"
            )
            parameters = [DocumentStatus(status).value]
        with self.database.connection() as connection:
            row = connection.execute(query, parameters).fetchone()
        return int(row["total"])

    def get_ready_document_versions(
        self, document_ids: Sequence[str] | None = None
    ) -> dict[str, int]:
        """Return the active vector version for documents currently safe to retrieve."""

        if document_ids is None:
            query = (
                "SELECT id, active_version FROM documents "
                "WHERE status = 'ready' AND deleted_at IS NULL"
            )
            parameters: list[str] = []
        else:
            normalized = tuple(dict.fromkeys(document_ids))
            if len(normalized) > 100:
                raise ValueError("at most 100 document identifiers may be checked")
            if not normalized:
                return {}
            query = (
                "SELECT id, active_version FROM documents "
                "WHERE status = 'ready' AND deleted_at IS NULL "
                "AND id IN (SELECT value FROM json_each(?))"
            )
            parameters = [json.dumps(normalized)]
        with self.database.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return {str(row["id"]): int(row["active_version"]) for row in rows}

    def get_job(self, job_id: str) -> JobRecord | None:
        with self.database.connection() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._job_from_row(row) if row is not None else None

    def get_latest_job(self, document_id: str) -> JobRecord | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE document_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (document_id,),
            ).fetchone()
        return self._job_from_row(row) if row is not None else None

    def list_jobs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: JobStatus | None = None,
        document_id: str | None = None,
    ) -> list[JobRecord]:
        self._validate_pagination(limit, offset)
        parameters: list[Any]
        if status is None and document_id is None:
            query = "SELECT * FROM jobs ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
            parameters = [limit, offset]
        elif status is not None and document_id is None:
            query = (
                "SELECT * FROM jobs WHERE status = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
            )
            parameters = [JobStatus(status).value, limit, offset]
        elif status is None and document_id is not None:
            query = (
                "SELECT * FROM jobs WHERE document_id = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
            )
            parameters = [document_id, limit, offset]
        else:
            query = (
                "SELECT * FROM jobs WHERE status = ? AND document_id = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
            )
            parameters = [
                cast(JobStatus, status).value,
                cast(str, document_id),
                limit,
                offset,
            ]
        with self.database.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._job_from_row(row) for row in rows]

    def count_jobs(
        self,
        *,
        status: JobStatus | None = None,
        document_id: str | None = None,
    ) -> int:
        parameters: list[Any]
        if status is None and document_id is None:
            query = "SELECT COUNT(*) AS total FROM jobs"
            parameters = []
        elif status is not None and document_id is None:
            query = "SELECT COUNT(*) AS total FROM jobs WHERE status = ?"
            parameters = [JobStatus(status).value]
        elif status is None and document_id is not None:
            query = "SELECT COUNT(*) AS total FROM jobs WHERE document_id = ?"
            parameters = [document_id]
        else:
            query = "SELECT COUNT(*) AS total FROM jobs WHERE status = ? AND document_id = ?"
            parameters = [cast(JobStatus, status).value, cast(str, document_id)]
        with self.database.connection() as connection:
            row = connection.execute(query, parameters).fetchone()
        return int(row["total"])

    def get_statistics(self) -> dict[str, int]:
        """Return the status-page counters from one consistent read transaction."""

        with self.database.transaction() as connection:
            documents = connection.execute(
                """
                SELECT
                    COUNT(*) AS document_count,
                    COALESCE(SUM(CASE WHEN status = 'ready' THEN 1 ELSE 0 END), 0)
                        AS ready_document_count,
                    COALESCE(SUM(chunk_count), 0) AS chunk_count
                FROM documents
                WHERE status <> 'deleted'
                """
            ).fetchone()
            jobs = connection.execute(
                """
                SELECT COUNT(*) AS queued_job_count
                FROM jobs
                WHERE status IN ('queued', 'retrying')
                """
            ).fetchone()
        return {
            "document_count": int(documents["document_count"]),
            "ready_document_count": int(documents["ready_document_count"]),
            "chunk_count": int(documents["chunk_count"]),
            "queued_job_count": int(jobs["queued_job_count"]),
        }

    def enqueue_job(self, document_id: str, kind: JobKind) -> JobRecord:
        kind = JobKind(kind)
        if kind is JobKind.REINDEX:
            return self.request_reindex(document_id)
        if kind is JobKind.DELETE:
            return self.request_delete(document_id)

        now = self._now()
        with self.database.transaction(immediate=True) as connection:
            document = self._require_document_row(connection, document_id)
            existing = self._active_job_row(connection, document_id)
            if existing is not None:
                if JobKind(existing["kind"]) is kind:
                    return self._job_from_row(existing)
                self._raise_active_job_conflict()
            status = DocumentStatus(document["status"])
            if status is not DocumentStatus.FAILED:
                raise RagError(
                    "invalid_document_transition",
                    "Only a failed document can be queued for ingestion again",
                    status_code=409,
                )
            self._set_document_state(connection, document, DocumentStatus.QUEUED, now)
            return self._insert_job(connection, document_id, kind, now)

    def request_reindex(self, document_id: str) -> JobRecord:
        now = self._now()
        with self.database.transaction(immediate=True) as connection:
            document = self._require_document_row(connection, document_id)
            existing = self._active_job_row(connection, document_id)
            if existing is not None:
                if JobKind(existing["kind"]) is JobKind.REINDEX:
                    return self._job_from_row(existing)
                self._raise_active_job_conflict()
            current = DocumentStatus(document["status"])
            if current not in {DocumentStatus.READY, DocumentStatus.FAILED}:
                raise RagError(
                    "invalid_document_transition",
                    f"A document in {current.value!r} state cannot be reindexed",
                    status_code=409,
                )
            self._set_document_state(
                connection, document, DocumentStatus.REINDEXING, now, clear_error=True
            )
            return self._insert_job(connection, document_id, JobKind.REINDEX, now)

    def request_delete(self, document_id: str) -> JobRecord:
        now = self._now()
        now_text = self._serialize_datetime(now)
        with self.database.transaction(immediate=True) as connection:
            self._reclaim_expired_leases(connection, now)
            document = self._require_document_row(connection, document_id, include_deleted=True)
            current = DocumentStatus(document["status"])
            if current is DocumentStatus.DELETED:
                latest = connection.execute(
                    """
                    SELECT * FROM jobs
                    WHERE document_id = ? AND kind = 'delete'
                    ORDER BY created_at DESC, id DESC LIMIT 1
                    """,
                    (document_id,),
                ).fetchone()
                if latest is not None:
                    return self._job_from_row(latest)
                raise RagError(
                    "document_deleted",
                    "The document is already deleted",
                    status_code=410,
                )

            active = self._active_job_row(connection, document_id)
            if active is not None and JobKind(active["kind"]) is JobKind.DELETE:
                return self._job_from_row(active)
            if active is not None:
                if JobStatus(active["status"]) is JobStatus.RUNNING:
                    raise RagError(
                        "document_busy",
                        "The document is currently processing; retry deletion when that "
                        "job finishes.",
                        status_code=409,
                        retryable=True,
                    )
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = ?, stage = ?, lease_owner = NULL,
                        lease_expires_at = NULL, error_code = ?, error_message = ?,
                        updated_at = ?, finished_at = ?
                    WHERE id = ?
                    """,
                    (
                        JobStatus.FAILED.value,
                        JobStage.FAILED.value,
                        "deletion_requested",
                        "The job was superseded by a document deletion request",
                        now_text,
                        now_text,
                        active["id"],
                    ),
                )
            if current is not DocumentStatus.DELETING:
                self._set_document_state(
                    connection, document, DocumentStatus.DELETING, now, clear_error=True
                )
            return self._insert_job(connection, document_id, JobKind.DELETE, now)

    def lease_next_job(
        self,
        worker_id: str,
        *,
        lease_seconds: int | None = None,
        kinds: Sequence[JobKind] | None = None,
    ) -> JobRecord | None:
        worker_id = self._validated_worker_id(worker_id)
        duration = self._validated_lease_seconds(lease_seconds)
        normalized_kinds = (
            tuple(dict.fromkeys(JobKind(kind) for kind in kinds)) if kinds is not None else None
        )
        if not normalized_kinds and kinds is not None:
            return None
        now = self._now()
        now_text = self._serialize_datetime(now)
        expires_text = self._serialize_datetime(now + timedelta(seconds=duration))

        with self.database.transaction(immediate=True) as connection:
            self._reclaim_expired_leases(connection, now)
            parameters: list[Any]
            query_base = (
                "SELECT * FROM jobs WHERE status IN ('queued', 'retrying') "
                "AND attempts < max_attempts"
            )
            if normalized_kinds is None:
                query = query_base + " ORDER BY created_at, id LIMIT 1"
                parameters = []
            elif len(normalized_kinds) == 1:
                query = query_base + " AND kind IN (?) ORDER BY created_at, id LIMIT 1"
                parameters = [normalized_kinds[0].value]
            elif len(normalized_kinds) == 2:
                query = query_base + " AND kind IN (?, ?) ORDER BY created_at, id LIMIT 1"
                parameters = [kind.value for kind in normalized_kinds]
            else:
                query = query_base + " AND kind IN (?, ?, ?) ORDER BY created_at, id LIMIT 1"
                parameters = [kind.value for kind in normalized_kinds]
            row = connection.execute(query, parameters).fetchone()
            if row is None:
                return None
            updated = connection.execute(
                """
                UPDATE jobs
                SET status = ?, attempts = attempts + 1, lease_owner = ?,
                    lease_expires_at = ?, error_code = NULL, error_message = NULL,
                    updated_at = ?, finished_at = NULL
                WHERE id = ? AND status IN ('queued', 'retrying')
                """,
                (
                    JobStatus.RUNNING.value,
                    worker_id,
                    expires_text,
                    now_text,
                    row["id"],
                ),
            )
            if updated.rowcount != 1:
                return None
            leased = connection.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
            return self._job_from_row(leased)

    def heartbeat_job(
        self,
        job_id: str,
        worker_id: str,
        *,
        lease_seconds: int | None = None,
    ) -> JobRecord:
        duration = self._validated_lease_seconds(lease_seconds)
        now = self._now()
        expires = now + timedelta(seconds=duration)
        with self.database.transaction(immediate=True) as connection:
            self._require_active_lease(connection, job_id, worker_id, now)
            connection.execute(
                "UPDATE jobs SET lease_expires_at = ?, updated_at = ? WHERE id = ?",
                (
                    self._serialize_datetime(expires),
                    self._serialize_datetime(now),
                    job_id,
                ),
            )
            return self._job_from_row(
                connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            )

    def update_job_stage(
        self,
        job_id: str,
        worker_id: str,
        stage: JobStage,
        *,
        progress: float,
        document_status: DocumentStatus | None = None,
    ) -> JobRecord:
        stage = JobStage(stage)
        if not 0 <= progress <= 1:
            raise ValueError("progress must be between 0 and 1")
        now = self._now()
        with self.database.transaction(immediate=True) as connection:
            job = self._require_active_lease(connection, job_id, worker_id, now)
            current_stage = JobStage(job["stage"])
            kind = JobKind(job["kind"])
            if stage is not current_stage and stage not in _JOB_STAGE_TRANSITIONS[current_stage]:
                raise RagError(
                    "invalid_job_transition",
                    f"Job stage cannot move from {current_stage.value!r} to {stage.value!r}",
                    status_code=409,
                )
            if progress < float(job["progress"]):
                raise RagError(
                    "invalid_job_transition",
                    "Job progress cannot move backwards",
                    status_code=409,
                )
            if kind is JobKind.DELETE and stage is not JobStage.DELETING:
                raise RagError(
                    "invalid_job_transition",
                    "Deletion jobs can only enter the deleting stage",
                    status_code=409,
                )
            if kind is not JobKind.DELETE and stage is JobStage.DELETING:
                raise RagError(
                    "invalid_job_transition",
                    "Ingestion jobs cannot enter the deleting stage",
                    status_code=409,
                )
            if document_status is not None:
                document_status = DocumentStatus(document_status)
                expected = _STAGE_DOCUMENT_STATUS.get(stage)
                if expected is not None and document_status is not expected:
                    raise RagError(
                        "invalid_document_transition",
                        f"Stage {stage.value!r} requires document status {expected.value!r}",
                        status_code=409,
                    )
                document = self._require_document_row(connection, job["document_id"])
                self._set_document_state(connection, document, document_status, now)
            connection.execute(
                "UPDATE jobs SET stage = ?, progress = ?, updated_at = ? WHERE id = ?",
                (stage.value, progress, self._serialize_datetime(now), job_id),
            )
            return self._job_from_row(
                connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            )

    def complete_job(
        self,
        job_id: str,
        worker_id: str,
        *,
        chunk_count: int | None = None,
        active_version: int | None = None,
        embedding_fingerprint: str | None = None,
    ) -> JobRecord:
        now = self._now()
        with self.database.transaction(immediate=True) as connection:
            job = self._require_active_lease(connection, job_id, worker_id, now)
            kind = JobKind(job["kind"])
            stage = JobStage(job["stage"])
            required_stage = JobStage.DELETING if kind is JobKind.DELETE else JobStage.VERIFYING
            if stage is not required_stage:
                raise RagError(
                    "invalid_job_transition",
                    f"A {kind.value} job can only complete from {required_stage.value!r}",
                    status_code=409,
                )
            document = self._require_document_row(
                connection, job["document_id"], include_deleted=True
            )
            if kind is JobKind.DELETE:
                self.purge_conversation_turns_for_document(connection, str(job["document_id"]))
                self._set_document_state(
                    connection, document, DocumentStatus.DELETED, now, clear_error=True
                )
            else:
                resolved_chunks = (
                    int(document["chunk_count"]) if chunk_count is None else chunk_count
                )
                resolved_version = (
                    int(document["active_version"]) + 1
                    if active_version is None
                    else active_version
                )
                if resolved_chunks < 0 or resolved_version < 1:
                    raise ValueError("chunk_count must be non-negative and active_version positive")
                self._set_document_state(
                    connection,
                    document,
                    DocumentStatus.READY,
                    now,
                    chunk_count=resolved_chunks,
                    active_version=resolved_version,
                    embedding_fingerprint=embedding_fingerprint,
                    clear_error=True,
                )
            now_text = self._serialize_datetime(now)
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, stage = ?, progress = 1, lease_owner = NULL,
                    lease_expires_at = NULL, error_code = NULL, error_message = NULL,
                    updated_at = ?, finished_at = ?
                WHERE id = ?
                """,
                (
                    JobStatus.SUCCEEDED.value,
                    JobStage.COMPLETE.value,
                    now_text,
                    now_text,
                    job_id,
                ),
            )
            return self._job_from_row(
                connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            )

    def fail_job(
        self,
        job_id: str,
        worker_id: str,
        *,
        error_code: str,
        error_message: str,
        retryable: bool,
    ) -> JobRecord:
        error_code = self._validated_error_text(error_code, "error_code", 100)
        error_message = self._validated_error_text(error_message, "error_message", 2_000)
        now = self._now()
        now_text = self._serialize_datetime(now)
        with self.database.transaction(immediate=True) as connection:
            job = self._require_active_lease(connection, job_id, worker_id, now)
            kind = JobKind(job["kind"])
            can_retry = retryable and int(job["attempts"]) < int(job["max_attempts"])
            document = self._require_document_row(
                connection, job["document_id"], include_deleted=True
            )
            if can_retry:
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = ?, stage = ?, progress = 0, lease_owner = NULL,
                        lease_expires_at = NULL, error_code = ?, error_message = ?,
                        updated_at = ?, finished_at = NULL
                    WHERE id = ?
                    """,
                    (
                        JobStatus.RETRYING.value,
                        JobStage.QUEUED.value,
                        error_code,
                        error_message,
                        now_text,
                        job_id,
                    ),
                )
                self._reset_document_for_retry(
                    connection, document, kind, now, error_code, error_message
                )
            else:
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = ?, stage = ?, lease_owner = NULL, lease_expires_at = NULL,
                        error_code = ?, error_message = ?, updated_at = ?, finished_at = ?
                    WHERE id = ?
                    """,
                    (
                        JobStatus.FAILED.value,
                        JobStage.FAILED.value,
                        error_code,
                        error_message,
                        now_text,
                        now_text,
                        job_id,
                    ),
                )
                terminal_status = (
                    DocumentStatus.DELETION_FAILED
                    if kind is JobKind.DELETE
                    else DocumentStatus.FAILED
                )
                self._set_document_state(
                    connection,
                    document,
                    terminal_status,
                    now,
                    error_code=error_code,
                    error_message=error_message,
                )
            return self._job_from_row(
                connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            )

    def requeue_job(
        self,
        job_id: str,
        worker_id: str,
        *,
        error_code: str,
        error_message: str,
    ) -> JobRecord:
        return self.fail_job(
            job_id,
            worker_id,
            error_code=error_code,
            error_message=error_message,
            retryable=True,
        )

    def update_document_status(
        self,
        document_id: str,
        status: DocumentStatus,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
        chunk_count: int | None = None,
        active_version: int | None = None,
    ) -> DocumentRecord:
        status = DocumentStatus(status)
        now = self._now()
        with self.database.transaction(immediate=True) as connection:
            document = self._require_document_row(connection, document_id, include_deleted=True)
            self._set_document_state(
                connection,
                document,
                status,
                now,
                error_code=error_code,
                error_message=error_message,
                chunk_count=chunk_count,
                active_version=active_version,
                clear_error=error_code is None and error_message is None,
            )
            return self._document_from_row(
                connection.execute(
                    "SELECT * FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
            )

    def mark_document_ready(
        self, document_id: str, *, chunk_count: int, active_version: int
    ) -> DocumentRecord:
        return self.update_document_status(
            document_id,
            DocumentStatus.READY,
            chunk_count=chunk_count,
            active_version=active_version,
        )

    def mark_document_deleted(self, document_id: str) -> DocumentRecord:
        return self.update_document_status(document_id, DocumentStatus.DELETED)

    def set_meta(self, key: str, value: str) -> None:
        key = self._validated_meta_text(key, "key", 200)
        value = self._validated_meta_text(value, "value", 20_000)
        now_text = self._serialize_datetime(self._now())
        with self.database.transaction(immediate=True) as connection:
            self._upsert_meta(connection, key, value, now_text)

    def get_meta(self, key: str) -> str | None:
        with self.database.connection() as connection:
            row = connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row is not None else None

    def record_worker_heartbeat(self, worker_id: str) -> datetime:
        worker_id = self._validated_worker_id(worker_id)
        now = self._now()
        now_text = self._serialize_datetime(now)
        with self.database.transaction(immediate=True) as connection:
            self._upsert_meta(connection, "worker_last_id", worker_id, now_text)
            self._upsert_meta(connection, "worker_last_seen_at", now_text, now_text)
        return now

    def read_worker_heartbeat(self) -> datetime | None:
        value = self.get_meta("worker_last_seen_at")
        return self._parse_datetime(value) if value is not None else None

    def _reclaim_expired_leases(self, connection: sqlite3.Connection, now: datetime) -> None:
        now_text = self._serialize_datetime(now)
        expired = connection.execute(
            """
            SELECT * FROM jobs
            WHERE status = 'running' AND lease_expires_at <= ?
            ORDER BY lease_expires_at, id
            """,
            (now_text,),
        ).fetchall()
        for job in expired:
            document = self._require_document_row(
                connection, job["document_id"], include_deleted=True
            )
            kind = JobKind(job["kind"])
            if int(job["attempts"]) >= int(job["max_attempts"]):
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = ?, stage = ?, lease_owner = NULL, lease_expires_at = NULL,
                        error_code = ?, error_message = ?, updated_at = ?, finished_at = ?
                    WHERE id = ? AND status = 'running'
                    """,
                    (
                        JobStatus.FAILED.value,
                        JobStage.FAILED.value,
                        "job_lease_expired",
                        "The worker lease expired and the attempt limit was reached",
                        now_text,
                        now_text,
                        job["id"],
                    ),
                )
                terminal = (
                    DocumentStatus.DELETION_FAILED
                    if kind is JobKind.DELETE
                    else DocumentStatus.FAILED
                )
                self._set_document_state(
                    connection,
                    document,
                    terminal,
                    now,
                    error_code="job_lease_expired",
                    error_message="The worker lease expired and the attempt limit was reached",
                )
            else:
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = ?, stage = ?, progress = 0, lease_owner = NULL,
                        lease_expires_at = NULL, error_code = ?, error_message = ?,
                        updated_at = ?, finished_at = NULL
                    WHERE id = ? AND status = 'running'
                    """,
                    (
                        JobStatus.RETRYING.value,
                        JobStage.QUEUED.value,
                        "job_lease_expired",
                        "The previous worker lease expired; the job was reclaimed",
                        now_text,
                        job["id"],
                    ),
                )
                self._reset_document_for_retry(
                    connection,
                    document,
                    kind,
                    now,
                    "job_lease_expired",
                    "The previous worker lease expired; the job was reclaimed",
                )

    def _require_active_lease(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        worker_id: str,
        now: datetime,
    ) -> sqlite3.Row:
        job = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if job is None:
            raise RagError("job_not_found", "Job not found", status_code=404)
        lease_expires_at = job["lease_expires_at"]
        owns_live_lease = (
            JobStatus(job["status"]) is JobStatus.RUNNING
            and job["lease_owner"] == worker_id
            and lease_expires_at is not None
            and self._parse_datetime(lease_expires_at) > now
        )
        if not owns_live_lease:
            raise RagError(
                "job_lease_conflict",
                "The job is not covered by a live lease owned by this worker",
                status_code=409,
            )
        return cast(sqlite3.Row, job)

    def _reset_document_for_retry(
        self,
        connection: sqlite3.Connection,
        document: sqlite3.Row,
        kind: JobKind,
        now: datetime,
        error_code: str,
        error_message: str,
    ) -> None:
        status = {
            JobKind.INGEST: DocumentStatus.QUEUED,
            JobKind.REINDEX: DocumentStatus.REINDEXING,
            JobKind.DELETE: DocumentStatus.DELETING,
        }[kind]
        connection.execute(
            """
            UPDATE documents
            SET status = ?, error_code = ?, error_message = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                status.value,
                error_code,
                error_message,
                self._serialize_datetime(now),
                document["id"],
            ),
        )

    def _set_document_state(
        self,
        connection: sqlite3.Connection,
        document: sqlite3.Row,
        status: DocumentStatus,
        now: datetime,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
        chunk_count: int | None = None,
        active_version: int | None = None,
        embedding_fingerprint: str | None = None,
        clear_error: bool = False,
    ) -> None:
        current = DocumentStatus(document["status"])
        if status is not current and status not in _DOCUMENT_TRANSITIONS[current]:
            raise RagError(
                "invalid_document_transition",
                f"Document status cannot move from {current.value!r} to {status.value!r}",
                status_code=409,
            )
        if chunk_count is not None and chunk_count < 0:
            raise ValueError("chunk_count must be non-negative")
        if active_version is not None and active_version < 0:
            raise ValueError("active_version must be non-negative")
        persisted_error_code = None if clear_error else error_code or document["error_code"]
        persisted_error_message = (
            None if clear_error else error_message or document["error_message"]
        )
        persisted_chunk_count = int(document["chunk_count"]) if chunk_count is None else chunk_count
        persisted_active_version = (
            int(document["active_version"]) if active_version is None else active_version
        )
        persisted_embedding_fingerprint = (
            str(document["embedding_fingerprint"])
            if embedding_fingerprint is None
            else embedding_fingerprint.lower()
        )
        if _HASH_PATTERN.fullmatch(persisted_embedding_fingerprint) is None:
            raise ValueError("embedding_fingerprint must be a lowercase SHA-256 hex digest")
        persisted_deleted_at = (
            self._serialize_datetime(now)
            if status is DocumentStatus.DELETED
            else document["deleted_at"]
        )
        connection.execute(
            """
            UPDATE documents
            SET status = ?, updated_at = ?, error_code = ?, error_message = ?,
                chunk_count = ?, active_version = ?, embedding_fingerprint = ?, deleted_at = ?
            WHERE id = ?
            """,
            (
                status.value,
                self._serialize_datetime(now),
                persisted_error_code,
                persisted_error_message,
                persisted_chunk_count,
                persisted_active_version,
                persisted_embedding_fingerprint,
                persisted_deleted_at,
                document["id"],
            ),
        )

    def _insert_job(
        self,
        connection: sqlite3.Connection,
        document_id: str,
        kind: JobKind,
        now: datetime,
    ) -> JobRecord:
        job_id = uuid4().hex
        now_text = self._serialize_datetime(now)
        connection.execute(
            """
            INSERT INTO jobs (
                id, document_id, kind, status, stage, max_attempts, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                document_id,
                kind.value,
                JobStatus.QUEUED.value,
                JobStage.QUEUED.value,
                self.max_attempts,
                now_text,
                now_text,
            ),
        )
        return self._job_from_row(
            connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        )

    @staticmethod
    def _active_job_row(connection: sqlite3.Connection, document_id: str) -> sqlite3.Row | None:
        row = connection.execute(
            """
            SELECT * FROM jobs
            WHERE document_id = ? AND status IN ('queued', 'running', 'retrying')
            ORDER BY created_at DESC, id DESC LIMIT 1
            """,
            (document_id,),
        ).fetchone()
        return cast(sqlite3.Row | None, row)

    @staticmethod
    def _raise_active_job_conflict() -> None:
        raise RagError(
            "active_job_conflict",
            "The document already has a different active job",
            status_code=409,
        )

    @staticmethod
    def _require_document_row(
        connection: sqlite3.Connection,
        document_id: str,
        *,
        include_deleted: bool = False,
    ) -> sqlite3.Row:
        query = (
            "SELECT * FROM documents WHERE id = ?"
            if include_deleted
            else "SELECT * FROM documents WHERE id = ? AND status <> 'deleted'"
        )
        row = connection.execute(query, (document_id,)).fetchone()
        if row is None:
            raise RagError("document_not_found", "Document not found", status_code=404)
        return cast(sqlite3.Row, row)

    @staticmethod
    def _upsert_meta(connection: sqlite3.Connection, key: str, value: str, updated_at: str) -> None:
        connection.execute(
            """
            INSERT INTO meta (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, updated_at),
        )

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord.model_validate(dict(row))

    def _conversation_summary_from_row(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> ConversationSummary:
        count = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM conversation_turns
            WHERE conversation_id = ? AND status = 'completed'
            """,
            (row["id"],),
        ).fetchone()
        return ConversationSummary(
            id=str(row["id"]),
            title=str(row["title"]),
            turn_count=int(count["total"]),
            created_at=self._parse_datetime(str(row["created_at"])),
            updated_at=self._parse_datetime(str(row["updated_at"])),
        )

    def _conversation_turn_from_row(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> ConversationTurn:
        citation_rows = connection.execute(
            """
            SELECT * FROM turn_citations
            WHERE turn_id = ?
            ORDER BY ordinal
            """,
            (row["id"],),
        ).fetchall()
        citations = [
            Citation(
                label=str(citation["label"]),
                document_id=str(citation["document_id"]),
                chunk_id=str(citation["chunk_id"]),
                document_name=str(citation["document_name"]),
                page_number=(
                    int(citation["page_number"]) if citation["page_number"] is not None else None
                ),
                section=(str(citation["section"]) if citation["section"] is not None else None),
                snippet=str(citation["snippet"]),
                score=float(citation["score"]) if citation["score"] is not None else None,
            )
            for citation in citation_rows
        ]
        document_ids_raw = row["document_ids_json"]
        document_ids: list[str] | None = None
        if document_ids_raw is not None:
            parsed = json.loads(str(document_ids_raw))
            if not isinstance(parsed, list) or not all(isinstance(value, str) for value in parsed):
                raise RuntimeError("Persisted conversation document scope is invalid")
            document_ids = parsed
        return ConversationTurn(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            client_turn_id=str(row["client_turn_id"]),
            status=ConversationTurnStatus(row["status"]),
            question=str(row["question"]),
            answer=str(row["answer"]) if row["answer"] is not None else None,
            citations=citations,
            no_answer=bool(row["no_answer"]),
            top_k=int(row["top_k"]) if row["top_k"] is not None else None,
            document_ids=document_ids,
            request_id=str(row["request_id"]) if row["request_id"] is not None else None,
            error_code=(str(row["error_code"]) if row["error_code"] is not None else None),
            retryable=bool(row["retryable"]),
            created_at=self._parse_datetime(str(row["created_at"])),
            updated_at=self._parse_datetime(str(row["updated_at"])),
        )

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> JobRecord:
        return JobRecord.model_validate(dict(row))

    def _upload_receipt(
        self,
        connection: sqlite3.Connection,
        document_id: str,
        job_id: str,
        *,
        duplicate: bool,
    ) -> UploadReceipt:
        document = self._document_from_row(
            connection.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        )
        job = self._job_from_row(
            connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        )
        return UploadReceipt(
            document=DocumentPublic.from_record(document), job=job, duplicate=duplicate
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise RuntimeError("Repository clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    @staticmethod
    def _serialize_datetime(value: datetime) -> str:
        return value.astimezone(UTC).isoformat(timespec="microseconds")

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    def _validated_lease_seconds(self, value: int | None) -> int:
        duration = self.lease_seconds if value is None else value
        if duration < 1 or duration > 86_400:
            raise ValueError("lease_seconds must be between 1 and 86400")
        return duration

    @staticmethod
    def _validate_pagination(limit: int, offset: int) -> None:
        if not 1 <= limit <= _MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {_MAX_PAGE_SIZE}")
        if offset < 0:
            raise ValueError("offset must be non-negative")

    @staticmethod
    def _validated_identifier(value: str, field: str) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > 128 or any(ord(char) < 32 for char in normalized):
            raise ValueError(f"{field} must contain 1 to 128 safe characters")
        return normalized

    @classmethod
    def _validated_client_turn_id(cls, value: str) -> str:
        normalized = cls._validated_identifier(value, "client_turn_id")
        if len(normalized) < 8:
            raise ValueError("client_turn_id must contain 8 to 128 safe characters")
        return normalized

    @staticmethod
    def _validated_conversation_text(value: str, field: str, maximum: int) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{field} must be text")
        normalized = value.strip()
        if not normalized or len(normalized) > maximum or "\x00" in normalized:
            raise ValueError(f"{field} must contain 1 to {maximum} safe characters")
        return normalized

    @classmethod
    def _validated_conversation_title(cls, value: str) -> str:
        normalized = " ".join(cls._validated_conversation_text(value, "title", 120).split())
        if not normalized:
            raise ValueError("title must contain 1 to 120 safe characters")
        return normalized

    @staticmethod
    def _validated_top_k(value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 50:
            raise ValueError("top_k must be between 1 and 50")
        return value

    @classmethod
    def _validated_document_ids(cls, values: Sequence[str] | None) -> list[str] | None:
        if values is None:
            return None
        normalized = [cls._validated_identifier(value, "document_id") for value in values]
        unique = list(dict.fromkeys(normalized))
        if len(unique) > 100:
            raise ValueError("at most 100 document identifiers may be stored")
        return unique

    @classmethod
    def _validated_citation(cls, citation: Citation) -> Citation:
        label = cls._validated_conversation_text(citation.label, "citation label", 32)
        document_id = cls._validated_identifier(citation.document_id, "citation document_id")
        chunk_id = cls._validated_identifier(citation.chunk_id, "citation chunk_id")
        document_name = cls._validated_conversation_text(
            citation.document_name, "citation document_name", 255
        )
        section = (
            cls._validated_conversation_text(citation.section, "citation section", 1_000)
            if citation.section is not None
            else None
        )
        snippet = cls._validated_conversation_text(citation.snippet, "citation snippet", 2_000)
        return citation.model_copy(
            update={
                "label": label,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "document_name": document_name,
                "section": section,
                "snippet": snippet,
            }
        )

    @classmethod
    def _derived_conversation_title(cls, question: str) -> str:
        normalized = " ".join(cls._validated_conversation_text(question, "question", 4_000).split())
        return normalized if len(normalized) <= 72 else f"{normalized[:71].rstrip()}…"

    @classmethod
    def _validated_worker_id(cls, value: str) -> str:
        return cls._validated_identifier(value, "worker_id")

    @staticmethod
    def _validated_error_text(value: str, field: str, maximum: int) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > maximum:
            raise ValueError(f"{field} must contain 1 to {maximum} characters")
        return normalized

    @staticmethod
    def _validated_meta_text(value: str, field: str, maximum: int) -> str:
        if not value or len(value) > maximum or "\x00" in value:
            raise ValueError(f"meta {field} must contain 1 to {maximum} characters")
        return value

    @staticmethod
    def _validated_upload_fields(**values: Any) -> dict[str, Any]:
        text_limits = {
            "display_name": 255,
            "stored_path": 4_096,
            "content_type": 255,
            "extension": 32,
        }
        normalized: dict[str, Any] = {}
        for field, maximum in text_limits.items():
            value = str(values[field]).strip()
            if not value or len(value) > maximum or "\x00" in value:
                raise ValueError(f"{field} must contain 1 to {maximum} safe characters")
            normalized[field] = value
        normalized["extension"] = normalized["extension"].lower()
        if not normalized["extension"].startswith("."):
            raise ValueError("extension must begin with a dot")
        storage_key = normalized["stored_path"]
        parsed_storage_key = PurePosixPath(storage_key)
        if (
            parsed_storage_key.is_absolute()
            or parsed_storage_key.parts != (storage_key,)
            or storage_key in {".", ".."}
            or "\\" in storage_key
            or not storage_key.endswith(normalized["extension"])
        ):
            raise ValueError("stored_path must be a relative managed upload key")
        for field in ("content_sha256", "embedding_fingerprint"):
            value = str(values[field]).lower()
            if _HASH_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
            normalized[field] = value
        size_bytes = values["size_bytes"]
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative integer")
        normalized["size_bytes"] = size_bytes
        idempotency_key = values["idempotency_key"]
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str)
            or not idempotency_key.strip()
            or len(idempotency_key) > 200
            or any(ord(char) < 32 for char in idempotency_key)
        ):
            raise ValueError("idempotency_key must contain 1 to 200 safe characters")
        return normalized

    @staticmethod
    def _upload_request_fingerprint(values: dict[str, Any]) -> str:
        # The server-generated storage path can legitimately change when an HTTP
        # retry has already staged another temporary file. It is not client intent.
        request_values = {key: value for key, value in values.items() if key != "stored_path"}
        payload = json.dumps(request_values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
