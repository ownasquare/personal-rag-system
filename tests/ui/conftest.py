"""AppTest fixtures for the Phase 2 Personal Library presentation layer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from personal_rag.config import Settings
from personal_rag.models import (
    ChatResponse,
    Citation,
    ConversationList,
    ConversationSummary,
    ConversationTurn,
    ConversationTurnCreate,
    ConversationTurnList,
    ConversationTurnStatus,
    DependencyState,
    DocumentList,
    DocumentPublic,
    DocumentSort,
    DocumentStatus,
    JobKind,
    JobList,
    JobRecord,
    JobStage,
    JobStatus,
    SortOrder,
    SystemStatus,
    UploadReceipt,
)
from personal_rag.ui.client import ApiClientError, HealthCheck

APP_PATH = Path(__file__).resolve().parents[2] / "src" / "personal_rag" / "ui" / "app.py"
NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


def make_document(
    name: str = "field-notes.md",
    *,
    document_id: str = "doc-1",
    status: DocumentStatus = DocumentStatus.READY,
) -> DocumentPublic:
    """Build a stable public document contract for UI tests."""

    return DocumentPublic(
        id=document_id,
        display_name=name,
        content_type="text/markdown",
        extension=".md",
        size_bytes=2048,
        status=status,
        active_version=1 if status is DocumentStatus.READY else 0,
        chunk_count=4 if status is DocumentStatus.READY else 0,
        error_code=None,
        error_message=None,
        created_at=NOW,
        updated_at=NOW,
    )


def make_job(
    *,
    job_id: str = "job-1",
    document_id: str = "doc-1",
    status: JobStatus = JobStatus.QUEUED,
    stage: JobStage = JobStage.QUEUED,
    kind: JobKind = JobKind.INGEST,
) -> JobRecord:
    """Build a stable durable job contract for UI tests."""

    return JobRecord(
        id=job_id,
        document_id=document_id,
        kind=kind,
        status=status,
        stage=stage,
        progress=1.0 if status is JobStatus.SUCCEEDED else 0.0,
        attempts=1,
        max_attempts=3,
        created_at=NOW,
        updated_at=NOW,
        finished_at=NOW if status in {JobStatus.SUCCEEDED, JobStatus.FAILED} else None,
    )


def make_conversation(
    *,
    conversation_id: str = "conversation-1",
    title: str = "Atlas launch notes",
    turn_count: int = 1,
) -> ConversationSummary:
    return ConversationSummary(
        id=conversation_id,
        title=title,
        turn_count=turn_count,
        created_at=NOW,
        updated_at=NOW,
    )


def make_turn(
    *,
    turn_id: str = "turn-1",
    conversation_id: str = "conversation-1",
    client_turn_id: str = "client-turn-1",
    status: ConversationTurnStatus = ConversationTurnStatus.COMPLETED,
    question: str = "What is the launch key?",
    answer: str | None = "The launch key is cobalt [S1].",
    no_answer: bool = False,
) -> ConversationTurn:
    citations = []
    if status is ConversationTurnStatus.COMPLETED and not no_answer:
        citations = [
            Citation(
                label="S1",
                document_id="doc-1",
                chunk_id="doc-1:1",
                document_name="field-notes.md",
                section="Launch checklist",
                snippet="The Atlas launch key is cobalt.",
                score=0.92,
            )
        ]
    return ConversationTurn(
        id=turn_id,
        conversation_id=conversation_id,
        client_turn_id=client_turn_id,
        status=status,
        question=question,
        answer=answer,
        citations=citations,
        no_answer=no_answer,
        top_k=5,
        document_ids=None,
        request_id="request-1",
        error_code="provider_unavailable" if status is ConversationTurnStatus.FAILED else None,
        retryable=status is ConversationTurnStatus.FAILED,
        created_at=NOW,
        updated_at=NOW,
    )


def ready_status(*, document_count: int = 0) -> SystemStatus:
    """Return a sanitized ready-system snapshot."""

    return SystemStatus(
        status="ready",
        collection="personal_knowledge",
        document_count=document_count,
        ready_document_count=document_count,
        chunk_count=document_count * 4,
        queued_job_count=0,
        embedding_provider="openai",
        embedding_model="text-embedding-3-large",
        embedding_dimensions=3072,
        chat_provider="openai",
        chat_model="gpt-4.1-mini",
        dependencies=[DependencyState(name="providers", status="ready")],
        worker_last_seen_at=NOW,
    )


@dataclass
class FakeRagClient:
    """In-memory API boundary used by AppTest; it never opens a socket."""

    documents: list[DocumentPublic] = field(default_factory=list)
    system_status: SystemStatus = field(default_factory=ready_status)
    jobs: dict[str, JobRecord] = field(default_factory=dict)
    conversations: list[ConversationSummary] = field(default_factory=list)
    turns: dict[str, list[ConversationTurn]] = field(default_factory=dict)
    upload_calls: list[str] = field(default_factory=list)
    upload_errors: dict[str, ApiClientError] = field(default_factory=dict)
    delete_calls: list[str] = field(default_factory=list)
    reindex_calls: list[str] = field(default_factory=list)
    turn_calls: list[ConversationTurnCreate] = field(default_factory=list)
    chat_error: ApiClientError | None = None
    status_error: ApiClientError | None = None
    documents_error: ApiClientError | None = None
    no_answer: bool = False
    get_status_calls: int = 0
    list_documents_calls: int = 0
    list_all_documents_calls: int = 0
    document_page_requests: list[dict[str, object]] = field(default_factory=list)
    inconsistent_document_page_once: bool = False
    list_jobs_calls: int = 0
    list_conversations_calls: int = 0
    health_live_calls: int = 0
    health_ready_calls: int = 0

    def health_live(self) -> HealthCheck:
        self.health_live_calls += 1
        return HealthCheck(status="alive")

    def health_ready(self) -> HealthCheck:
        self.health_ready_calls += 1
        return HealthCheck(status="ready")

    def get_status(self) -> SystemStatus:
        self.get_status_calls += 1
        if self.status_error is not None:
            raise self.status_error
        return self.system_status

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
        self.list_documents_calls += 1
        normalized_statuses = (
            tuple(dict.fromkeys(DocumentStatus(status) for status in statuses))
            if statuses
            else None
        )
        normalized_sort = DocumentSort(sort)
        normalized_order = SortOrder(order)
        self.document_page_requests.append(
            {
                "limit": limit,
                "offset": offset,
                "query": query,
                "statuses": normalized_statuses,
                "sort": normalized_sort,
                "order": normalized_order,
            }
        )

        items = [
            document for document in self.documents if document.status is not DocumentStatus.DELETED
        ]
        if query is not None:
            needle = query.casefold()
            items = [
                document
                for document in items
                if needle in document.display_name.casefold()
                or needle in document.extension.casefold()
            ]
        if normalized_statuses is not None:
            allowed = set(normalized_statuses)
            items = [document for document in items if document.status in allowed]

        def sort_key(document: DocumentPublic) -> tuple[object, str]:
            if normalized_sort is DocumentSort.NAME:
                return document.display_name.casefold(), document.id
            if normalized_sort is DocumentSort.UPDATED:
                return document.updated_at, document.id
            return document.created_at, document.id

        items.sort(key=sort_key, reverse=normalized_order is SortOrder.DESC)
        total = len(items)
        if self.inconsistent_document_page_once and total > 0:
            self.inconsistent_document_page_once = False
            return DocumentList(items=[], total=total, limit=limit, offset=offset)
        return DocumentList(
            items=items[offset : offset + limit],
            total=total,
            limit=limit,
            offset=offset,
        )

    def list_all_documents(self, *, max_documents: int = 2000) -> list[DocumentPublic]:
        self.list_all_documents_calls += 1
        if self.documents_error is not None:
            raise self.documents_error
        return self.documents[:max_documents]

    def get_document(self, document_id: str) -> DocumentPublic:
        return next(document for document in self.documents if document.id == document_id)

    def upload_document(self, filename: str, content: bytes, content_type: str) -> UploadReceipt:
        del content, content_type
        self.upload_calls.append(filename)
        if filename in self.upload_errors:
            raise self.upload_errors[filename]
        index = len(self.upload_calls)
        document = make_document(
            filename,
            document_id=f"uploaded-{index}",
            status=DocumentStatus.QUEUED,
        )
        job = make_job(job_id=f"upload-job-{index}", document_id=document.id)
        self.documents.append(document)
        self.jobs[job.id] = job
        return UploadReceipt(document=document, job=job, duplicate=False)

    def get_job(self, job_id: str) -> JobRecord:
        return self.jobs[job_id]

    def list_jobs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: JobStatus | None = None,
        document_id: str | None = None,
    ) -> JobList:
        self.list_jobs_calls += 1
        items = list(reversed(list(self.jobs.values())))
        if status is not None:
            items = [item for item in items if item.status is status]
        if document_id is not None:
            items = [item for item in items if item.document_id == document_id]
        total = len(items)
        return JobList(
            items=items[offset : offset + limit],
            total=total,
            limit=limit,
            offset=offset,
        )

    def reindex_document(self, document_id: str) -> JobRecord:
        self.reindex_calls.append(document_id)
        job = make_job(
            job_id=f"reindex-{document_id}",
            document_id=document_id,
            kind=JobKind.REINDEX,
        )
        self.jobs[job.id] = job
        return job

    def delete_document(self, document_id: str) -> JobRecord:
        self.delete_calls.append(document_id)
        job = make_job(
            job_id=f"delete-{document_id}",
            document_id=document_id,
            kind=JobKind.DELETE,
        )
        self.jobs[job.id] = job
        return job

    def create_conversation(self, title: str | None = None) -> ConversationSummary:
        index = len(self.conversations) + 1
        conversation = make_conversation(
            conversation_id=f"conversation-{index}",
            title=title or "New conversation",
            turn_count=0,
        )
        self.conversations.insert(0, conversation)
        self.turns[conversation.id] = []
        return conversation

    def list_conversations(self, *, limit: int = 50, offset: int = 0) -> ConversationList:
        self.list_conversations_calls += 1
        return ConversationList(
            items=self.conversations[offset : offset + limit],
            total=len(self.conversations),
            limit=limit,
            offset=offset,
        )

    def get_conversation(self, conversation_id: str) -> ConversationSummary:
        return next(item for item in self.conversations if item.id == conversation_id)

    def delete_conversation(self, conversation_id: str) -> None:
        self.conversations = [item for item in self.conversations if item.id != conversation_id]
        self.turns.pop(conversation_id, None)

    def list_conversation_turns(
        self,
        conversation_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> ConversationTurnList:
        turns = self.turns.get(conversation_id, [])
        return ConversationTurnList(
            items=turns[offset : offset + limit],
            total=len(turns),
            limit=limit,
            offset=offset,
        )

    def create_conversation_turn(
        self,
        conversation_id: str,
        turn: ConversationTurnCreate,
    ) -> ConversationTurn:
        self.turn_calls.append(turn)
        if self.chat_error is not None:
            raise self.chat_error
        existing = next(
            (
                item
                for item in self.turns.get(conversation_id, [])
                if item.client_turn_id == turn.client_turn_id
            ),
            None,
        )
        if existing is not None:
            if (
                existing.question != turn.message
                or existing.top_k != turn.top_k
                or existing.document_ids != turn.document_ids
            ):
                raise ApiClientError(
                    code="idempotency_conflict",
                    message="This saved request identifier belongs to different question details.",
                    status_code=409,
                    retryable=False,
                )
            if existing.status in {
                ConversationTurnStatus.PENDING,
                ConversationTurnStatus.FAILED,
            }:
                completed = make_turn(
                    turn_id=existing.id,
                    conversation_id=conversation_id,
                    client_turn_id=turn.client_turn_id,
                    question=turn.message,
                ).model_copy(update={"top_k": turn.top_k, "document_ids": turn.document_ids})
                conversation_turns = self.turns.get(conversation_id, [])
                self.turns[conversation_id] = [
                    completed if item.id == existing.id else item for item in conversation_turns
                ]
                return completed
            return existing
        index = sum(len(items) for items in self.turns.values()) + 1
        created = make_turn(
            turn_id=f"turn-{index}",
            conversation_id=conversation_id,
            client_turn_id=turn.client_turn_id,
            question=turn.message,
            answer=(
                "I could not find enough support in the selected documents."
                if self.no_answer
                else "The launch key is cobalt [S1]."
            ),
            no_answer=self.no_answer,
        ).model_copy(update={"top_k": turn.top_k, "document_ids": turn.document_ids})
        self.turns.setdefault(conversation_id, []).append(created)
        for position, conversation in enumerate(self.conversations):
            if conversation.id != conversation_id:
                continue
            title = conversation.title
            if conversation.turn_count == 0:
                title = turn.message[:72]
            self.conversations[position] = conversation.model_copy(
                update={
                    "title": title,
                    "turn_count": conversation.turn_count + 1,
                    "updated_at": NOW + timedelta(seconds=index),
                }
            )
            break
        return created

    def chat(self, request: object) -> ChatResponse:
        del request
        raise AssertionError("The Phase 2 UI must use durable conversation turns")

    def close(self) -> None:
        return None


@pytest.fixture
def ui_settings() -> Settings:
    return Settings(
        environment="test",
        auth_enabled=False,
        retrieval_top_k=5,
        retrieval_max_top_k=10,
        ui_poll_seconds=10.0,
        ui_poll_timeout_seconds=30.0,
    )


@pytest.fixture
def fake_client() -> FakeRagClient:
    return FakeRagClient()


@pytest.fixture
def app_test(fake_client: FakeRagClient, ui_settings: Settings) -> AppTest:
    test = AppTest.from_file(str(APP_PATH), default_timeout=8)
    test.session_state["_rag_client"] = fake_client
    test.session_state["_rag_settings"] = ui_settings
    return test
