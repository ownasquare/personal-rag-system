"""AppTest fixtures for the Streamlit presentation layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from personal_rag.config import Settings
from personal_rag.models import (
    ChatRequest,
    ChatResponse,
    Citation,
    DependencyState,
    DocumentList,
    DocumentPublic,
    DocumentStatus,
    JobKind,
    JobRecord,
    JobStage,
    JobStatus,
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
        active_version=1 if status == DocumentStatus.READY else 0,
        chunk_count=4 if status == DocumentStatus.READY else 0,
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
    """Build a stable job contract for UI tests."""

    return JobRecord(
        id=job_id,
        document_id=document_id,
        kind=kind,
        status=status,
        stage=stage,
        progress=1.0 if status == JobStatus.SUCCEEDED else 0.0,
        attempts=0,
        max_attempts=3,
        created_at=NOW,
        updated_at=NOW,
        finished_at=NOW if status in {JobStatus.SUCCEEDED, JobStatus.FAILED} else None,
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
    upload_calls: list[str] = field(default_factory=list)
    delete_calls: list[str] = field(default_factory=list)
    reindex_calls: list[str] = field(default_factory=list)
    chat_calls: list[ChatRequest] = field(default_factory=list)
    chat_error: ApiClientError | None = None
    chat_response: ChatResponse = field(
        default_factory=lambda: ChatResponse(
            answer="The launch key is cobalt [S1].",
            citations=[
                Citation(
                    label="S1",
                    document_id="doc-1",
                    chunk_id="doc-1:1",
                    document_name="field-notes.md",
                    page_number=None,
                    section="Launch checklist",
                    snippet="The Atlas launch key is cobalt.",
                    score=0.92,
                )
            ],
            no_answer=False,
            request_id="request-1",
        )
    )

    def health_live(self) -> HealthCheck:
        return HealthCheck(status="alive")

    def health_ready(self) -> HealthCheck:
        return HealthCheck(status="ready")

    def get_status(self) -> SystemStatus:
        return self.system_status

    def list_documents(self, *, limit: int = 100, offset: int = 0) -> DocumentList:
        items = self.documents[offset : offset + limit]
        return DocumentList(items=items, total=len(self.documents), limit=limit, offset=offset)

    def list_all_documents(self, *, max_documents: int = 2000) -> list[DocumentPublic]:
        return self.documents[:max_documents]

    def get_document(self, document_id: str) -> DocumentPublic:
        return next(document for document in self.documents if document.id == document_id)

    def upload_document(self, filename: str, content: bytes, content_type: str) -> UploadReceipt:
        del content, content_type
        self.upload_calls.append(filename)
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

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_calls.append(request)
        if self.chat_error is not None:
            raise self.chat_error
        return self.chat_response

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
