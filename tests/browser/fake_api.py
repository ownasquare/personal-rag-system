"""Stateful no-network API fixture for rendered Streamlit browser proof."""

from __future__ import annotations

import unicodedata
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from personal_rag.document_types import DOCUMENT_TYPES_BY_EXTENSION
from personal_rag.models import (
    ChatRequest,
    ChatResponse,
    Citation,
    ConversationCreate,
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

TOKEN = "browser-proof-token-long-enough"
NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)
bearer = HTTPBearer(auto_error=False)
app = FastAPI(title="Personal RAG browser fixture")

READY_DOCUMENT = DocumentPublic(
    id="browser-doc-1",
    display_name="atlas-launch-notes.md",
    content_type="text/markdown",
    extension=".md",
    size_bytes=2048,
    status=DocumentStatus.READY,
    active_version=1,
    chunk_count=4,
    error_code=None,
    error_message=None,
    created_at=NOW - timedelta(days=2),
    updated_at=NOW - timedelta(hours=2),
)
FAILED_DOCUMENT = DocumentPublic(
    id="browser-doc-failed",
    display_name="field-interview-notes.pdf",
    content_type="application/pdf",
    extension=".pdf",
    size_bytes=8192,
    status=DocumentStatus.FAILED,
    active_version=0,
    chunk_count=0,
    error_code="embedding_provider_timeout",
    error_message="The document provider timed out.",
    created_at=NOW - timedelta(days=1),
    updated_at=NOW - timedelta(hours=1),
)


def _library_document(index: int) -> DocumentPublic:
    """Build stable older records so rendered proof spans multiple result pages."""

    statuses = (
        DocumentStatus.READY,
        DocumentStatus.QUEUED,
        DocumentStatus.FAILED,
        DocumentStatus.DELETION_FAILED,
    )
    status = statuses[index % len(statuses)]
    display_name = "Résumé 100%_plan.md" if index == 3 else f"project-{index:02d}-reference.md"
    return DocumentPublic(
        id=f"browser-library-{index:02d}",
        display_name=display_name,
        content_type="text/markdown",
        extension=".md",
        size_bytes=1024 + index,
        status=status,
        active_version=1 if status is DocumentStatus.READY else 0,
        chunk_count=3 if status is DocumentStatus.READY else 0,
        error_code=(
            "fixture_processing_failed"
            if status in {DocumentStatus.FAILED, DocumentStatus.DELETION_FAILED}
            else None
        ),
        error_message=None,
        created_at=NOW - timedelta(days=10 + index),
        updated_at=NOW - timedelta(days=5 + index),
    )


LIBRARY_DOCUMENTS = [_library_document(index) for index in range(1, 15)]
ATLAS_CITATION = Citation(
    label="S1",
    document_id=READY_DOCUMENT.id,
    chunk_id="browser-chunk-1",
    document_name=READY_DOCUMENT.display_name,
    section="Launch checklist",
    snippet="The Atlas launch key is cobalt blue.",
    score=0.94,
)

_documents: dict[str, DocumentPublic] = {
    READY_DOCUMENT.id: READY_DOCUMENT,
    FAILED_DOCUMENT.id: FAILED_DOCUMENT,
    **{document.id: document for document in LIBRARY_DOCUMENTS},
}
_jobs: dict[str, JobRecord] = {
    "browser-job-ready": JobRecord(
        id="browser-job-ready",
        document_id=READY_DOCUMENT.id,
        kind=JobKind.INGEST,
        status=JobStatus.SUCCEEDED,
        stage=JobStage.COMPLETE,
        progress=1.0,
        attempts=1,
        max_attempts=3,
        created_at=NOW - timedelta(days=2),
        updated_at=NOW - timedelta(days=2) + timedelta(minutes=2),
        finished_at=NOW - timedelta(days=2) + timedelta(minutes=2),
    ),
    "browser-job-failed": JobRecord(
        id="browser-job-failed",
        document_id=FAILED_DOCUMENT.id,
        kind=JobKind.INGEST,
        status=JobStatus.FAILED,
        stage=JobStage.FAILED,
        progress=0.45,
        attempts=3,
        max_attempts=3,
        error_code="embedding_provider_timeout",
        error_message="The document provider timed out.",
        created_at=NOW - timedelta(days=1),
        updated_at=NOW - timedelta(hours=1),
        finished_at=NOW - timedelta(hours=1),
    ),
}
_conversations: dict[str, ConversationSummary] = {
    "browser-conversation-1": ConversationSummary(
        id="browser-conversation-1",
        title="Atlas launch checklist",
        turn_count=1,
        created_at=NOW - timedelta(hours=3),
        updated_at=NOW - timedelta(hours=2),
    )
}
_turns: dict[str, list[ConversationTurn]] = {
    "browser-conversation-1": [
        ConversationTurn(
            id="browser-turn-1",
            conversation_id="browser-conversation-1",
            client_turn_id="browser-client-turn-1",
            status=ConversationTurnStatus.COMPLETED,
            question="What is the Atlas launch key?",
            answer="The Atlas launch key is cobalt blue [S1].",
            citations=[ATLAS_CITATION],
            no_answer=False,
            top_k=5,
            document_ids=[READY_DOCUMENT.id],
            request_id="browser-proof-saved-turn",
            created_at=NOW - timedelta(hours=3),
            updated_at=NOW - timedelta(hours=2),
        )
    ]
}
_sequence = {"conversation": 1, "document": 2, "job": 2, "turn": 1}
_job_reads: dict[str, int] = {}


def authorize(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> None:
    if credentials is None or credentials.credentials != TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")


def _next_identifier(kind: str) -> str:
    _sequence[kind] += 1
    return f"browser-{kind}-{_sequence[kind]}"


def _fold_document_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _visible_documents(
    *,
    sort: DocumentSort = DocumentSort.CREATED,
    order: SortOrder = SortOrder.DESC,
) -> list[DocumentPublic]:
    key = {
        DocumentSort.CREATED: lambda document: (document.created_at, document.id),
        DocumentSort.UPDATED: lambda document: (document.updated_at, document.id),
        DocumentSort.NAME: lambda document: (
            _fold_document_text(document.display_name),
            document.id,
        ),
    }[sort]
    return sorted(
        (
            document
            for document in _documents.values()
            if document.status is not DocumentStatus.DELETED
        ),
        key=key,
        reverse=order is SortOrder.DESC,
    )


def _require_document(document_id: str) -> DocumentPublic:
    document = _documents.get(document_id)
    if document is None or document.status is DocumentStatus.DELETED:
        raise HTTPException(status_code=404, detail="document not found")
    return document


def _require_conversation(conversation_id: str) -> ConversationSummary:
    conversation = _conversations.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conversation


def _new_job(document_id: str, kind: JobKind) -> JobRecord:
    job_id = _next_identifier("job")
    created_at = NOW + timedelta(minutes=_sequence["job"])
    job = JobRecord(
        id=job_id,
        document_id=document_id,
        kind=kind,
        status=JobStatus.QUEUED,
        stage=JobStage.QUEUED,
        progress=0.0,
        attempts=0,
        max_attempts=3,
        created_at=created_at,
        updated_at=created_at,
    )
    _jobs[job.id] = job
    _job_reads[job.id] = 0
    return job


def _purge_document_history(document_id: str) -> None:
    """Mirror the real deletion privacy rule in the rendered-proof fixture."""

    for conversation_id, turns in list(_turns.items()):
        remaining = [
            turn
            for turn in turns
            if all(citation.document_id != document_id for citation in turn.citations)
        ]
        if len(remaining) == len(turns):
            continue
        if not remaining:
            _turns.pop(conversation_id, None)
            _conversations.pop(conversation_id, None)
            continue
        conversation = _conversations[conversation_id]
        first_question = " ".join(remaining[0].question.split())
        title = first_question if len(first_question) <= 72 else f"{first_question[:71].rstrip()}…"
        _turns[conversation_id] = remaining
        _conversations[conversation_id] = conversation.model_copy(
            update={
                "title": title,
                "turn_count": len(remaining),
                "updated_at": remaining[-1].updated_at,
            }
        )


def _advance_job(job_id: str) -> JobRecord:
    """Advance fixture work one observable step without wall-clock timing."""

    job = _jobs[job_id]
    if job.status not in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRYING}:
        return job
    observations = _job_reads.get(job_id, 0) + 1
    _job_reads[job_id] = observations
    updated_at = job.created_at + timedelta(seconds=observations)
    if observations == 1:
        running = job.model_copy(
            update={
                "status": JobStatus.RUNNING,
                "stage": (JobStage.DELETING if job.kind is JobKind.DELETE else JobStage.EMBEDDING),
                "progress": 0.55,
                "attempts": 1,
                "updated_at": updated_at,
            }
        )
        _jobs[job_id] = running
        return running

    completed = job.model_copy(
        update={
            "status": JobStatus.SUCCEEDED,
            "stage": JobStage.COMPLETE,
            "progress": 1.0,
            "attempts": max(1, job.attempts),
            "updated_at": updated_at,
            "finished_at": updated_at,
        }
    )
    _jobs[job_id] = completed
    document = _documents[job.document_id]
    if job.kind is JobKind.DELETE:
        _documents[job.document_id] = document.model_copy(
            update={
                "status": DocumentStatus.DELETED,
                "error_code": None,
                "error_message": None,
                "updated_at": updated_at,
            }
        )
        _purge_document_history(job.document_id)
    else:
        _documents[job.document_id] = document.model_copy(
            update={
                "status": DocumentStatus.READY,
                "active_version": max(1, document.active_version + 1),
                "chunk_count": max(3, document.chunk_count),
                "error_code": None,
                "error_message": None,
                "updated_at": updated_at,
            }
        )
    return completed


def _advance_active_jobs() -> None:
    for job_id in list(_jobs):
        _advance_job(job_id)


def _answer_turn(
    conversation_id: str,
    body: ConversationTurnCreate,
) -> ConversationTurn:
    existing = next(
        (
            turn
            for turn in _turns.get(conversation_id, [])
            if turn.client_turn_id == body.client_turn_id
        ),
        None,
    )
    if existing is not None:
        if (
            existing.question != body.message
            or existing.top_k != body.top_k
            or existing.document_ids != body.document_ids
        ):
            raise HTTPException(status_code=409, detail="client turn identifier conflict")
        return existing

    turn_id = _next_identifier("turn")
    created_at = NOW + timedelta(minutes=_sequence["turn"])
    turn = ConversationTurn(
        id=turn_id,
        conversation_id=conversation_id,
        client_turn_id=body.client_turn_id,
        status=ConversationTurnStatus.COMPLETED,
        question=body.message,
        answer="The Atlas launch key is cobalt blue [S1].",
        citations=[ATLAS_CITATION],
        no_answer=False,
        top_k=body.top_k,
        document_ids=body.document_ids,
        request_id=f"browser-proof-{turn_id}",
        created_at=created_at,
        updated_at=created_at,
    )
    conversation_turns = _turns.setdefault(conversation_id, [])
    conversation_turns.append(turn)

    conversation = _require_conversation(conversation_id)
    title = conversation.title
    if title == "New conversation":
        normalized_question = " ".join(body.message.split())
        title = (
            normalized_question
            if len(normalized_question) <= 72
            else f"{normalized_question[:69].rstrip()}..."
        )
    _conversations[conversation_id] = conversation.model_copy(
        update={
            "title": title,
            "turn_count": len(conversation_turns),
            "updated_at": created_at,
        }
    )
    return turn


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "alive", "version": "browser-proof"}


@app.get("/health/ready")
def ready() -> dict[str, object]:
    return {
        "status": "ready",
        "checks": {
            "metadata": "ready",
            "qdrant": "ready",
            "vector_inventory": "ready",
            "providers": "ready",
            "worker": "ready",
        },
    }


@app.get("/api/v1/status", dependencies=[Depends(authorize)])
def system_status() -> SystemStatus:
    documents = _visible_documents()
    return SystemStatus(
        status="ready",
        collection="personal_knowledge",
        document_count=len(documents),
        ready_document_count=sum(document.status is DocumentStatus.READY for document in documents),
        chunk_count=sum(document.chunk_count for document in documents),
        queued_job_count=sum(
            job.status in {JobStatus.QUEUED, JobStatus.RETRYING} for job in _jobs.values()
        ),
        embedding_provider="openai",
        embedding_model="text-embedding-3-large",
        embedding_dimensions=3072,
        chat_provider="openai",
        chat_model="gpt-4.1-mini",
        dependencies=[
            DependencyState(name="metadata", status="ready"),
            DependencyState(name="qdrant", status="ready"),
            DependencyState(name="vector_inventory", status="ready"),
            DependencyState(name="providers", status="ready"),
            DependencyState(name="worker", status="ready"),
        ],
        worker_last_seen_at=NOW,
    )


@app.get("/api/v1/documents", dependencies=[Depends(authorize)])
def list_documents(
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    query: Annotated[str | None, Query(alias="q", max_length=200)] = None,
    statuses: Annotated[list[DocumentStatus] | None, Query(alias="status")] = None,
    sort: DocumentSort = DocumentSort.CREATED,
    order: SortOrder = SortOrder.DESC,
) -> DocumentList:
    documents = _visible_documents(sort=sort, order=order)
    if query is not None:
        if any(unicodedata.category(character).startswith("C") for character in query):
            raise HTTPException(
                status_code=422,
                detail="document query contains control characters",
            )
        normalized_query = unicodedata.normalize("NFC", query).strip()
        if normalized_query:
            folded_query = _fold_document_text(normalized_query)
            documents = [
                document
                for document in documents
                if folded_query in _fold_document_text(document.display_name)
                or folded_query in _fold_document_text(document.extension)
            ]
    if statuses:
        accepted = set(statuses)
        documents = [document for document in documents if document.status in accepted]
    return DocumentList(
        items=documents[offset : offset + limit],
        total=len(documents),
        limit=limit,
        offset=offset,
    )


@app.post("/api/v1/documents", response_model=UploadReceipt, status_code=202)
async def upload_document(
    file: Annotated[UploadFile, File()],
    _authorized: Annotated[None, Depends(authorize)],
) -> UploadReceipt:
    filename = Path(file.filename or "upload").name
    extension = Path(filename).suffix.lower()
    document_type = DOCUMENT_TYPES_BY_EXTENSION.get(extension)
    if document_type is None:
        await file.close()
        raise HTTPException(status_code=415, detail="unsupported file type")
    content = await file.read()
    await file.close()
    if not content:
        raise HTTPException(status_code=422, detail="empty file")

    document_id = _next_identifier("document")
    created_at = NOW + timedelta(minutes=_sequence["document"])
    document = DocumentPublic(
        id=document_id,
        display_name=filename,
        content_type=document_type.content_type,
        extension=extension,
        size_bytes=len(content),
        status=DocumentStatus.QUEUED,
        active_version=0,
        chunk_count=0,
        error_code=None,
        error_message=None,
        created_at=created_at,
        updated_at=created_at,
    )
    _documents[document.id] = document
    return UploadReceipt(document=document, job=_new_job(document.id, JobKind.INGEST))


@app.get("/api/v1/documents/{document_id}", dependencies=[Depends(authorize)])
def get_document(document_id: str) -> DocumentPublic:
    return _require_document(document_id)


@app.post(
    "/api/v1/documents/{document_id}/reindex",
    response_model=JobRecord,
    status_code=202,
    dependencies=[Depends(authorize)],
)
def reindex_document(document_id: str) -> JobRecord:
    document = _require_document(document_id)
    existing = next(
        (
            job
            for job in _jobs.values()
            if job.document_id == document_id
            and job.kind is JobKind.REINDEX
            and job.status in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRYING}
        ),
        None,
    )
    if existing is not None:
        return existing
    job = _new_job(document_id, JobKind.REINDEX)
    _documents[document_id] = document.model_copy(
        update={
            "status": DocumentStatus.REINDEXING,
            "error_code": None,
            "error_message": None,
            "updated_at": job.created_at,
        }
    )
    return job


@app.delete(
    "/api/v1/documents/{document_id}",
    response_model=JobRecord,
    status_code=202,
    dependencies=[Depends(authorize)],
)
def delete_document(document_id: str) -> JobRecord:
    document = _require_document(document_id)
    existing = next(
        (
            job
            for job in _jobs.values()
            if job.document_id == document_id
            and job.kind is JobKind.DELETE
            and job.status in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRYING}
        ),
        None,
    )
    if existing is not None:
        return existing
    active = next(
        (
            job
            for job in _jobs.values()
            if job.document_id == document_id
            and job.status in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRYING}
        ),
        None,
    )
    if active is not None and active.status is JobStatus.RUNNING:
        raise HTTPException(status_code=409, detail="document is currently processing")
    if active is not None:
        failed_at = NOW + timedelta(minutes=_sequence["job"], seconds=1)
        _jobs[active.id] = active.model_copy(
            update={
                "status": JobStatus.FAILED,
                "stage": JobStage.FAILED,
                "error_code": "deletion_requested",
                "error_message": "The job was superseded by a document deletion request.",
                "updated_at": failed_at,
                "finished_at": failed_at,
            }
        )
    job = _new_job(document_id, JobKind.DELETE)
    _documents[document_id] = document.model_copy(
        update={"status": DocumentStatus.DELETING, "updated_at": job.created_at}
    )
    return job


@app.get("/api/v1/jobs", response_model=JobList, dependencies=[Depends(authorize)])
def list_jobs(
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    job_status: Annotated[JobStatus | None, Query(alias="status")] = None,
    document_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
) -> JobList:
    _advance_active_jobs()
    jobs = sorted(
        _jobs.values(),
        key=lambda job: (job.created_at, job.id),
        reverse=True,
    )
    if job_status is not None:
        jobs = [job for job in jobs if job.status is job_status]
    if document_id is not None:
        jobs = [job for job in jobs if job.document_id == document_id]
    return JobList(
        items=jobs[offset : offset + limit],
        total=len(jobs),
        limit=limit,
        offset=offset,
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobRecord, dependencies=[Depends(authorize)])
def get_job(job_id: str) -> JobRecord:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _advance_job(job_id)


@app.post(
    "/api/v1/conversations",
    response_model=ConversationSummary,
    status_code=201,
    dependencies=[Depends(authorize)],
)
def create_conversation(body: ConversationCreate) -> ConversationSummary:
    conversation_id = _next_identifier("conversation")
    created_at = NOW + timedelta(minutes=_sequence["conversation"])
    conversation = ConversationSummary(
        id=conversation_id,
        title=body.title or "New conversation",
        turn_count=0,
        created_at=created_at,
        updated_at=created_at,
    )
    _conversations[conversation.id] = conversation
    _turns[conversation.id] = []
    return conversation


@app.get(
    "/api/v1/conversations",
    response_model=ConversationList,
    dependencies=[Depends(authorize)],
)
def list_conversations(
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ConversationList:
    conversations = sorted(
        _conversations.values(),
        key=lambda conversation: (conversation.updated_at, conversation.id),
        reverse=True,
    )
    return ConversationList(
        items=conversations[offset : offset + limit],
        total=len(conversations),
        limit=limit,
        offset=offset,
    )


@app.get(
    "/api/v1/conversations/{conversation_id}/turns",
    response_model=ConversationTurnList,
    dependencies=[Depends(authorize)],
)
def list_conversation_turns(
    conversation_id: str,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ConversationTurnList:
    _require_conversation(conversation_id)
    turns = _turns.get(conversation_id, [])
    return ConversationTurnList(
        items=turns[offset : offset + limit],
        total=len(turns),
        limit=limit,
        offset=offset,
    )


@app.post(
    "/api/v1/conversations/{conversation_id}/turns",
    response_model=ConversationTurn,
    dependencies=[Depends(authorize)],
)
def create_conversation_turn(
    conversation_id: str,
    body: ConversationTurnCreate,
) -> ConversationTurn:
    _require_conversation(conversation_id)
    return _answer_turn(conversation_id, body)


@app.get(
    "/api/v1/conversations/{conversation_id}",
    response_model=ConversationSummary,
    dependencies=[Depends(authorize)],
)
def get_conversation(conversation_id: str) -> ConversationSummary:
    return _require_conversation(conversation_id)


@app.delete(
    "/api/v1/conversations/{conversation_id}",
    status_code=204,
    dependencies=[Depends(authorize)],
)
def delete_conversation(conversation_id: str) -> Response:
    _require_conversation(conversation_id)
    _conversations.pop(conversation_id)
    _turns.pop(conversation_id, None)
    return Response(status_code=204)


@app.post("/api/v1/chat", dependencies=[Depends(authorize)])
def chat(_request: ChatRequest) -> ChatResponse:
    return ChatResponse(
        answer="The Atlas launch key is cobalt blue [S1].",
        citations=[ATLAS_CITATION],
        no_answer=False,
        request_id="browser-proof-request",
    )
