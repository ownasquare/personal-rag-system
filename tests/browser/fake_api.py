"""No-network FastAPI fixture used only for rendered Streamlit browser proof."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from personal_rag.models import (
    ChatRequest,
    ChatResponse,
    Citation,
    DependencyState,
    DocumentList,
    DocumentPublic,
    DocumentStatus,
    SystemStatus,
)

TOKEN = "browser-proof-token-long-enough"
NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)
bearer = HTTPBearer(auto_error=False)
app = FastAPI(title="Personal RAG browser fixture")


def authorize(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> None:
    if credentials is None or credentials.credentials != TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")


DOCUMENT = DocumentPublic(
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
    created_at=NOW,
    updated_at=NOW,
)


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
def status() -> SystemStatus:
    return SystemStatus(
        status="ready",
        collection="personal_knowledge",
        document_count=1,
        ready_document_count=1,
        chunk_count=4,
        queued_job_count=0,
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
def documents(limit: int = 50, offset: int = 0) -> DocumentList:
    items = [DOCUMENT] if offset == 0 and limit > 0 else []
    return DocumentList(items=items, total=1, limit=limit, offset=offset)


@app.post("/api/v1/chat", dependencies=[Depends(authorize)])
def chat(_request: ChatRequest) -> ChatResponse:
    return ChatResponse(
        answer="The Atlas launch key is cobalt blue [S1].",
        citations=[
            Citation(
                label="S1",
                document_id=DOCUMENT.id,
                chunk_id="browser-chunk-1",
                document_name=DOCUMENT.display_name,
                section="Launch checklist",
                snippet="The Atlas launch key is cobalt blue.",
                score=0.94,
            )
        ],
        no_answer=False,
        request_id="browser-proof-request",
    )
