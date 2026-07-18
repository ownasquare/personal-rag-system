"""Durable ingestion/deletion job readback."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import APIRouter, Query, Request

from personal_rag.errors import RagError
from personal_rag.models import JobList, JobRecord, JobStatus
from personal_rag.repository import Repository

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=JobList)
def list_jobs(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    status: JobStatus | None = None,
    document_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
) -> JobList:
    repository = cast(Repository, request.app.state.container.repository)
    return JobList(
        items=repository.list_jobs(
            limit=limit,
            offset=offset,
            status=status,
            document_id=document_id,
        ),
        total=repository.count_jobs(status=status, document_id=document_id),
        limit=limit,
        offset=offset,
    )


@router.get("/{job_id}", response_model=JobRecord)
def get_job(job_id: str, request: Request) -> JobRecord:
    repository = cast(Repository, request.app.state.container.repository)
    job = repository.get_job(job_id)
    if job is None:
        raise RagError("job_not_found", "The requested job does not exist.", status_code=404)
    return job
