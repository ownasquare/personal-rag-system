"""Durable ingestion/deletion job readback."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request

from personal_rag.errors import RagError
from personal_rag.models import JobRecord
from personal_rag.repository import Repository

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobRecord)
def get_job(job_id: str, request: Request) -> JobRecord:
    repository = cast(Repository, request.app.state.container.repository)
    job = repository.get_job(job_id)
    if job is None:
        raise RagError("job_not_found", "The requested job does not exist.", status_code=404)
    return job
