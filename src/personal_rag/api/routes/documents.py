"""Upload and document-library lifecycle endpoints."""

from __future__ import annotations

import hashlib
import os
import unicodedata
import uuid
from pathlib import Path
from typing import Annotated, cast

import anyio
from fastapi import APIRouter, File, Header, Query, Request, UploadFile

from personal_rag.document_types import (
    DOCUMENT_TYPES_BY_EXTENSION,
    SUPPORTED_DOCUMENT_TYPES_LABEL,
)
from personal_rag.errors import RagError
from personal_rag.models import (
    DocumentList,
    DocumentPublic,
    DocumentSort,
    DocumentStatus,
    JobRecord,
    SortOrder,
    UploadReceipt,
)
from personal_rag.parsers import safe_display_name
from personal_rag.repository import Repository
from personal_rag.security import idempotency_key_digest, validate_idempotency_key
from personal_rag.storage import managed_upload_key

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_DOCUMENT_QUERY_CHARACTERS = 200


def _normalize_document_query(value: str | None) -> str | None:
    """Normalize optional metadata search without retaining invisible control text."""

    if value is None:
        return None
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise RagError(
            "invalid_document_query",
            "Search text contains unsupported characters.",
            status_code=422,
        )
    normalized = unicodedata.normalize("NFC", value).strip()
    if not normalized:
        return None
    return normalized


async def _stream_upload(
    upload: UploadFile, destination: Path, *, max_bytes: int, read_bytes: int
) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    async with await anyio.open_file(destination, "wb") as output:
        while chunk := await upload.read(read_bytes):
            size += len(chunk)
            if size > max_bytes:
                raise RagError(
                    "file_too_large",
                    "The file exceeds the configured upload limit.",
                    status_code=413,
                )
            digest.update(chunk)
            await output.write(chunk)
    if size == 0:
        raise RagError("empty_file", "The uploaded file is empty.", status_code=422)
    os.chmod(destination, 0o600)
    return size, digest.hexdigest()


@router.post("", response_model=UploadReceipt, status_code=202)
async def upload_document(
    request: Request,
    file: Annotated[
        UploadFile,
        File(description=f"Supported types: {SUPPORTED_DOCUMENT_TYPES_LABEL}"),
    ],
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> UploadReceipt:
    container = request.app.state.container
    repository = cast(Repository, container.repository)
    settings = container.settings
    filename = safe_display_name(file.filename or "upload")
    extension = Path(filename).suffix.lower()
    document_type = DOCUMENT_TYPES_BY_EXTENSION.get(extension)
    if document_type is None:
        raise RagError(
            "unsupported_file_type",
            f"Use one of the supported document types: {SUPPORTED_DOCUMENT_TYPES_LABEL}.",
            status_code=415,
        )

    document_id = uuid.uuid4().hex
    staging = settings.staging_dir / f"{document_id}.part"
    storage_key = managed_upload_key(document_id, extension)
    stored = settings.uploads_dir / storage_key
    settings.ensure_directories()
    try:
        size_bytes, content_hash = await _stream_upload(
            file,
            staging,
            max_bytes=settings.upload_max_bytes,
            read_bytes=settings.upload_read_bytes,
        )
        os.replace(staging, stored)
        receipt = repository.create_document_with_job(
            document_id=document_id,
            display_name=filename,
            stored_path=storage_key,
            content_type=document_type.content_type,
            extension=extension,
            content_sha256=content_hash,
            size_bytes=size_bytes,
            embedding_fingerprint=settings.embedding_profile.fingerprint,
            idempotency_key=idempotency_key_digest(validate_idempotency_key(idempotency_key)),
        )
        if receipt.duplicate:
            stored.unlink(missing_ok=True)
        return receipt
    except Exception:
        staging.unlink(missing_ok=True)
        if stored.exists() and repository.get_document(document_id) is None:
            stored.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@router.get("", response_model=DocumentList)
def list_documents(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    q: Annotated[str | None, Query(max_length=MAX_DOCUMENT_QUERY_CHARACTERS)] = None,
    status: Annotated[list[DocumentStatus] | None, Query()] = None,
    sort: DocumentSort = DocumentSort.CREATED,
    order: SortOrder = SortOrder.DESC,
) -> DocumentList:
    repository = cast(Repository, request.app.state.container.repository)
    query = _normalize_document_query(q)
    statuses = list(dict.fromkeys(status or [])) or None
    records = repository.list_documents(
        limit=limit,
        offset=offset,
        query=query,
        statuses=statuses,
        sort=sort,
        order=order,
    )
    return DocumentList(
        items=[DocumentPublic.from_record(record) for record in records],
        total=repository.count_documents(query=query, statuses=statuses),
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentPublic)
def get_document(document_id: str, request: Request) -> DocumentPublic:
    repository = cast(Repository, request.app.state.container.repository)
    record = repository.get_document(document_id)
    if record is None or record.status is DocumentStatus.DELETED:
        raise RagError(
            "document_not_found", "The requested document does not exist.", status_code=404
        )
    return DocumentPublic.from_record(record)


@router.post("/{document_id}/reindex", response_model=JobRecord, status_code=202)
def reindex_document(document_id: str, request: Request) -> JobRecord:
    repository = cast(Repository, request.app.state.container.repository)
    job = repository.request_reindex(document_id)
    if job is None:
        raise RagError(
            "document_not_found", "The requested document does not exist.", status_code=404
        )
    return job


@router.delete("/{document_id}", response_model=JobRecord, status_code=202)
def delete_document(document_id: str, request: Request) -> JobRecord:
    repository = cast(Repository, request.app.state.container.repository)
    job = repository.request_delete(document_id)
    if job is None:
        raise RagError(
            "document_not_found", "The requested document does not exist.", status_code=404
        )
    return job
