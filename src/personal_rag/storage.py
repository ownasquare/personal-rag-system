"""Managed upload-key construction and runtime path resolution."""

from __future__ import annotations

import re
from pathlib import Path

from personal_rag.errors import RagError

_DOCUMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_EXTENSION_PATTERN = re.compile(r"^\.[A-Za-z0-9]{1,16}$")


def managed_upload_key(document_id: str, extension: str) -> str:
    """Return the portable, single-component key for one managed upload."""

    if _DOCUMENT_ID_PATTERN.fullmatch(document_id) is None:
        raise ValueError("document_id is not safe for managed upload storage")
    normalized_extension = extension.lower()
    if _EXTENSION_PATTERN.fullmatch(normalized_extension) is None:
        raise ValueError("extension is not safe for managed upload storage")
    return f"{document_id}{normalized_extension}"


def resolve_managed_upload_path(
    uploads_dir: Path,
    stored_path: str,
    *,
    document_id: str,
    extension: str,
) -> Path:
    """Resolve a persisted key under the current data root.

    Older databases persisted an absolute path. Its directory is deliberately
    ignored after validating the expected filename, so moved data directories
    remain usable without ever authorizing access outside the current root.
    """

    try:
        expected_key = managed_upload_key(document_id, extension)
    except ValueError as exc:
        raise RagError(
            "unsafe_stored_path",
            "The stored upload identifier is invalid.",
            status_code=409,
        ) from exc

    persisted = Path(stored_path)
    if (not persisted.is_absolute() and persisted.parts != (expected_key,)) or (
        persisted.is_absolute() and persisted.name != expected_key
    ):
        raise RagError(
            "unsafe_stored_path",
            "The stored upload path is not a managed document key.",
            status_code=409,
        )

    uploads_root = uploads_dir.resolve()
    candidate = (uploads_root / expected_key).resolve(strict=False)
    try:
        candidate.relative_to(uploads_root)
    except ValueError as exc:
        raise RagError(
            "unsafe_stored_path",
            "The stored upload path is outside the managed data directory.",
            status_code=409,
        ) from exc
    return candidate
