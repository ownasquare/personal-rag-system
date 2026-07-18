from __future__ import annotations

from pathlib import Path

import pytest

from personal_rag.errors import RagError
from personal_rag.storage import managed_upload_key, resolve_managed_upload_path


def test_managed_upload_key_is_portable() -> None:
    assert managed_upload_key("doc-atlas", ".MD") == "doc-atlas.md"


@pytest.mark.parametrize(
    ("document_id", "extension"),
    [("../doc-atlas", ".md"), ("doc-atlas", ".tar.gz")],
)
def test_managed_upload_key_rejects_unsafe_components(document_id: str, extension: str) -> None:
    with pytest.raises(ValueError, match="managed upload storage"):
        managed_upload_key(document_id, extension)


def test_resolver_rebases_legacy_absolute_path_under_current_root(tmp_path: Path) -> None:
    current_uploads = tmp_path / "restored" / "uploads"

    resolved = resolve_managed_upload_path(
        current_uploads,
        "/old/runtime/data/uploads/doc-atlas.md",
        document_id="doc-atlas",
        extension=".md",
    )

    assert resolved == current_uploads.resolve() / "doc-atlas.md"


@pytest.mark.parametrize(
    "stored_path",
    ["../doc-atlas.md", "nested/doc-atlas.md", "/old/uploads/another.md"],
)
def test_resolver_rejects_nonmanaged_paths(tmp_path: Path, stored_path: str) -> None:
    with pytest.raises(RagError, match="managed document key"):
        resolve_managed_upload_path(
            tmp_path / "uploads",
            stored_path,
            document_id="doc-atlas",
            extension=".md",
        )


def test_resolver_uses_canonical_configured_upload_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    uploads = tmp_path / "uploads"
    uploads.symlink_to(outside, target_is_directory=True)

    resolved = resolve_managed_upload_path(
        uploads,
        "doc-atlas.md",
        document_id="doc-atlas",
        extension=".md",
    )

    assert resolved == outside / "doc-atlas.md"


def test_resolver_rejects_managed_file_symlink_escape(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (uploads / "doc-atlas.md").symlink_to(outside)

    with pytest.raises(RagError, match="outside the managed data directory"):
        resolve_managed_upload_path(
            uploads,
            "doc-atlas.md",
            document_id="doc-atlas",
            extension=".md",
        )
