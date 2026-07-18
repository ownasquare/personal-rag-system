from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import personal_rag.api.routes.documents as document_routes
import personal_rag.parsers as parsers
from personal_rag.document_types import (
    DOCUMENT_TYPE_SPECS,
    DOCUMENT_TYPES_BY_EXTENSION,
    SUPPORTED_DOCUMENT_TYPES_LABEL,
    SUPPORTED_EXTENSIONS,
    UI_UPLOAD_SUFFIXES,
)


def test_document_type_registry_is_normalized_complete_and_shared() -> None:
    expected_content_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".txt": "text/plain",
    }

    assert tuple(DOCUMENT_TYPES_BY_EXTENSION) == tuple(expected_content_types)
    assert {
        extension: spec.content_type for extension, spec in DOCUMENT_TYPES_BY_EXTENSION.items()
    } == expected_content_types
    assert frozenset(expected_content_types) == SUPPORTED_EXTENSIONS
    assert UI_UPLOAD_SUFFIXES == ("docx", "markdown", "md", "pdf", "txt")
    assert tuple(sorted(UI_UPLOAD_SUFFIXES)) == UI_UPLOAD_SUFFIXES
    assert SUPPORTED_DOCUMENT_TYPES_LABEL == "PDF, DOCX, Markdown, and text"
    assert document_routes.DOCUMENT_TYPES_BY_EXTENSION is DOCUMENT_TYPES_BY_EXTENSION
    assert parsers.SUPPORTED_EXTENSIONS is SUPPORTED_EXTENSIONS


def test_document_type_specs_and_registry_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        DOCUMENT_TYPE_SPECS[0].label = "Changed"  # type: ignore[misc]

    with pytest.raises(TypeError):
        DOCUMENT_TYPES_BY_EXTENSION[".csv"] = DOCUMENT_TYPE_SPECS[0]  # type: ignore[index]
