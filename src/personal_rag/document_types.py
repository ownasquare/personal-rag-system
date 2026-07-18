"""Shared, immutable document-type contracts for uploads, parsing, and the UI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class DocumentTypeSpec:
    """One user-facing document type and its accepted filename extensions."""

    label: str
    extensions: tuple[str, ...]
    content_type: str

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("document type label must not be empty")
        if not self.content_type.strip() or "/" not in self.content_type:
            raise ValueError("document content type must be a canonical MIME type")
        if not self.extensions:
            raise ValueError("document type must define at least one extension")
        if len(set(self.extensions)) != len(self.extensions):
            raise ValueError("document type extensions must be unique")
        for extension in self.extensions:
            if extension != extension.strip().lower() or not extension.startswith("."):
                raise ValueError("document type extensions must be normalized dot suffixes")


DOCUMENT_TYPE_SPECS: tuple[DocumentTypeSpec, ...] = (
    DocumentTypeSpec(
        label="PDF",
        extensions=(".pdf",),
        content_type="application/pdf",
    ),
    DocumentTypeSpec(
        label="DOCX",
        extensions=(".docx",),
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    DocumentTypeSpec(
        label="Markdown",
        extensions=(".md", ".markdown"),
        content_type="text/markdown",
    ),
    DocumentTypeSpec(
        label="text",
        extensions=(".txt",),
        content_type="text/plain",
    ),
)


def _index_specs(specs: tuple[DocumentTypeSpec, ...]) -> Mapping[str, DocumentTypeSpec]:
    indexed: dict[str, DocumentTypeSpec] = {}
    for spec in specs:
        for extension in spec.extensions:
            if extension in indexed:
                raise ValueError(f"duplicate document extension: {extension}")
            indexed[extension] = spec
    return MappingProxyType(indexed)


DOCUMENT_TYPES_BY_EXTENSION: Mapping[str, DocumentTypeSpec] = _index_specs(DOCUMENT_TYPE_SPECS)
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(DOCUMENT_TYPES_BY_EXTENSION)
UI_UPLOAD_SUFFIXES: tuple[str, ...] = tuple(
    sorted(extension.removeprefix(".") for extension in SUPPORTED_EXTENSIONS)
)
SUPPORTED_DOCUMENT_TYPES_LABEL: str = (
    ", ".join(spec.label for spec in DOCUMENT_TYPE_SPECS[:-1])
    + f", and {DOCUMENT_TYPE_SPECS[-1].label}"
)


__all__ = [
    "DOCUMENT_TYPES_BY_EXTENSION",
    "DOCUMENT_TYPE_SPECS",
    "SUPPORTED_DOCUMENT_TYPES_LABEL",
    "SUPPORTED_EXTENSIONS",
    "UI_UPLOAD_SUFFIXES",
    "DocumentTypeSpec",
]
