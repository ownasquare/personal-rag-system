"""Bounded, signature-aware document extraction into LlamaIndex documents."""

from __future__ import annotations

import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

from docx import Document as open_docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from llama_index.core import Document
from pypdf import PdfReader

from personal_rag.config import Settings
from personal_rag.errors import RagError

_SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".md", ".markdown", ".txt"})
_PDF_SIGNATURE = b"%PDF-"
_ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_OLE_SIGNATURE = bytes.fromhex("D0CF11E0A1B11AE1")
_HEADING_PATTERN = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE_PATTERN = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_DOCX_HEADING_PATTERN = re.compile(r"^Heading [1-6]$", re.IGNORECASE)


def safe_display_name(value: str, *, maximum_characters: int = 255) -> str:
    """Return a path-free, control-free name suitable only for display."""

    if maximum_characters < 16:
        raise ValueError("maximum_characters must be at least 16")
    basename = str(value).replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    normalized = unicodedata.normalize("NFC", basename)
    normalized = "".join(
        character for character in normalized if unicodedata.category(character) not in {"Cc", "Cf"}
    ).strip(" .\t\r\n")
    if not normalized:
        normalized = "document"
    if len(normalized) <= maximum_characters:
        return normalized

    suffix = Path(normalized).suffix
    if len(suffix) > 32:
        suffix = ""
    stem_characters = maximum_characters - len(suffix)
    shortened = normalized[:stem_characters].rstrip(" .")
    return f"{shortened}{suffix}" or "document"


def validate_file_signature(
    path: Path,
    extension: str,
    *,
    max_docx_uncompressed_bytes: int,
) -> None:
    """Reject common extension spoofing and unsafe Office containers."""

    with path.open("rb") as stream:
        prefix = stream.read(8_192)

    if extension == ".pdf":
        if not prefix.startswith(_PDF_SIGNATURE):
            raise RagError(
                "invalid_file_signature",
                "The file content does not match the PDF extension",
                status_code=415,
            )
        return

    if extension == ".docx":
        if prefix.startswith(_OLE_SIGNATURE):
            raise RagError(
                "encrypted_document",
                "Encrypted or legacy Office documents are not supported",
                status_code=422,
            )
        if not prefix.startswith(_ZIP_SIGNATURES):
            raise RagError(
                "invalid_file_signature",
                "The file content does not match the DOCX extension",
                status_code=415,
            )
        _validate_docx_container(path, max_docx_uncompressed_bytes)
        return

    if prefix.startswith((_PDF_SIGNATURE, *_ZIP_SIGNATURES, _OLE_SIGNATURE)) or b"\x00" in prefix:
        raise RagError(
            "invalid_file_signature",
            "The file content does not match its text-document extension",
            status_code=415,
        )


class DocumentParser:
    """Extract supported files while enforcing configured resource ceilings."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def parse(self, path: Path, display_name: str | None = None) -> list[Document]:
        source_path = Path(path)
        extension = source_path.suffix.lower()
        if extension not in _SUPPORTED_EXTENSIONS:
            raise RagError(
                "unsupported_file_type",
                "Supported document types are PDF, DOCX, Markdown, and plain text",
                status_code=415,
            )
        self._validate_path(source_path)
        name = safe_display_name(display_name or source_path.name)
        max_docx_uncompressed = max(
            8 * 1024 * 1024,
            self.settings.max_extracted_characters * 8,
            self.settings.upload_max_bytes * 4,
        )
        validate_file_signature(
            source_path,
            extension,
            max_docx_uncompressed_bytes=max_docx_uncompressed,
        )

        if extension == ".pdf":
            return self._parse_pdf(source_path, name, extension)
        if extension == ".docx":
            return self._parse_docx(source_path, name, extension)
        if extension in {".md", ".markdown"}:
            text = self._read_utf8_text(source_path)
            return self._parse_markdown(text, name, extension)
        text = self._read_utf8_text(source_path)
        return self._single_text_document(text, name, extension)

    def _validate_path(self, path: Path) -> None:
        if path.is_symlink():
            raise RagError(
                "unsafe_file",
                "Symbolic links cannot be parsed as uploaded documents",
                status_code=400,
            )
        try:
            stat = path.stat()
        except FileNotFoundError as exc:
            raise RagError(
                "file_not_found", "Uploaded document not found", status_code=404
            ) from exc
        if not path.is_file():
            raise RagError("unsafe_file", "Uploaded document is not a regular file")
        if stat.st_size > self.settings.upload_max_bytes:
            raise RagError(
                "file_too_large",
                f"Document exceeds the {self.settings.upload_max_bytes}-byte upload limit",
                status_code=413,
            )
        if stat.st_size == 0:
            raise RagError("empty_document", "The document is empty", status_code=422)

    def _read_utf8_text(self, path: Path) -> str:
        try:
            text = path.read_bytes().decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise RagError(
                "invalid_text_encoding",
                "Text and Markdown documents must use UTF-8 encoding",
                status_code=422,
            ) from exc
        self._check_character_limit(len(text))
        if not text.strip():
            raise RagError("empty_document", "The document has no readable text", status_code=422)
        return text

    def _parse_pdf(self, path: Path, name: str, extension: str) -> list[Document]:
        try:
            reader = PdfReader(path, strict=False)
            if reader.is_encrypted:
                raise RagError(
                    "encrypted_document",
                    "Encrypted PDF documents are not supported",
                    status_code=422,
                )
            page_count = len(reader.pages)
            if page_count > self.settings.max_pdf_pages:
                raise RagError(
                    "page_limit_exceeded",
                    f"PDF exceeds the {self.settings.max_pdf_pages}-page limit",
                    status_code=413,
                )
            documents: list[Document] = []
            total_characters = 0
            found_image = False
            for page_number, page in enumerate(reader.pages, start=1):
                extracted = page.extract_text() or ""
                text = _clean_extracted_text(extracted)
                found_image = found_image or _page_has_images(page)
                if not text:
                    continue
                total_characters += len(text)
                self._check_character_limit(total_characters)
                documents.append(
                    self._document(
                        text,
                        name,
                        extension,
                        len(documents),
                        page_number=page_number,
                    )
                )
        except RagError:
            raise
        except Exception as exc:
            raise RagError(
                "unreadable_document",
                "The PDF could not be read safely",
                status_code=422,
            ) from exc

        if not documents:
            if found_image:
                raise RagError(
                    "image_only_document",
                    "The PDF contains images but no extractable text; OCR is not enabled",
                    status_code=422,
                )
            raise RagError("empty_document", "The PDF has no readable text", status_code=422)
        return documents

    def _parse_docx(self, path: Path, name: str, extension: str) -> list[Document]:
        try:
            source = open_docx(str(path))
            sections: list[tuple[str, str]] = []
            title = "Document"
            lines: list[str] = []
            total_characters = 0

            def append_text(value: str) -> None:
                nonlocal total_characters
                cleaned = _clean_extracted_text(value)
                if not cleaned:
                    return
                total_characters += len(cleaned) + (2 if lines else 0)
                self._check_character_limit(total_characters)
                lines.append(cleaned)

            def flush_section() -> None:
                if lines:
                    sections.append((title, "\n\n".join(lines)))
                    lines.clear()

            for block in source.iter_inner_content():
                if isinstance(block, Paragraph):
                    text = block.text.strip()
                    style_name = block.style.name if block.style is not None else ""
                    if text and _DOCX_HEADING_PATTERN.fullmatch(style_name):
                        flush_section()
                        title = _section_title(text)
                        append_text(text)
                    else:
                        append_text(text)
                elif isinstance(block, Table):
                    for row in block.rows:
                        cells = [_clean_extracted_text(cell.text) for cell in row.cells]
                        if any(cells):
                            append_text(" | ".join(cells))
            flush_section()
        except RagError:
            raise
        except Exception as exc:
            raise RagError(
                "unreadable_document",
                "The DOCX document could not be read safely",
                status_code=422,
            ) from exc

        if not sections:
            raise RagError(
                "empty_document", "The DOCX document has no readable text", status_code=422
            )
        return [
            self._document(text, name, extension, index, section=section)
            for index, (section, text) in enumerate(sections)
        ]

    def _parse_markdown(self, text: str, name: str, extension: str) -> list[Document]:
        sections: list[tuple[str, str]] = []
        title = "Document"
        lines: list[str] = []
        fence_character: str | None = None

        def flush_section() -> None:
            content = "\n".join(lines).strip()
            if content:
                sections.append((title, content))
            lines.clear()

        for line in text.splitlines():
            fence = _FENCE_PATTERN.match(line)
            if fence is not None:
                marker_character = fence.group(1)[0]
                if fence_character is None:
                    fence_character = marker_character
                elif fence_character == marker_character:
                    fence_character = None
                lines.append(line)
                continue
            heading = _HEADING_PATTERN.match(line) if fence_character is None else None
            if heading is not None:
                flush_section()
                title = _section_title(heading.group(2))
                lines.append(line)
            else:
                lines.append(line)
        flush_section()

        if not sections:
            raise RagError(
                "empty_document", "The Markdown document has no readable text", status_code=422
            )
        return [
            self._document(content, name, extension, index, section=section)
            for index, (section, content) in enumerate(sections)
        ]

    def _single_text_document(self, text: str, name: str, extension: str) -> list[Document]:
        content = text.strip()
        return [self._document(content, name, extension, 0, section="Document")]

    def _document(
        self,
        text: str,
        name: str,
        extension: str,
        unit_index: int,
        *,
        page_number: int | None = None,
        section: str | None = None,
    ) -> Document:
        metadata: dict[str, str | int | float | bool] = {
            "source_name": name,
            "source_extension": extension,
            "parser_version": self.settings.parser_version,
            "unit_index": unit_index,
        }
        if page_number is not None:
            metadata["page_number"] = page_number
        if section is not None:
            metadata["section"] = section
        return Document(text=text, metadata=metadata)

    def _check_character_limit(self, character_count: int) -> None:
        if character_count > self.settings.max_extracted_characters:
            raise RagError(
                "character_limit_exceeded",
                "Extracted text exceeds the configured character limit",
                status_code=413,
            )


def _validate_docx_container(path: Path, maximum_uncompressed_bytes: int) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = {entry.filename for entry in entries}
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise RagError(
                    "invalid_file_signature",
                    "The ZIP container is not a valid DOCX document",
                    status_code=415,
                )
            if any(entry.flag_bits & 0x1 for entry in entries):
                raise RagError(
                    "encrypted_document",
                    "Encrypted DOCX documents are not supported",
                    status_code=422,
                )
            if len(entries) > 10_000:
                raise RagError(
                    "document_too_complex",
                    "The DOCX container contains too many entries",
                    status_code=413,
                )
            if sum(entry.file_size for entry in entries) > maximum_uncompressed_bytes:
                raise RagError(
                    "document_too_complex",
                    "The expanded DOCX document exceeds the safe processing limit",
                    status_code=413,
                )
    except RagError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise RagError(
            "invalid_file_signature",
            "The DOCX container is invalid or damaged",
            status_code=415,
        ) from exc


def _clean_extracted_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).replace("\x00", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.strip()


def _section_title(value: str) -> str:
    title = _clean_extracted_text(value)
    return title[:200] if title else "Untitled section"


def _page_has_images(page: Any) -> bool:
    try:
        images = getattr(page, "images", ())
        return len(images) > 0
    except Exception:
        return False
