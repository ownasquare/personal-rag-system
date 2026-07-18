#!/usr/bin/env python3
"""Safely restore and verify an offline Personal RAG backup."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from personal_rag.storage import managed_upload_key

MAX_ARCHIVE_MEMBERS = 10_000
MAX_UNCOMPRESSED_BYTES = 5 * 1024 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_members(archive: tarfile.TarFile) -> None:
    members = archive.getmembers()
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise ValueError("Backup contains too many archive members")
    total_size = 0
    seen: set[str] = set()
    for member in members:
        candidate = PurePosixPath(member.name)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"Unsafe archive member: {member.name}")
        if not candidate.parts or candidate.parts[0] != "personal-rag-backup":
            raise ValueError(f"Archive member is outside the backup root: {member.name}")
        normalized = candidate.as_posix()
        if normalized in seen:
            raise ValueError(f"Duplicate archive member: {member.name}")
        seen.add(normalized)
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"Only regular files and directories are allowed: {member.name}")
        total_size += member.size
        if total_size > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("Backup exceeds the uncompressed restore limit")


def verify_manifest(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported or malformed backup manifest")
    expected_files: dict[str, str] = {}
    for relative_name, expected in manifest["files"].items():
        if not isinstance(relative_name, str) or not isinstance(expected, str):
            raise ValueError("Backup manifest file entries must be strings")
        candidate = PurePosixPath(relative_name)
        if (
            not candidate.parts
            or candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.as_posix() == "manifest.json"
        ):
            raise ValueError(f"Unsafe backup manifest path: {relative_name}")
        if len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected
        ):
            raise ValueError(f"Invalid backup manifest digest: {relative_name}")
        normalized = candidate.as_posix()
        expected_files[normalized] = expected
        path = root.joinpath(*candidate.parts)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Backup integrity check failed for: {relative_name}")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_files != set(expected_files):
        raise ValueError("Backup contents do not exactly match the manifest inventory")


def validate_target(path: Path) -> Path:
    target = path.expanduser().resolve()
    if target in {Path("/").resolve(), Path.home().resolve()}:
        raise ValueError("Refusing to restore into a broad system or home directory")
    if target.exists() and any(target.iterdir()):
        raise ValueError("Restore target must not exist or must be empty")
    return target


def normalize_document_storage_keys(database_path: Path, uploads_dir: Path) -> int:
    """Rewrite legacy absolute upload paths as portable managed storage keys."""

    connection = sqlite3.connect(database_path)
    try:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'documents'"
        ).fetchone()
        if table is None:
            return 0
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(documents)").fetchall()
        }
        required = {"id", "extension", "stored_path", "status"}
        if not required.issubset(columns):
            raise ValueError("Documents table is missing required storage columns")

        updates: list[tuple[str, str]] = []
        rows = connection.execute(
            "SELECT id, extension, stored_path, status FROM documents"
        ).fetchall()
        for document_id, extension, stored_path, status in rows:
            if not all(
                isinstance(value, str) for value in (document_id, extension, stored_path, status)
            ):
                raise ValueError("Document storage metadata contains non-text values")
            try:
                storage_key = managed_upload_key(document_id, extension)
            except ValueError as exc:
                raise ValueError("Document storage metadata contains an unsafe identifier") from exc
            persisted = Path(stored_path)
            if (not persisted.is_absolute() and persisted.parts != (storage_key,)) or (
                persisted.is_absolute() and persisted.name != storage_key
            ):
                raise ValueError(f"Document has an unsafe stored path: {document_id}")
            if status != "deleted" and not (uploads_dir / storage_key).is_file():
                raise ValueError(f"Document source upload is missing: {document_id}")
            if stored_path != storage_key:
                updates.append((storage_key, document_id))

        with connection:
            connection.executemany(
                "UPDATE documents SET stored_path = ? WHERE id = ?",
                updates,
            )
        return len(updates)
    finally:
        connection.close()


def restore_backup(archive_path: Path, target_dir: Path) -> Path:
    archive_file = archive_path.expanduser().resolve()
    if not archive_file.is_file():
        raise FileNotFoundError(f"Backup does not exist: {archive_file}")
    target = validate_target(target_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="personal-rag-restore-", dir=target.parent) as temp:
        temporary_root = Path(temp)
        with tarfile.open(archive_file, "r:gz") as archive:
            validate_members(archive)
            archive.extractall(temporary_root, filter="data")
        extracted = temporary_root / "personal-rag-backup"
        verify_manifest(extracted)
        if not (extracted / "personal_rag.sqlite3").is_file():
            raise ValueError("Backup does not contain the metadata database")
        normalize_document_storage_keys(
            extracted / "personal_rag.sqlite3",
            extracted / "uploads",
        )
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        for source in extracted.iterdir():
            if source.name != "manifest.json":
                shutil.move(str(source), target / source.name)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore a verified offline backup into an empty data directory."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument(
        "--confirm",
        choices=["RESTORE"],
        required=True,
        help="Explicit destructive-action confirmation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = restore_backup(args.archive, args.target_dir)
    print(f"Backup restored and verified: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
