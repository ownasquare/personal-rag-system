from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import tarfile
from pathlib import Path
from typing import cast

import pytest

from scripts.backup import create_backup
from scripts.restore import MAX_UNCOMPRESSED_BYTES, restore_backup, validate_members


def test_backup_and_restore_round_trip(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    database = data_dir / "personal_rag.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE facts (value TEXT NOT NULL)")
    connection.execute("INSERT INTO facts VALUES ('cobalt')")
    connection.commit()
    connection.close()
    uploads = data_dir / "uploads"
    uploads.mkdir()
    (uploads / "doc.txt").write_text("launch key", encoding="utf-8")

    archive = create_backup(data_dir, tmp_path / "backup.tar.gz")
    restored = restore_backup(archive, tmp_path / "restored")

    restored_connection = sqlite3.connect(restored / "personal_rag.sqlite3")
    try:
        assert restored_connection.execute("SELECT value FROM facts").fetchone() == ("cobalt",)
    finally:
        restored_connection.close()
    assert (restored / "uploads" / "doc.txt").read_text(encoding="utf-8") == "launch key"


def test_backup_refuses_missing_database(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="metadata database"):
        create_backup(tmp_path / "empty", tmp_path / "backup.tar.gz")


def test_backup_refuses_output_inside_copied_data_tree(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    qdrant_dir = data_dir / "qdrant"
    qdrant_dir.mkdir(parents=True)
    sqlite3.connect(data_dir / "personal_rag.sqlite3").close()

    output = qdrant_dir / "backups" / "recursive.tar.gz"
    with pytest.raises(ValueError, match="outside the application data directory"):
        create_backup(data_dir, output)

    assert output.exists() is False
    assert list(qdrant_dir.glob("personal-rag-backup-*")) == []


def test_restore_refuses_nonempty_target(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sqlite3.connect(data_dir / "personal_rag.sqlite3").close()
    archive = create_backup(data_dir, tmp_path / "backup.tar.gz")
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="must be empty"):
        restore_backup(archive, target)
    assert (target / "keep.txt").read_text(encoding="utf-8") == "preserve"


def test_restore_rejects_path_traversal_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "hostile.tar.gz"
    payload = b"unsafe"
    member = tarfile.TarInfo("../outside.txt")
    member.size = len(payload)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="Unsafe archive member"):
        restore_backup(archive_path, tmp_path / "restore")
    assert not (tmp_path / "outside.txt").exists()


def test_restore_rejects_files_missing_from_manifest_inventory(tmp_path: Path) -> None:
    source = tmp_path / "source" / "personal-rag-backup"
    source.mkdir(parents=True)
    database = source / "personal_rag.sqlite3"
    database.write_bytes(b"sqlite-placeholder")
    (source / "unlisted.txt").write_text("must not pass", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "files": {"personal_rag.sqlite3": hashlib.sha256(database.read_bytes()).hexdigest()},
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    archive_path = tmp_path / "unlisted.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source, arcname="personal-rag-backup")

    with pytest.raises(ValueError, match="exactly match"):
        restore_backup(archive_path, tmp_path / "restore")


def test_restore_rejects_unsafe_manifest_paths(tmp_path: Path) -> None:
    source = tmp_path / "source" / "personal-rag-backup"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": {"../outside": "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    archive_path = tmp_path / "unsafe-manifest.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source, arcname="personal-rag-backup")

    with pytest.raises(ValueError, match="Unsafe backup manifest path"):
        restore_backup(archive_path, tmp_path / "restore")


def test_restore_rejects_tar_bomb_size_before_extraction() -> None:
    member = tarfile.TarInfo("personal-rag-backup/huge.bin")
    member.size = MAX_UNCOMPRESSED_BYTES + 1

    class OversizedArchive:
        def getmembers(self) -> list[tarfile.TarInfo]:
            return [member]

    with pytest.raises(ValueError, match="uncompressed restore limit"):
        validate_members(cast(tarfile.TarFile, OversizedArchive()))
