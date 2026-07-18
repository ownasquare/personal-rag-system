from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import tarfile
from pathlib import Path
from typing import cast

import pytest

from personal_rag.database import Database
from personal_rag.models import Citation
from personal_rag.repository import Repository
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


def test_backup_and_restore_preserves_conversation_turns(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    repository = Repository(Database(data_dir / "personal_rag.sqlite3"))
    repository.initialize()
    receipt = repository.create_document_with_job(
        document_id="doc-1",
        display_name="Notes.md",
        stored_path="doc-1.md",
        content_type="text/markdown",
        extension=".md",
        content_sha256="a" * 64,
        size_bytes=42,
        embedding_fingerprint="b" * 64,
        idempotency_key="backup-conversation-document",
    )
    with repository.database.transaction(immediate=True) as connection:
        connection.execute(
            """
            UPDATE documents
            SET status = 'ready', active_version = 1, chunk_count = 1
            WHERE id = ?
            """,
            (receipt.document.id,),
        )
    uploads = data_dir / "uploads"
    uploads.mkdir()
    (uploads / "doc-1.md").write_text("Cobalt.", encoding="utf-8")
    conversation = repository.create_conversation("Atlas")
    reservation = repository.reserve_conversation_turn(
        conversation.id,
        client_turn_id="client-turn-001",
        question="What is the key?",
        top_k=5,
        document_ids=["doc-1"],
        request_fingerprint="a" * 64,
    )
    repository.complete_conversation_turn(
        reservation.turn.id,
        reservation_token=reservation.reservation_token or "missing-reservation-token",
        answer="Cobalt [S1].",
        citations=[
            Citation(
                label="S1",
                document_id="doc-1",
                chunk_id="chunk-1",
                document_name="Notes.md",
                snippet="Cobalt.",
            )
        ],
        no_answer=False,
        request_id="request-1",
    )

    archive = create_backup(data_dir, tmp_path / "backup.tar.gz")
    restored = restore_backup(archive, tmp_path / "restored")
    restored_repository = Repository(Database(restored / "personal_rag.sqlite3"))

    turns = restored_repository.list_conversation_turns(conversation.id)
    assert len(turns) == 1
    assert turns[0].answer == "Cobalt [S1]."
    assert turns[0].citations[0].snippet == "Cobalt."


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
