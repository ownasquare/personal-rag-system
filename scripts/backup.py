#!/usr/bin/env python3
"""Create an offline, integrity-manifested backup of local Personal RAG data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_data_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve()}
    if resolved in forbidden:
        raise ValueError("Refusing to back up a broad system or home directory")
    database = resolved / "personal_rag.sqlite3"
    if not database.is_file():
        raise ValueError(f"Expected metadata database is missing: {database}")
    return resolved


def sqlite_snapshot(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()


def create_backup(data_dir: Path, output: Path) -> Path:
    source = validate_data_dir(data_dir)
    destination = output.expanduser().resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError("Backup output must be outside the application data directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Backup already exists: {destination}")

    with tempfile.TemporaryDirectory(prefix="personal-rag-backup-", dir=destination.parent) as temp:
        snapshot_root = Path(temp) / "personal-rag-backup"
        snapshot_root.mkdir(mode=0o700)
        sqlite_snapshot(source / "personal_rag.sqlite3", snapshot_root / "personal_rag.sqlite3")
        for directory_name in ("uploads", "qdrant"):
            candidate = source / directory_name
            if candidate.is_dir():
                shutil.copytree(candidate, snapshot_root / directory_name, symlinks=False)

        files = sorted(path for path in snapshot_root.rglob("*") if path.is_file())
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "files": {str(path.relative_to(snapshot_root)): sha256_file(path) for path in files},
        }
        (snapshot_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        temporary_archive = destination.with_suffix(destination.suffix + ".partial")
        with tarfile.open(temporary_archive, "w:gz") as archive:
            archive.add(snapshot_root, arcname="personal-rag-backup", recursive=True)
        os.chmod(temporary_archive, 0o600)
        temporary_archive.replace(destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an offline backup after API, worker, and Qdrant are stopped."
    )
    parser.add_argument("--data-dir", type=Path, default=Path(".data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--acknowledge-services-stopped",
        action="store_true",
        help="Required acknowledgement that all writers and Qdrant are stopped.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.acknowledge_services_stopped:
        raise SystemExit("Stop API, worker, and Qdrant, then pass --acknowledge-services-stopped")
    result = create_backup(args.data_dir, args.output)
    print(f"Backup created: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
