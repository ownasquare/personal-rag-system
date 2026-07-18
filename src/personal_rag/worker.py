"""Lease-driven worker loop with heartbeat and graceful shutdown support."""

from __future__ import annotations

import signal
import socket
import threading
import uuid
from collections.abc import Sequence
from types import FrameType
from typing import Protocol

from personal_rag.config import Settings, get_settings
from personal_rag.database import Database
from personal_rag.job_service import JobProcessor
from personal_rag.models import JobKind, JobRecord
from personal_rag.observability import configure_logging, get_logger
from personal_rag.parsers import DocumentParser
from personal_rag.providers import build_embedding, build_llm
from personal_rag.rag_service import RAGService
from personal_rag.repository import Repository
from personal_rag.vector_store import VectorStore

logger = get_logger("worker").bind(service="worker")


class WorkerRepository(Protocol):
    def lease_next_job(
        self,
        worker_id: str,
        *,
        lease_seconds: int | None = None,
        kinds: Sequence[JobKind] | None = None,
    ) -> JobRecord | None: ...

    def record_worker_heartbeat(self, worker_id: str) -> object: ...

    def heartbeat_job(
        self,
        job_id: str,
        worker_id: str,
        *,
        lease_seconds: int | None = None,
    ) -> JobRecord: ...


class Worker:
    """One cooperative worker that finishes its current job before stopping."""

    def __init__(
        self,
        settings: Settings,
        repository: WorkerRepository,
        processor: JobProcessor,
        *,
        worker_id: str | None = None,
        stop_event: threading.Event | None = None,
        kinds: Sequence[JobKind] | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.processor = processor
        self.worker_id = worker_id or _default_worker_id()
        self.stop_event = stop_event or threading.Event()
        self.kinds = tuple(kinds) if kinds else None

    def run_once(self) -> bool:
        """Lease and process at most one job; return whether work was found."""

        self.repository.record_worker_heartbeat(self.worker_id)
        if self.stop_event.is_set():
            return False
        job = self.repository.lease_next_job(
            self.worker_id,
            lease_seconds=self.settings.job_lease_seconds,
            kinds=self.kinds,
        )
        if job is None:
            return False
        self._process_with_lease_heartbeat(job)
        self.repository.record_worker_heartbeat(self.worker_id)
        return True

    def run_forever(self) -> None:
        """Poll until asked to stop, tolerating transient loop-level failures."""

        while not self.stop_event.is_set():
            try:
                processed = self.run_once()
            except Exception as exc:
                # Avoid logging exception values: upstream SDK messages can contain
                # endpoints or request details. The type is enough for operations.
                logger.error(
                    "worker_iteration_failed",
                    error_type=type(exc).__name__,
                    worker_id=self.worker_id,
                )
                processed = False
            if not processed:
                self.stop_event.wait(self.settings.worker_poll_seconds)

    def stop(self) -> None:
        """Request a graceful stop after the current synchronous job step."""

        self.stop_event.set()

    def install_signal_handlers(self) -> None:
        """Translate SIGINT/SIGTERM into the same graceful stop event."""

        def request_stop(signum: int, frame: FrameType | None) -> None:
            del signum, frame
            self.stop()

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)

    def _process_with_lease_heartbeat(self, job: JobRecord) -> None:
        """Keep a lease live even while one provider call takes several minutes."""

        finished = threading.Event()
        interval = max(0.25, min(30.0, self.settings.job_lease_seconds / 3))

        def maintain_lease() -> None:
            while not finished.wait(interval):
                try:
                    self.repository.heartbeat_job(
                        job.id,
                        self.worker_id,
                        lease_seconds=self.settings.job_lease_seconds,
                    )
                    self.repository.record_worker_heartbeat(self.worker_id)
                except Exception as exc:
                    logger.error(
                        "job_lease_heartbeat_failed",
                        error_type=type(exc).__name__,
                        worker_id=self.worker_id,
                        job_id=job.id,
                        document_id=job.document_id,
                    )
                    return

        keeper = threading.Thread(
            target=maintain_lease,
            name=f"lease-heartbeat-{job.id[:12]}",
            daemon=True,
        )
        keeper.start()
        logger.info(
            "job_processing_started",
            worker_id=self.worker_id,
            job_id=job.id,
            document_id=job.document_id,
            job_kind=job.kind.value,
            stage=job.stage.value,
        )
        try:
            self.processor.process(job)
        except Exception as exc:
            logger.error(
                "job_processing_failed",
                worker_id=self.worker_id,
                job_id=job.id,
                document_id=job.document_id,
                job_kind=job.kind.value,
                stage=job.stage.value,
                error_type=type(exc).__name__,
            )
            raise
        else:
            logger.info(
                "job_processing_finished",
                worker_id=self.worker_id,
                job_id=job.id,
                document_id=job.document_id,
                job_kind=job.kind.value,
            )
        finally:
            finished.set()
            keeper.join(timeout=1.0)


def _default_worker_id() -> str:
    return f"{socket.gethostname()}-{uuid.uuid4().hex[:12]}"


def build_worker(settings: Settings | None = None, *, worker_id: str | None = None) -> Worker:
    """Compose a standalone worker process without importing the HTTP API."""

    resolved = settings or get_settings()
    resolved.ensure_directories()
    repository = Repository(
        Database(resolved.database_path),
        lease_seconds=resolved.job_lease_seconds,
        max_attempts=resolved.job_max_attempts,
    )
    repository.initialize()
    embedding = build_embedding(resolved)
    vector_store = VectorStore(resolved, embedding)
    rag_service = RAGService(
        resolved,
        vector_store,
        embedding,
        build_llm(resolved),
        repository,
    )
    processor = JobProcessor(
        resolved,
        repository,
        DocumentParser(resolved),
        rag_service,
    )
    return Worker(
        resolved,
        repository,
        processor,
        worker_id=worker_id,
    )


def main() -> None:
    """Run the production worker until SIGINT or SIGTERM requests shutdown."""

    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)
    worker = build_worker(settings)
    worker.install_signal_handlers()
    logger.info("worker_started", worker_id=worker.worker_id)
    try:
        worker.run_forever()
    finally:
        logger.info("worker_stopped", worker_id=worker.worker_id)


if __name__ == "__main__":
    main()
