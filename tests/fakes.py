"""Deterministic, offline provider fakes shared by RAG and worker tests."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_TOKENS = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
}


class DeterministicEmbedding:
    """A stable signed hashing vectorizer with no model downloads or network."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions
        self.text_calls = 0
        self.query_calls = 0

    def get_text_embedding_batch(
        self, texts: list[str], *, show_progress: bool = False
    ) -> list[list[float]]:
        del show_progress
        self.text_calls += 1
        return [self._embed(text) for text in texts]

    def get_query_embedding(self, query: str) -> list[float]:
        self.query_calls += 1
        return self._embed(query)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = [token for token in _TOKENS.findall(text.lower()) if token not in _STOP_WORDS]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


@dataclass(slots=True)
class FakeCompletion:
    text: str


class FakeLLM:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def complete(self, prompt: str, **kwargs: object) -> FakeCompletion:
        del kwargs
        self.prompts.append(prompt)
        return FakeCompletion(self.answer)


class RaisingLLM:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def complete(self, prompt: str, **kwargs: object) -> FakeCompletion:
        del prompt, kwargs
        raise self.error


class FakeManifest:
    """A mutable manifest containing only ready document versions."""

    def __init__(self, versions: Mapping[str, int]) -> None:
        self.versions = dict(versions)
        self.calls: list[tuple[str, ...] | None] = []

    def get_ready_document_versions(
        self, document_ids: Sequence[str] | None = None
    ) -> dict[str, int]:
        self.calls.append(None if document_ids is None else tuple(document_ids))
        if document_ids is None:
            return dict(self.versions)
        return {
            document_id: self.versions[document_id]
            for document_id in dict.fromkeys(document_ids)
            if document_id in self.versions
        }


class SequencedManifest(FakeManifest):
    """Return successive manifest snapshots to exercise state-change races."""

    def __init__(self, snapshots: Sequence[Mapping[str, int]]) -> None:
        if not snapshots:
            raise ValueError("at least one manifest snapshot is required")
        super().__init__(snapshots[0])
        self.snapshots = [dict(snapshot) for snapshot in snapshots]

    def get_ready_document_versions(
        self, document_ids: Sequence[str] | None = None
    ) -> dict[str, int]:
        snapshot = self.snapshots[min(len(self.calls), len(self.snapshots) - 1)]
        self.versions = snapshot
        return super().get_ready_document_versions(document_ids)
