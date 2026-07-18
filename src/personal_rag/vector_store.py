"""Qdrant lifecycle, LlamaIndex integration, retrieval, and deletion proof."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    FilterCondition,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from personal_rag.config import Settings
from personal_rag.errors import ConfigurationError, RagError

type MetadataScalar = str | int | float | bool

_DENSE_VECTOR_NAME = "personal-rag-dense"
_SCROLL_BATCH_SIZE = 256


@dataclass(frozen=True, slots=True)
class VectorNode:
    """One fully embedded chunk ready for durable storage."""

    id: str
    text: str
    embedding: Sequence[float]
    metadata: Mapping[str, MetadataScalar]


@dataclass(frozen=True, slots=True)
class RetrievedNode:
    """A retrieved chunk whose score and metadata came from the backend."""

    id: str
    text: str
    metadata: Mapping[str, MetadataScalar]
    score: float


def build_qdrant_client(settings: Settings) -> QdrantClient:
    """Construct server, persistent-local, or in-memory Qdrant explicitly."""

    mode = settings.qdrant_mode
    if mode == "http":
        api_key_value = settings.qdrant_api_key
        api_key = api_key_value.get_secret_value() if api_key_value is not None else None
        return QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            https=settings.qdrant_https,
            api_key=api_key,
            timeout=math.ceil(settings.provider_timeout_seconds),
            check_compatibility=True,
        )
    if mode == "persistent":
        persist_dir = Path(settings.qdrant_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        return QdrantClient(path=str(persist_dir))
    if mode == "memory":
        return QdrantClient(location=":memory:")
    raise ConfigurationError(f"Unsupported Qdrant mode: {mode}")


class VectorStore:
    """Profile-locked Qdrant adapter backed by LlamaIndex's official integration."""

    def __init__(
        self,
        settings: Settings,
        embedding: object | None = None,
        *,
        client: QdrantClient | None = None,
    ) -> None:
        self.settings = settings
        # Models remain explicit at the composition root. Qdrant never embeds
        # implicitly; RAGService supplies every stored and query vector.
        self.embedding = embedding
        self.mode = settings.qdrant_mode
        self.client = client or build_qdrant_client(settings)
        self.collection_name = settings.qdrant_collection
        self._open_collection_with_retry()
        self.integration = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            dense_config=models.VectorParams(
                size=settings.embedding_dimensions,
                distance=models.Distance.COSINE,
            ),
            dense_vector_name=_DENSE_VECTOR_NAME,
            enable_hybrid=False,
            index_doc_id=False,
            flat_metadata=True,
            batch_size=64,
            parallel=1,
            max_retries=settings.provider_max_retries,
        )

    def _open_collection_with_retry(self) -> None:
        """Bound only HTTP connection/5xx startup races; never retry profile errors."""

        if self.mode != "http":
            self._open_collection()
            return
        retries = self.settings.provider_max_retries
        delay_seconds = 0.25
        for attempt in range(retries + 1):
            try:
                self._open_collection()
                return
            except ConfigurationError:
                raise
            except Exception as exc:
                if attempt >= retries or not _retryable_qdrant_startup_error(exc):
                    raise
                time.sleep(delay_seconds)
                delay_seconds = min(2.0, delay_seconds * 2)

    def _profile_metadata(self) -> dict[str, MetadataScalar]:
        profile = self.settings.embedding_profile
        return {
            "rag_embedding_fingerprint": profile.fingerprint,
            "rag_embedding_provider": profile.provider,
            "rag_embedding_model": profile.model,
            "rag_embedding_dimensions": profile.dimensions,
            "rag_distance_metric": profile.distance_metric,
            "rag_parser_version": profile.parser_version,
            "rag_chunker": profile.chunker,
            "rag_chunk_size": profile.chunk_size,
            "rag_chunk_overlap": profile.chunk_overlap,
        }

    def _open_collection(self) -> None:
        expected = self._profile_metadata()
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    _DENSE_VECTOR_NAME: models.VectorParams(
                        size=self.settings.embedding_dimensions,
                        distance=models.Distance.COSINE,
                    )
                },
                metadata=expected,
            )

        info = self.client.get_collection(self.collection_name)
        actual = info.config.metadata or {}
        if actual.get("rag_embedding_fingerprint") != expected["rag_embedding_fingerprint"]:
            raise ConfigurationError(
                "The Qdrant collection embedding profile does not match this runtime. "
                "Restore the matching settings or reindex into a new collection."
            )
        vectors = info.config.params.vectors
        dense = vectors.get(_DENSE_VECTOR_NAME) if isinstance(vectors, dict) else None
        if (
            dense is None
            or dense.size != self.settings.embedding_dimensions
            or dense.distance is not models.Distance.COSINE
        ):
            raise ConfigurationError(
                "The Qdrant collection vector configuration does not match this runtime."
            )
        if self.mode == "http":
            document_id_index = info.payload_schema.get("document_id")
            if (
                document_id_index is not None
                and document_id_index.data_type is not models.PayloadSchemaType.KEYWORD
            ):
                raise ConfigurationError(
                    "The Qdrant document_id payload index has an incompatible type."
                )
            if document_id_index is None:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            verified = self.client.get_collection(self.collection_name).payload_schema.get(
                "document_id"
            )
            if verified is None or verified.data_type is not models.PayloadSchemaType.KEYWORD:
                raise ConfigurationError(
                    "The Qdrant document_id payload index could not be verified."
                )

    def heartbeat(self) -> int:
        """Prove the configured Qdrant service is reachable."""

        self.client.get_collections()
        return time.time_ns()

    def close(self) -> None:
        """Release HTTP transports or persistent-local file locks."""

        self.client.close()

    def count(self, *, document_id: str | None = None) -> int:
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=(self._document_filter(document_id) if document_id is not None else None),
            exact=True,
        )
        return int(result.count)

    def get_document_node_ids(self, document_id: str) -> list[str]:
        """Read every backend point ID for one document with complete pagination."""

        point_ids: list[str] = []
        offset: Any = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._document_filter(document_id),
                limit=_SCROLL_BATCH_SIZE,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            point_ids.extend(str(record.id) for record in records)
            if next_offset is None:
                break
            offset = next_offset
        return point_ids

    def upsert_nodes(self, nodes: Sequence[VectorNode]) -> None:
        """Upsert complete embeddings through the official LlamaIndex adapter."""

        if not nodes:
            return
        dimensions = self.settings.embedding_dimensions
        text_nodes: list[TextNode] = []
        for node in nodes:
            if len(node.embedding) != dimensions:
                raise RagError(
                    "embedding_dimension_mismatch",
                    "The embedding provider returned an unexpected vector size.",
                    status_code=503,
                )
            self._validate_metadata(node.metadata)
            document_id = node.metadata.get("document_id")
            if not isinstance(document_id, str) or not document_id:
                raise ValueError("vector nodes require a non-empty document_id")
            text_nodes.append(
                TextNode(
                    id_=node.id,
                    text=node.text,
                    embedding=list(node.embedding),
                    metadata=dict(node.metadata),
                    relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=document_id)},
                )
            )
        persisted = self.integration.add(text_nodes)
        if set(persisted) != {node.id for node in nodes}:
            raise RagError(
                "vector_index_incomplete",
                "The vector index did not acknowledge every document chunk.",
                status_code=503,
                retryable=True,
            )

    def replace_document(self, document_id: str, nodes: Sequence[VectorNode]) -> None:
        """Stage a complete new version, remove stale IDs, and prove exact readback."""

        if any(node.metadata.get("document_id") != document_id for node in nodes):
            raise ValueError("every replacement node must belong to the target document")
        target_ids = {node.id for node in nodes}
        if len(target_ids) != len(nodes):
            raise ValueError("replacement node IDs must be unique")
        self.upsert_nodes(nodes)
        persisted_ids = set(self.get_document_node_ids(document_id))
        stale_ids = sorted(persisted_ids - target_ids)
        if stale_ids:
            self.integration.delete_nodes(node_ids=stale_ids)
        final_ids = set(self.get_document_node_ids(document_id))
        if final_ids != target_ids:
            raise RagError(
                "vector_index_incomplete",
                "The vector index did not persist every document chunk.",
                status_code=503,
                retryable=True,
            )

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[RetrievedNode]:
        """Retrieve chunks, optionally filtered to an explicit document allowlist."""

        if len(query_embedding) != self.settings.embedding_dimensions:
            raise RagError(
                "embedding_dimension_mismatch",
                "The embedding provider returned an unexpected vector size.",
                status_code=503,
            )
        available = self.count()
        if available == 0 or top_k <= 0:
            return []

        filters: MetadataFilters | None = None
        if document_ids is not None:
            unique_ids = list(dict.fromkeys(document_ids))
            if not unique_ids:
                return []
            filters = MetadataFilters(
                filters=[
                    ExactMatchFilter(key="document_id", value=document_id)
                    for document_id in unique_ids
                ],
                condition=FilterCondition.OR,
            )
        result = self.integration.query(
            VectorStoreQuery(
                query_embedding=list(query_embedding),
                similarity_top_k=min(top_k, available),
                filters=filters,
            )
        )
        nodes = result.nodes or []
        similarities = result.similarities or []
        ids = result.ids or []
        retrieved: list[RetrievedNode] = []
        for node, score, node_id in zip(nodes, similarities, ids, strict=True):
            retrieved.append(
                RetrievedNode(
                    id=str(node_id),
                    text=node.get_content(),
                    metadata=_scalar_metadata(node.metadata),
                    score=max(-1.0, min(1.0, float(score))),
                )
            )
        return retrieved

    def delete_document(self, document_id: str) -> int:
        """Delete exact backend IDs and require a zero-result readback."""

        ids = self.get_document_node_ids(document_id)
        if ids:
            self.integration.delete_nodes(node_ids=ids)
        remaining = self.count(document_id=document_id)
        if remaining != 0:
            raise RagError(
                "vector_delete_incomplete",
                "The vector backend did not confirm complete document deletion.",
                status_code=503,
                retryable=True,
            )
        return len(ids)

    @staticmethod
    def _document_filter(document_id: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        )

    @staticmethod
    def _validate_metadata(metadata: Mapping[str, object]) -> None:
        for key, value in metadata.items():
            if not isinstance(key, str) or not key:
                raise ValueError("vector metadata keys must be non-empty strings")
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError("vector metadata values must be scalar")


def _scalar_metadata(metadata: Mapping[str, object]) -> dict[str, MetadataScalar]:
    """Narrow LlamaIndex/Qdrant payload metadata to this application's contract."""

    return {
        key: value for key, value in metadata.items() if isinstance(value, (str, int, float, bool))
    }


def _retryable_qdrant_startup_error(exc: Exception) -> bool:
    """Limit startup retries to transport failures and server-side responses."""

    if isinstance(exc, UnexpectedResponse):
        return exc.status_code is not None and exc.status_code >= 500
    if isinstance(exc, ResponseHandlingException):
        return isinstance(
            exc.source,
            (httpx.TransportError, ConnectionError, TimeoutError),
        )
    return isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError))
