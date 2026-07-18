"""Durable personal conversation and grounded-turn endpoints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from functools import partial
from typing import Annotated, cast

import anyio
from fastapi import APIRouter, Path, Query, Request, Response, status

from personal_rag.api.dependencies import get_rag_service
from personal_rag.errors import RagError
from personal_rag.models import (
    ChatRequest,
    ChatResponse,
    ConversationCreate,
    ConversationList,
    ConversationSummary,
    ConversationTurn,
    ConversationTurnCreate,
    ConversationTurnList,
    ConversationTurnStatus,
)
from personal_rag.observability import current_request_id
from personal_rag.repository import Repository

router = APIRouter(prefix="/conversations", tags=["conversations"])
TURN_RESERVATION_SECONDS = 120
TURN_HEARTBEAT_SECONDS = 30


def _turn_request_fingerprint(body: ConversationTurnCreate) -> str:
    payload = json.dumps(
        {
            "document_ids": body.document_ids,
            "message": body.message,
            "top_k": body.top_k,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_conversation(repository: Repository, conversation_id: str) -> ConversationSummary:
    conversation = repository.get_conversation(conversation_id)
    if conversation is None:
        raise RagError(
            "conversation_not_found",
            "The requested conversation does not exist.",
            status_code=404,
        )
    return conversation


async def _run_with_reservation_heartbeat(
    repository: Repository,
    *,
    turn_id: str,
    reservation_token: str,
    call: Callable[[], ChatResponse],
) -> ChatResponse:
    """Keep a live provider call owned while preserving crash recovery."""

    async def heartbeat() -> None:
        while True:
            await anyio.sleep(TURN_HEARTBEAT_SECONDS)
            renewed = await anyio.to_thread.run_sync(
                partial(
                    repository.renew_conversation_turn_reservation,
                    turn_id,
                    reservation_token=reservation_token,
                    reservation_seconds=TURN_RESERVATION_SECONDS,
                )
            )
            if not renewed:
                return

    result: ChatResponse | None = None
    provider_error: Exception | None = None
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(heartbeat)
        try:
            result = await anyio.to_thread.run_sync(call)
        except Exception as error:  # Preserve the original sanitized domain error type.
            provider_error = error
        finally:
            task_group.cancel_scope.cancel()
    if provider_error is not None:
        raise provider_error
    if result is None:
        raise RuntimeError("The reserved answer call returned no result")
    return result


@router.post("", response_model=ConversationSummary, status_code=status.HTTP_201_CREATED)
def create_conversation(body: ConversationCreate, request: Request) -> ConversationSummary:
    repository = cast(Repository, request.app.state.container.repository)
    return repository.create_conversation(body.title)


@router.get("", response_model=ConversationList)
def list_conversations(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ConversationList:
    repository = cast(Repository, request.app.state.container.repository)
    return ConversationList(
        items=repository.list_conversations(limit=limit, offset=offset),
        total=repository.count_conversations(),
        limit=limit,
        offset=offset,
    )


@router.get("/{conversation_id}/turns", response_model=ConversationTurnList)
def list_conversation_turns(
    conversation_id: Annotated[str, Path(min_length=1, max_length=128)],
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ConversationTurnList:
    repository = cast(Repository, request.app.state.container.repository)
    _require_conversation(repository, conversation_id)
    return ConversationTurnList(
        items=repository.list_conversation_turns(
            conversation_id,
            limit=limit,
            offset=offset,
            include_incomplete=True,
        ),
        total=repository.count_conversation_turns(
            conversation_id,
            include_incomplete=True,
        ),
        limit=limit,
        offset=offset,
    )


@router.post("/{conversation_id}/turns", response_model=ConversationTurn)
async def create_conversation_turn(
    conversation_id: Annotated[str, Path(min_length=1, max_length=128)],
    body: ConversationTurnCreate,
    request: Request,
) -> ConversationTurn:
    container = request.app.state.container
    repository = cast(Repository, container.repository)
    if body.top_k is not None and body.top_k > container.settings.retrieval_max_top_k:
        raise RagError(
            "invalid_top_k",
            f"top_k cannot exceed {container.settings.retrieval_max_top_k}.",
            status_code=422,
        )

    reservation = repository.reserve_conversation_turn(
        conversation_id,
        client_turn_id=body.client_turn_id,
        question=body.message,
        top_k=body.top_k,
        document_ids=body.document_ids,
        request_fingerprint=_turn_request_fingerprint(body),
        reservation_seconds=TURN_RESERVATION_SECONDS,
    )
    if reservation.cached_turn is not None:
        return reservation.cached_turn
    reservation_token = reservation.reservation_token
    if reservation_token is None:
        raise RuntimeError("A newly reserved conversation turn has no ownership token")

    history = repository.conversation_history(
        conversation_id,
        limit=container.settings.max_history_messages,
    )
    try:
        rag_service = get_rag_service(request)
        result = await _run_with_reservation_heartbeat(
            repository,
            turn_id=reservation.turn.id,
            reservation_token=reservation_token,
            call=partial(
                rag_service.chat,
                ChatRequest(
                    message=body.message,
                    history=history,
                    top_k=body.top_k,
                    document_ids=body.document_ids,
                ),
            ),
        )
    except RagError as error:
        persisted = repository.fail_conversation_turn(
            reservation.turn.id,
            reservation_token=reservation_token,
            error_code=error.code,
            retryable=error.retryable,
        )
        if persisted.status is ConversationTurnStatus.COMPLETED:
            return persisted
        raise
    except Exception:
        persisted = repository.fail_conversation_turn(
            reservation.turn.id,
            reservation_token=reservation_token,
            error_code="internal_error",
            retryable=False,
        )
        if persisted.status is ConversationTurnStatus.COMPLETED:
            return persisted
        raise

    try:
        return repository.complete_conversation_turn(
            reservation.turn.id,
            reservation_token=reservation_token,
            answer=result.answer,
            citations=result.citations,
            no_answer=result.no_answer,
            request_id=current_request_id(),
        )
    except RagError as error:
        persisted = repository.fail_conversation_turn(
            reservation.turn.id,
            reservation_token=reservation_token,
            error_code=error.code,
            retryable=error.retryable,
        )
        if persisted.status is ConversationTurnStatus.COMPLETED:
            return persisted
        raise
    except Exception:
        persisted = repository.fail_conversation_turn(
            reservation.turn.id,
            reservation_token=reservation_token,
            error_code="internal_error",
            retryable=False,
        )
        if persisted.status is ConversationTurnStatus.COMPLETED:
            return persisted
        raise


@router.get("/{conversation_id}", response_model=ConversationSummary)
def get_conversation(
    conversation_id: Annotated[str, Path(min_length=1, max_length=128)],
    request: Request,
) -> ConversationSummary:
    repository = cast(Repository, request.app.state.container.repository)
    return _require_conversation(repository, conversation_id)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: Annotated[str, Path(min_length=1, max_length=128)],
    request: Request,
) -> Response:
    repository = cast(Repository, request.app.state.container.repository)
    if not repository.delete_conversation(conversation_id):
        raise RagError(
            "conversation_not_found",
            "The requested conversation does not exist.",
            status_code=404,
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
