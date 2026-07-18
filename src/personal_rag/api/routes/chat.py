"""Grounded, citation-bearing chat endpoint."""

from __future__ import annotations

from functools import partial

import anyio
from fastapi import APIRouter, Request

from personal_rag.api.dependencies import get_rag_service
from personal_rag.errors import RagError
from personal_rag.models import ChatRequest, ChatResponse
from personal_rag.observability import current_request_id

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    container = request.app.state.container
    if body.top_k is not None and body.top_k > container.settings.retrieval_max_top_k:
        raise RagError(
            "invalid_top_k",
            f"top_k cannot exceed {container.settings.retrieval_max_top_k}.",
            status_code=422,
        )
    body.history = body.history[-container.settings.max_history_messages :]
    rag_service = get_rag_service(request)
    result: ChatResponse = await anyio.to_thread.run_sync(partial(rag_service.chat, body))
    result.request_id = current_request_id()
    return result
