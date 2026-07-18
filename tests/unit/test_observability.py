from __future__ import annotations

import pytest
from starlette.types import Message, Receive, Scope, Send

from personal_rag.observability import RequestBodyLimitMiddleware


@pytest.mark.asyncio
async def test_streamed_upload_body_is_bounded_without_content_length() -> None:
    downstream_completed = False

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        del scope, send
        nonlocal downstream_completed
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        downstream_completed = True

    messages = iter(
        [
            {"type": "http.request", "body": b"abc", "more_body": True},
            {"type": "http.request", "body": b"def", "more_body": False},
        ]
    )
    sent: list[Message] = []

    async def receive() -> Message:
        return next(messages)

    async def send(message: Message) -> None:
        sent.append(message)

    middleware = RequestBodyLimitMiddleware(
        downstream,
        max_upload_bytes=5,
        multipart_overhead_bytes=0,
    )
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/documents",
        "headers": [],
    }
    await middleware(
        scope,
        receive,
        send,
    )

    assert downstream_completed is False
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
    assert b"request_too_large" in sent[1]["body"]
