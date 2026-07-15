import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from urgenurse.agents.agent.requests import AgentRequest, AgentRequestPayloadFile

# analyze_with_ocr is the Docling + LLM boundary; stub it so the runtime tests
# exercise dispatch/response wiring without loading models.
_FAKE_ANALYSIS = ("transcribed text", "summary", {"patient_name": "Jane"}, 0.7)


def _make_raw_request(req_id: str, path: str, mime: str, filename: str) -> bytes:
    return (
        AgentRequest(
            id=req_id,
            payload=AgentRequestPayloadFile(
                attachment_id="att-x",
                filename=filename,
                mime_type=mime,  # type: ignore[arg-type]
                path=path,
            ),
        )
        .model_dump_json()
        .encode()
    )


@pytest.mark.asyncio
async def test_nats_subscribe_registered_on_correct_subject() -> None:
    """Agent must subscribe to 'attachment.document' on startup."""
    from urgenurse.agents.ocr.main import agent

    assert "attachment.document" in agent._handlers


@pytest.mark.asyncio
async def test_valid_request_produces_respond_call() -> None:
    """runtime._dispatch → response is valid JSON containing ok=true."""
    from urgenurse.agents.agent.runtime import _dispatch
    from urgenurse.agents.ocr.main import agent

    raw = _make_raw_request("r1", "/data/img.jpg", "image/jpeg", "img.jpg")
    entry = agent._handlers["attachment.document"]
    with patch(
        "urgenurse.agents.ocr.handlers.analyze_with_ocr",
        new=AsyncMock(return_value=_FAKE_ANALYSIS),
    ):
        response_bytes = await _dispatch("attachment.document", raw, entry, agent)

    parsed = json.loads(response_bytes)
    assert parsed["ok"] is True
    assert parsed["id"] == "r1"


@pytest.mark.asyncio
async def test_invalid_json_dispatch_raises_parse_error() -> None:
    from urgenurse.agents.agent.errors import MessageParseError
    from urgenurse.agents.agent.runtime import _dispatch
    from urgenurse.agents.ocr.main import agent

    entry = agent._handlers["attachment.document"]
    with pytest.raises(MessageParseError):
        await _dispatch("attachment.document", b"not-json", entry, agent)


@pytest.mark.asyncio
async def test_runtime_loop_calls_respond_on_valid_message() -> None:
    """Full run_loop integration: nc.subscribe registered, callback calls msg.respond."""
    import asyncio
    import os
    import signal

    from urgenurse.agents.agent.runtime import run_loop
    from urgenurse.agents.ocr.main import agent

    mock_nc = AsyncMock()
    mock_nc.drain = AsyncMock()
    subscriptions: list[tuple[str, object]] = []

    async def _subscribe(subject, cb):
        subscriptions.append((subject, cb))
        return MagicMock()

    mock_nc.subscribe = AsyncMock(side_effect=_subscribe)

    raw = _make_raw_request("r2", "/data/doc.pdf", "application/pdf", "doc.pdf")

    async def _drive():
        await asyncio.sleep(0.05)
        assert subscriptions, "subscribe() was never called"
        subject, cb = subscriptions[0]
        assert subject == "attachment.document"
        msg = MagicMock()
        msg.data = raw
        msg.respond = AsyncMock()
        await cb(msg)
        await asyncio.sleep(0.05)
        msg.respond.assert_called_once()
        resp = json.loads(msg.respond.call_args[0][0])
        assert resp["ok"] is True
        os.kill(os.getpid(), signal.SIGTERM)

    with (
        patch("nats.connect", return_value=mock_nc),
        patch(
            "urgenurse.agents.ocr.handlers.analyze_with_ocr",
            new=AsyncMock(return_value=_FAKE_ANALYSIS),
        ),
    ):
        await asyncio.gather(
            run_loop(agent.config, agent._handlers, agent),
            asyncio.create_task(_drive()),
        )
