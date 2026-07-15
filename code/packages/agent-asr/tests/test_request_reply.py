import asyncio
import json
import os
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from urgenurse.agents.agent.requests import AgentRequest, AgentRequestPayloadFile

# analyze_audio is the Whisper + LLM boundary; stub it so runtime tests exercise
# dispatch/response wiring without loading models.
_FAKE_ANALYSIS = ("transcribed audio", "summary", {"patient_name": "Jane"}, {}, 0.7)


def _make_raw_request(req_id: str) -> bytes:
    return (
        AgentRequest(
            id=req_id,
            payload=AgentRequestPayloadFile(
                attachment_id="att-x",
                filename="voice.wav",
                mime_type="audio/wav",
                path="/data/attachments/voice.wav",
            ),
        )
        .model_dump_json()
        .encode()
    )


@pytest.mark.asyncio
async def test_asr_subject_registered() -> None:
    from urgenurse.agents.asr.main import agent

    assert "attachment.audio" in agent._handlers


@pytest.mark.asyncio
async def test_valid_request_produces_ok_response() -> None:
    from urgenurse.agents.agent.runtime import _dispatch
    from urgenurse.agents.asr.main import agent

    raw = _make_raw_request("r1")
    entry = agent._handlers["attachment.audio"]
    with patch(
        "urgenurse.agents.asr.handlers.analyze_audio",
        new=AsyncMock(return_value=_FAKE_ANALYSIS),
    ):
        response_bytes = await _dispatch("attachment.audio", raw, entry, agent)

    parsed = json.loads(response_bytes)
    assert parsed["ok"] is True
    assert parsed["id"] == "r1"


@pytest.mark.asyncio
async def test_invalid_json_raises_parse_error() -> None:
    from urgenurse.agents.agent.errors import MessageParseError
    from urgenurse.agents.agent.runtime import _dispatch
    from urgenurse.agents.asr.main import agent

    entry = agent._handlers["attachment.audio"]
    with pytest.raises(MessageParseError):
        await _dispatch("attachment.audio", b"not-json", entry, agent)


@pytest.mark.asyncio
async def test_runtime_loop_calls_respond() -> None:
    from urgenurse.agents.agent.runtime import run_loop
    from urgenurse.agents.asr.main import agent

    mock_nc = AsyncMock()
    mock_nc.drain = AsyncMock()
    subscriptions: list[tuple[str, object]] = []

    async def _subscribe(subject, cb):
        subscriptions.append((subject, cb))
        return MagicMock()

    mock_nc.subscribe = AsyncMock(side_effect=_subscribe)

    raw = _make_raw_request("r2")

    async def _drive():
        await asyncio.sleep(0.05)
        assert subscriptions
        subject, cb = subscriptions[0]
        assert subject == "attachment.audio"
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
            "urgenurse.agents.asr.handlers.analyze_audio",
            new=AsyncMock(return_value=_FAKE_ANALYSIS),
        ),
    ):
        await asyncio.gather(
            run_loop(agent.config, agent._handlers, agent),
            asyncio.create_task(_drive()),
        )
