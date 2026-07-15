import asyncio
import json
import os
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from urgenurse.agents.agent.requests import AgentRequest, AgentRequestPayloadTriage

# evaluate_triage is the LLM boundary; stub it so runtime tests exercise
# dispatch/response wiring without reaching the model.
_FAKE_TRIAGE = (True, [], 3, "Reviewed documentation; vitals adequate.")


def _make_raw_request(req_id: str, case_id: str) -> bytes:
    return (
        AgentRequest(
            id=req_id,
            payload=AgentRequestPayloadTriage(
                case_id=case_id,
                patient={"name": "Patient"},
                description="shortness of breath",
            ),
        )
        .model_dump_json()
        .encode()
    )


@pytest.mark.asyncio
async def test_triage_subject_registered() -> None:
    from urgenurse.agents.triage.main import agent

    assert "triage.request" in agent._handlers


@pytest.mark.asyncio
async def test_valid_request_produces_ok_response() -> None:
    from urgenurse.agents.agent.runtime import _dispatch
    from urgenurse.agents.triage.main import agent

    raw = _make_raw_request("r1", "case-001")
    entry = agent._handlers["triage.request"]
    with patch(
        "urgenurse.agents.triage.handlers.evaluate_triage",
        new=AsyncMock(return_value=_FAKE_TRIAGE),
    ):
        response_bytes = await _dispatch("triage.request", raw, entry, agent)

    parsed = json.loads(response_bytes)
    assert parsed["ok"] is True
    assert parsed["id"] == "r1"


@pytest.mark.asyncio
async def test_invalid_json_raises_parse_error() -> None:
    from urgenurse.agents.agent.errors import MessageParseError
    from urgenurse.agents.agent.runtime import _dispatch
    from urgenurse.agents.triage.main import agent

    entry = agent._handlers["triage.request"]
    with pytest.raises(MessageParseError):
        await _dispatch("triage.request", b"not-json", entry, agent)


@pytest.mark.asyncio
async def test_runtime_loop_calls_respond() -> None:
    from urgenurse.agents.agent.runtime import run_loop
    from urgenurse.agents.triage.main import agent

    mock_nc = AsyncMock()
    mock_nc.drain = AsyncMock()
    subscriptions: list[tuple[str, object]] = []

    async def _subscribe(subject, cb):
        subscriptions.append((subject, cb))
        return MagicMock()

    mock_nc.subscribe = AsyncMock(side_effect=_subscribe)

    raw = _make_raw_request("r2", "case-002")

    async def _drive():
        await asyncio.sleep(0.05)
        assert subscriptions
        subject, cb = subscriptions[0]
        assert subject == "triage.request"
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
            "urgenurse.agents.triage.handlers.evaluate_triage",
            new=AsyncMock(return_value=_FAKE_TRIAGE),
        ),
    ):
        await asyncio.gather(
            run_loop(agent.config, agent._handlers, agent),
            asyncio.create_task(_drive()),
        )
