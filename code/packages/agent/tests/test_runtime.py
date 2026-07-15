import asyncio
import os
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from urgenurse.agents.agent import Agent
from urgenurse.agents.agent.config import AgentConfig
from urgenurse.agents.agent.runtime import run_loop


class _Req(BaseModel):
    value: int


class _Resp(BaseModel):
    result: int


async def _echo_handler(_agent: Agent, msg: _Req) -> _Resp:
    return _Resp(result=msg.value * 2)


def _make_config() -> AgentConfig:
    return AgentConfig()


def _make_agent(config: AgentConfig, handlers: dict) -> Agent:
    agent = Agent(config)
    agent._handlers = handlers
    return agent


def _make_handlers() -> dict:
    return {"test.subject": (_Req, _Resp, 5, _echo_handler)}


def _make_msg(data: bytes) -> MagicMock:
    msg = MagicMock()
    msg.data = data
    msg.respond = AsyncMock()
    return msg


@pytest.fixture
def mock_nats():
    mock_nc = AsyncMock()
    mock_nc.drain = AsyncMock()

    subscriptions: list[tuple[str, object]] = []

    async def _subscribe(subject, cb):
        subscriptions.append((subject, cb))
        return MagicMock()

    mock_nc.subscribe = AsyncMock(side_effect=_subscribe)

    with patch("nats.connect", return_value=mock_nc):
        yield mock_nc, subscriptions


@pytest.mark.asyncio
async def test_valid_message_handler_responds(mock_nats):
    mock_nc, subscriptions = mock_nats
    handlers = _make_handlers()
    config = _make_config()

    async def _run():
        await asyncio.sleep(0.05)
        _, cb = subscriptions[0]
        msg = _make_msg(b'{"value": 21}')
        await cb(msg)
        await asyncio.sleep(0.05)
        os.kill(os.getpid(), signal.SIGTERM)

    agent = _make_agent(config, handlers)
    loop_task = asyncio.create_task(run_loop(config, handlers, agent))
    runner_task = asyncio.create_task(_run())

    await asyncio.gather(loop_task, runner_task)

    msg = _make_msg(b'{"value": 21}')
    _, cb = subscriptions[0]
    await cb(msg)
    msg.respond.assert_called_once()
    assert b'"result":42' in msg.respond.call_args[0][0]


@pytest.mark.asyncio
async def test_invalid_json_responds_with_error(mock_nats):
    mock_nc, subscriptions = mock_nats
    handlers = _make_handlers()
    config = _make_config()

    async def _run():
        await asyncio.sleep(0.05)
        _, cb = subscriptions[0]
        bad_msg = _make_msg(b"not-valid-json")
        await cb(bad_msg)
        bad_msg.respond.assert_called_once()
        os.kill(os.getpid(), signal.SIGTERM)

    agent = _make_agent(config, handlers)
    loop_task = asyncio.create_task(run_loop(config, handlers, agent))
    runner_task = asyncio.create_task(_run())

    await asyncio.gather(loop_task, runner_task)


@pytest.mark.asyncio
async def test_sigterm_exits_cleanly(mock_nats):
    mock_nc, subscriptions = mock_nats
    config = _make_config()

    async def _send_sigterm():
        await asyncio.sleep(0.1)
        os.kill(os.getpid(), signal.SIGTERM)

    handlers = _make_handlers()
    agent = _make_agent(config, handlers)
    start = asyncio.get_event_loop().time()
    loop_task = asyncio.create_task(run_loop(config, handlers, agent))
    sig_task = asyncio.create_task(_send_sigterm())

    await asyncio.gather(loop_task, sig_task)
    elapsed = asyncio.get_event_loop().time() - start

    assert elapsed < 3.0
    mock_nc.drain.assert_called_once()
