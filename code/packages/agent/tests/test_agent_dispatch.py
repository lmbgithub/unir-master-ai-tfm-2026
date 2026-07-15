import asyncio

import pytest
from pydantic import BaseModel

from urgenurse.agents.agent import Agent, AgentConfig
from urgenurse.agents.agent.errors import HandlerTimeoutError, MessageParseError
from urgenurse.agents.agent.runtime import _dispatch


class Ping(BaseModel):
    value: int


class Pong(BaseModel):
    result: int


@pytest.fixture
def agent() -> Agent:
    return Agent(AgentConfig())


@pytest.mark.asyncio
async def test_valid_payload_returns_response(agent: Agent) -> None:
    @agent.subscribe(subject="test.ping", request=Ping, response=Pong)
    async def handle(_agent: Agent, msg: Ping) -> Pong:
        return Pong(result=msg.value * 2)

    entry = agent._handlers["test.ping"]
    result = await _dispatch("test.ping", b'{"value": 21}', entry, agent)
    assert Pong.model_validate_json(result) == Pong(result=42)


@pytest.mark.asyncio
async def test_invalid_json_raises_parse_error(agent: Agent) -> None:
    @agent.subscribe(subject="test.bad", request=Ping, response=Pong)
    async def handle(_agent: Agent, msg: Ping) -> Pong:  # pragma: no cover
        return Pong(result=0)

    entry = agent._handlers["test.bad"]
    with pytest.raises(MessageParseError):
        await _dispatch("test.bad", b"not json", entry, agent)


@pytest.mark.asyncio
async def test_handler_timeout_raises_timeout_error(agent: Agent) -> None:
    @agent.subscribe(subject="test.slow", request=Ping, response=Pong, timeout=1)
    async def handle(_agent: Agent, msg: Ping) -> Pong:  # pragma: no cover
        await asyncio.sleep(5)
        return Pong(result=0)

    entry = agent._handlers["test.slow"]
    with pytest.raises(HandlerTimeoutError):
        await _dispatch("test.slow", b'{"value": 1}', entry, agent)
