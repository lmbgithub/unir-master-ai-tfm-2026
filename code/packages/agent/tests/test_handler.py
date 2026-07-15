import pytest

from urgenurse.agents.agent import agent_handler
from urgenurse.agents.agent.requests import (
    AgentRequest,
    AgentRequestPayloadTriage,
    AgentResponse,
)


def _request() -> AgentRequest:
    return AgentRequest(
        id="req-1",
        payload=AgentRequestPayloadTriage(case_id="c1", patient={}, description="x"),
    )


@pytest.mark.asyncio
async def test_agent_handler_passes_through_success() -> None:
    @agent_handler
    async def handler(_agent: object, req: AgentRequest) -> AgentResponse:
        return AgentResponse(id=req.id, ok=True)

    result = await handler(object(), _request())

    assert result.ok is True
    assert result.id == "req-1"
    assert result.error is None


@pytest.mark.asyncio
async def test_agent_handler_converts_exception_to_error_response() -> None:
    @agent_handler
    async def handler(_agent: object, req: AgentRequest) -> AgentResponse:
        raise RuntimeError("boom")

    result = await handler(object(), _request())

    assert result.ok is False
    assert result.id == "req-1"
    assert "boom" in (result.error or "")
