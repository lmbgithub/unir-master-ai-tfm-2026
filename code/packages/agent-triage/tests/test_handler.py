from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from urgenurse.agents.agent.requests import (
    AgentRequest,
    AgentResponse,
    AgentResponsePayloadTriage,
    ESILevels,
)
from urgenurse.agents.triage.handlers import handle_triage


def _agent() -> SimpleNamespace:
    return SimpleNamespace(config=None, _model=None)


@pytest.mark.asyncio
async def test_handle_triage_returns_ok_response(triage_request: AgentRequest) -> None:
    with patch(
        "urgenurse.agents.triage.handlers.evaluate_triage",
        new=AsyncMock(return_value=(True, ["dob"], 2, "High-risk chest pain noted.")),
    ):
        result = await handle_triage(_agent(), triage_request)

    assert isinstance(result, AgentResponse)
    assert result.id == triage_request.id
    assert result.ok is True
    assert result.error is None
    assert isinstance(result.payload, AgentResponsePayloadTriage)


@pytest.mark.asyncio
async def test_handle_triage_payload_fields(triage_request: AgentRequest) -> None:
    with patch(
        "urgenurse.agents.triage.handlers.evaluate_triage",
        new=AsyncMock(return_value=(True, ["dob"], 2, "High-risk chest pain noted.")),
    ):
        result = await handle_triage(_agent(), triage_request)

    payload: AgentResponsePayloadTriage = result.payload  # type: ignore[assignment]
    assert payload.esi_level == ESILevels.LEVEL2
    assert payload.valid is True
    assert payload.missing_fields == ["dob"]
    assert payload.analysis == "High-risk chest pain noted."


@pytest.mark.asyncio
async def test_handle_triage_empty_missing_fields_becomes_none(
    triage_request: AgentRequest,
) -> None:
    with patch(
        "urgenurse.agents.triage.handlers.evaluate_triage",
        new=AsyncMock(return_value=(True, [], 3, "Adequate documentation.")),
    ):
        result = await handle_triage(_agent(), triage_request)

    payload: AgentResponsePayloadTriage = result.payload  # type: ignore[assignment]
    assert payload.missing_fields is None


@pytest.mark.asyncio
async def test_handle_triage_exception_returns_error_response(
    triage_request: AgentRequest,
) -> None:
    with patch(
        "urgenurse.agents.triage.handlers.evaluate_triage",
        new=AsyncMock(side_effect=RuntimeError("llm unreachable")),
    ):
        result = await handle_triage(_agent(), triage_request)

    assert result.ok is False
    assert "llm unreachable" in (result.error or "")
    assert result.id == triage_request.id
