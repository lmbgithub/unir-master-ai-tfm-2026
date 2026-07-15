from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from urgenurse.agents.agent.requests import (
    AgentRequest,
    AgentResponse,
    AttachmentTranscriptions,
)
from urgenurse.agents.asr.handlers import handle_asr


def _agent() -> SimpleNamespace:
    return SimpleNamespace(config=None, _model=object())


@pytest.mark.asyncio
async def test_handle_asr_returns_ok_response(audio_request: AgentRequest) -> None:
    with patch(
        "urgenurse.agents.asr.handlers.analyze_audio",
        new=AsyncMock(
            return_value=(
                "patient reports chest pain",
                "Chest pain, onset 1h.",
                {"patient_name": "John Doe"},
                {"situation": "chest pain"},
                0.8,
            )
        ),
    ):
        result = await handle_asr(_agent(), audio_request)

    assert isinstance(result, AgentResponse)
    assert result.id == audio_request.id
    assert result.ok is True
    assert result.error is None
    assert isinstance(result.payload, AttachmentTranscriptions)
    assert result.payload.name == "voice.wav"
    assert result.payload.content == "patient reports chest pain"
    assert result.payload.sbar == {"situation": "chest pain"}
    assert result.payload.confidence == 0.8


@pytest.mark.asyncio
async def test_handle_asr_exception_returns_error_response(
    audio_request: AgentRequest,
) -> None:
    with patch(
        "urgenurse.agents.asr.handlers.analyze_audio",
        new=AsyncMock(side_effect=RuntimeError("model load failed")),
    ):
        result = await handle_asr(_agent(), audio_request)

    assert result.ok is False
    assert "model load failed" in (result.error or "")
    assert result.id == audio_request.id
