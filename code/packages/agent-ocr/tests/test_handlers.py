from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from urgenurse.agents.agent.requests import (
    AgentRequest,
    AgentResponse,
    AttachmentTranscriptions,
)
from urgenurse.agents.ocr.handlers import handle_ocr


def _agent() -> SimpleNamespace:
    # handle_ocr only reads .config and ._model before delegating to analyze_with_ocr
    return SimpleNamespace(config=None, _model=object())


@pytest.mark.asyncio
async def test_image_request_returns_ok_response(image_request: AgentRequest) -> None:
    with patch(
        "urgenurse.agents.ocr.handlers.analyze_with_ocr",
        new=AsyncMock(
            return_value=(
                "Patient: John Doe",
                "Routine check, no findings.",
                {"patient_name": "John Doe"},
                0.9,
            )
        ),
    ):
        result = await handle_ocr(_agent(), image_request)

    assert isinstance(result, AgentResponse)
    assert result.id == image_request.id
    assert result.ok is True
    assert result.error is None
    assert isinstance(result.payload, AttachmentTranscriptions)
    assert result.payload.name == "scan.jpg"
    assert result.payload.content == "Patient: John Doe"
    assert result.payload.summary == "Routine check, no findings."
    assert result.payload.ner == {"patient_name": "John Doe"}
    assert result.payload.confidence == 0.9


@pytest.mark.asyncio
async def test_pdf_request_returns_ok_response(pdf_request: AgentRequest) -> None:
    with patch(
        "urgenurse.agents.ocr.handlers.analyze_with_ocr",
        new=AsyncMock(return_value=("lab report text", "summary", {}, 0.5)),
    ):
        result = await handle_ocr(_agent(), pdf_request)

    assert result.id == pdf_request.id
    assert result.ok is True
    assert isinstance(result.payload, AttachmentTranscriptions)
    assert result.payload.name == "report.pdf"
    assert result.payload.content == "lab report text"


@pytest.mark.asyncio
async def test_empty_transcription_returns_ok_with_blank_fields(
    image_request: AgentRequest,
) -> None:
    # analyze_with_ocr yields empty results when OCR finds no text
    with patch(
        "urgenurse.agents.ocr.handlers.analyze_with_ocr",
        new=AsyncMock(return_value=("", "", {}, None)),
    ):
        result = await handle_ocr(_agent(), image_request)

    assert result.ok is True
    assert isinstance(result.payload, AttachmentTranscriptions)
    assert result.payload.content == ""
    assert result.payload.confidence is None


@pytest.mark.asyncio
async def test_analyze_exception_returns_error_response(
    image_request: AgentRequest,
) -> None:
    with patch(
        "urgenurse.agents.ocr.handlers.analyze_with_ocr",
        new=AsyncMock(side_effect=RuntimeError("disk read failed")),
    ):
        result = await handle_ocr(_agent(), image_request)

    assert result.ok is False
    assert "disk read failed" in (result.error or "")
    assert result.id == image_request.id
