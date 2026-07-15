from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from urgenurse.agents.ocr.config import OcrAgentConfig
from urgenurse.agents.ocr.llm import _ocr_file, analyze_with_ocr


@pytest.mark.asyncio
async def test_ocr_file_exports_stripped_text() -> None:
    converter = MagicMock()
    converter.convert.return_value.document.export_to_text.return_value = (
        "  Patient: John Doe\nDiagnosis: Flu  "
    )

    text = await _ocr_file(converter, "/tmp/scan.png")

    assert text == "Patient: John Doe\nDiagnosis: Flu"
    converter.convert.assert_called_once_with("/tmp/scan.png")


@pytest.mark.asyncio
async def test_analyze_with_ocr_skips_llm_on_empty_transcription() -> None:
    config = OcrAgentConfig()
    with (
        patch("urgenurse.agents.ocr.llm._ocr_file", new=AsyncMock(return_value="")),
        patch("urgenurse.agents.ocr.llm.call_llm", new=AsyncMock()) as mock_llm,
    ):
        transcription, summary, ner, confidence = await analyze_with_ocr(
            config, MagicMock(), "/tmp/blank.png"
        )

    assert transcription == ""
    assert summary == ""
    assert ner == {}
    assert confidence is None
    mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_with_ocr_runs_llm_on_transcription() -> None:
    config = OcrAgentConfig()
    with (
        patch(
            "urgenurse.agents.ocr.llm._ocr_file",
            new=AsyncMock(return_value="Patient John Doe"),
        ),
        patch(
            "urgenurse.agents.ocr.llm.call_llm",
            new=AsyncMock(
                return_value={
                    "summary": "Routine visit.",
                    "ner": {"patient_name": "John Doe"},
                    "confidence": 0.8,
                }
            ),
        ) as mock_llm,
    ):
        transcription, summary, ner, confidence = await analyze_with_ocr(
            config, MagicMock(), "/tmp/scan.png"
        )

    assert transcription == "Patient John Doe"
    assert summary == "Routine visit."
    assert ner == {"patient_name": "John Doe"}
    assert confidence == 0.8
    mock_llm.assert_called_once()
