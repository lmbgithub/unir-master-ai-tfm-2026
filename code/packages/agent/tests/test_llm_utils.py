from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from urgenurse.agents.agent.config import LLMAgentConfig
from urgenurse.agents.agent.llm_utils import (
    call_llm,
    clamp_confidence,
    filter_ner,
)


def test_clamp_confidence_none_returns_none() -> None:
    assert clamp_confidence(None) is None


def test_clamp_confidence_clamps_to_unit_range() -> None:
    assert clamp_confidence(1.5) == 1.0
    assert clamp_confidence(-0.3) == 0.0
    assert clamp_confidence(0.42) == 0.42


def test_clamp_confidence_parses_numeric_string() -> None:
    assert clamp_confidence("0.7") == 0.7


def test_clamp_confidence_non_numeric_returns_none() -> None:
    # the LLM sometimes emits a word instead of a float — must not crash
    assert clamp_confidence("high") is None
    assert clamp_confidence({"x": 1}) is None


def test_filter_ner_drops_empty_and_placeholder_values() -> None:
    raw = {
        "patient_name": "Jane Doe",
        "doctor_name": "",
        "institution": "None",
        "diagnosis": "n/a",
        "allergies": None,
        "dosage": "10mg",
    }
    assert filter_ner(raw) == {"patient_name": "Jane Doe", "dosage": "10mg"}


def test_filter_ner_coerces_values_to_str() -> None:
    assert filter_ner({"date": 2024}) == {"date": "2024"}


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    resp.text = content
    return resp


def _patch_client(client: AsyncMock):
    # httpx.AsyncClient(...) is used as `async with ... as client`
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return patch("urgenurse.agents.agent.llm_utils.httpx.AsyncClient", return_value=cm)


@pytest.mark.asyncio
async def test_call_llm_strips_markdown_fences() -> None:
    config = LLMAgentConfig()
    client = AsyncMock()
    client.post = AsyncMock(
        return_value=_mock_response('```json\n{"summary": "ok"}\n```')
    )
    with _patch_client(client):
        data = await call_llm(config, messages=[{"role": "user", "content": "x"}])
    assert data == {"summary": "ok"}


@pytest.mark.asyncio
async def test_call_llm_repairs_malformed_json() -> None:
    config = LLMAgentConfig()
    client = AsyncMock()
    broken = '{"summary": "ok", "confidence": 0.9,}'  # trailing comma
    client.post = AsyncMock(return_value=_mock_response(broken))
    with _patch_client(client):
        data = await call_llm(config, messages=[{"role": "user", "content": "x"}])
    assert data["summary"] == "ok"


@pytest.mark.asyncio
async def test_call_llm_connect_error_raises_connection_error() -> None:
    config = LLMAgentConfig()
    client = AsyncMock()
    client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
    with _patch_client(client):
        with pytest.raises(ConnectionError):
            await call_llm(config, messages=[{"role": "user", "content": "x"}])
