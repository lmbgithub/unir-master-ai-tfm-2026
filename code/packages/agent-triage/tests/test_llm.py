from unittest.mock import AsyncMock, patch

import pytest

from urgenurse.agents.triage.config import TriageAgentConfig
from urgenurse.agents.triage.llm import evaluate_triage


@pytest.mark.asyncio
async def test_evaluate_triage_keeps_llm_level_when_vitals_are_safe() -> None:
    config = TriageAgentConfig()
    patient = {"pulse": 84, "respiratory_rate": 16, "oxygen_saturation_percent": 98}
    with patch(
        "urgenurse.agents.triage.llm.call_llm",
        new=AsyncMock(
            return_value={
                "valid": True,
                "missing_fields": [],
                "esi_level": 4,
                "analysis": "Stable vitals, minor complaint.",
            }
        ),
    ):
        valid, missing_fields, esi_level, analysis = await evaluate_triage(
            config, patient, "sore throat", []
        )

    assert valid is True
    assert esi_level == 4
    assert "Auto-escalated" not in analysis


@pytest.mark.asyncio
async def test_evaluate_triage_escalates_to_esi_2_on_danger_zone_vitals() -> None:
    config = TriageAgentConfig()
    patient = {"pulse": 140, "respiratory_rate": 16, "oxygen_saturation_percent": 98}
    with patch(
        "urgenurse.agents.triage.llm.call_llm",
        new=AsyncMock(
            return_value={
                "valid": True,
                "missing_fields": [],
                "esi_level": 4,
                "analysis": "Minor complaint documented.",
            }
        ),
    ):
        valid, missing_fields, esi_level, analysis = await evaluate_triage(
            config, patient, "med refill", []
        )

    assert esi_level == 2
    assert "Auto-escalated from ESI 4 to ESI 2" in analysis


@pytest.mark.asyncio
async def test_evaluate_triage_never_escalates_below_esi_2() -> None:
    config = TriageAgentConfig()
    patient = {"pulse": 140, "respiratory_rate": 16, "oxygen_saturation_percent": 98}
    with patch(
        "urgenurse.agents.triage.llm.call_llm",
        new=AsyncMock(
            return_value={
                "valid": True,
                "missing_fields": [],
                "esi_level": 1,
                "analysis": "Immediate lifesaving intervention required.",
            }
        ),
    ):
        _, _, esi_level, analysis = await evaluate_triage(config, patient, "cardiac arrest", [])

    assert esi_level == 1
    assert "Auto-escalated" not in analysis


@pytest.mark.asyncio
async def test_evaluate_triage_handles_null_esi_level_without_crashing() -> None:
    """Regresión: data.get('esi_level', 3) no cubre 'esi_level': null explícito
    en el JSON del modelo -> int(None) reventaba. Ver hallazgo en la corrida
    de evaluación n=300 (5 casos ESI-1 fallando con este error)."""
    config = TriageAgentConfig()
    patient = {"pulse": 84, "respiratory_rate": 16, "oxygen_saturation_percent": 98}
    with patch(
        "urgenurse.agents.triage.llm.call_llm",
        new=AsyncMock(
            return_value={
                "valid": True,
                "missing_fields": [],
                "esi_level": None,
                "analysis": "Uncertain.",
            }
        ),
    ):
        _, _, esi_level, _ = await evaluate_triage(config, patient, "unclear complaint", [])

    assert esi_level == 3
