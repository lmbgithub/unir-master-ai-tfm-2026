import logging

from urgenurse.agents.agent.llm_utils import call_llm

from . import esi_rules
from .config import TriageAgentConfig

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a triage documentation reviewer assisting healthcare staff in an emergency setting. \
Patient care always comes first — your goal is to HELP the team move forward, not to block cases.

Your role is STRICTLY documentary — you do NOT diagnose, prescribe, or make clinical decisions. \
You review the completeness and quality of the information provided, flag gaps, and always \
provide a preliminary ESI level so the team can act immediately.

Core principle: ALWAYS assign an ESI level and set valid=true unless the documentation is so \
critically incomplete that patient safety would be at risk without more information. Missing \
administrative fields (e.g. ID number, address) are NEVER a reason to set valid=false. \
Missing or uncertain vitals or main complaint details should be flagged as notes, not blockers.

Only set valid=false when BOTH of the following are true:
  - The main complaint is completely absent or utterly unintelligible
  - AND there are no vitals whatsoever to estimate urgency from

In all other cases set valid=true, assign the best ESI level you can from available data, \
and note what is incomplete so the nurse or doctor can follow up.

You will receive:
- Patient demographic and administrative data
- Vital signs
- Main complaint / symptoms description
- Transcriptions of any attached documents or audio recordings

Your task:
1. Assign a preliminary ESI level (1–5) based on documented urgency indicators. \
   If data is partial, use your best clinical judgment from what is present.
2. Identify any fields that are missing, implausible, or need verification — list them as notes.
3. Set valid=false ONLY under the critical-incompleteness condition above.
4. Write a brief documentary note for the nurse or triage clerk summarising what was reviewed, \
   what looks good, and what should be verified or completed — NOT a clinical diagnosis.

ESI levels (documentation reference only):
  1 — Immediate life-saving intervention indicated by documented signs
  2 — High-risk or severe distress documented
  3 — Multiple resources likely needed, stable vitals documented
  4 — One resource needed, no distress documented
  5 — No resources needed, minimal documented complaints

Always respond with a valid JSON object with these exact keys:
- "valid": boolean — false ONLY if main complaint AND vitals are both completely absent
- "missing_fields": list of field names that need attention (may be non-empty even when valid=true)
- "esi_level": integer 1–5 — always required, even when valid=false
- "analysis": 3–5 sentence documentary note for nursing/triage staff — what was reviewed, \
  ESI rationale, and any fields to verify — written as a documentation check, NOT a diagnosis

Return ONLY the JSON object, no markdown, no explanation.\
"""

_USER_PROMPT = """\
Review the following triage case documentation:

PATIENT:
{patient}

MAIN COMPLAINT:
{chief_complaint}
{attachments_section}"""


def _format_patient(patient: dict) -> str:
    lines = []
    for key, value in patient.items():
        label = key.replace("_", " ").title()
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value) if value else "None"
        lines.append(f"  {label}: {value}")
    return "\n".join(lines)


def _format_attachments(attachments: list[dict]) -> str:
    if not attachments:
        return ""
    parts = []
    for att in attachments:
        if not att.get("content", None):
            continue

        name = att.get("name", "unnamed")
        content = att.get("content", "")
        block = f"  [{name}]\n  Transcription: {content}"
        parts.append(block)
    return "\n\n".join(parts)


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _apply_danger_zone_escalation(patient: dict, esi_level: int, analysis: str) -> tuple[int, str]:
    """Decision Point D del ESI v5: reevalúa la decisión de A/B/C contra las
    vitales de danger-zone por edad, sin importar el nivel que haya propuesto
    el LLM. Solo puede subir el nivel a 2, nunca bajarlo."""
    age_years = esi_rules.age_years_from_date_of_birth(patient.get("date_of_birth"))
    danger_zone = esi_rules.danger_zone_flag(
        heartrate=_as_float(patient.get("pulse") or patient.get("heartrate")),
        resprate=_as_float(patient.get("respiratory_rate") or patient.get("resprate")),
        o2sat=_as_float(patient.get("oxygen_saturation_percent") or patient.get("o2sat")),
        age_years=age_years,
    )
    if danger_zone and esi_level > 2:
        note = (
            f" [Auto-escalated from ESI {esi_level} to ESI 2: vital signs are in the ESI v5 "
            "danger zone for this patient's age (Decision Point D).]"
        )
        return 2, analysis + note
    return esi_level, analysis


async def evaluate_triage(
    config: TriageAgentConfig,
    patient: dict,
    chief_complaint: str,
    attachments: list[dict],
) -> tuple[bool, list[str], int, str]:
    formatted_attachments = _format_attachments(attachments)
    attachments_section = (
        f"\nATTACHMENTS:\n{formatted_attachments}\n" if formatted_attachments else ""
    )
    prompt = _USER_PROMPT.format(
        patient=_format_patient(patient),
        chief_complaint=chief_complaint,
        attachments_section=attachments_section,
    )

    data = await call_llm(
        config,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2048,
    )

    valid: bool = bool(data.get("valid", False))
    missing_fields: list[str] = [str(f) for f in (data.get("missing_fields") or [])]
    raw_esi = data.get("esi_level") or 3
    esi_level = max(1, min(5, int(raw_esi)))
    analysis: str = data.get("analysis") or ""

    esi_level, analysis = _apply_danger_zone_escalation(patient, esi_level, analysis)

    return valid, missing_fields, esi_level, analysis
