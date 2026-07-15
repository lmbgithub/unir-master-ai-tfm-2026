import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from faster_whisper import WhisperModel

from urgenurse.agents.agent.llm_utils import call_llm, clamp_confidence, filter_ner

from .config import AsrAgentConfig

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a professional medical transcriptionist. Given a verbatim transcription of a medical \
audio recording, extract structured clinical information.

Always respond with a valid JSON object with EXACTLY these four keys:

"summary": concise clinical summary (2-4 sentences) derived from the transcription.

"ner": flat dictionary of named medical entities. Include ONLY keys explicitly present in the \
transcription; omit any that are absent or unclear. \
Allowed keys: patient_name, date, doctor_name, institution, diagnosis, \
medications (comma-separated), lab_values (comma-separated key:value pairs), \
allergies, dosage, instructions.

"sbar": object that MUST always contain all four sub-keys — infer from context if not explicit: \
situation (what is happening), background (relevant history), \
assessment (likely clinical picture), recommendation (suggested next step).

"confidence": float 0.0–1.0 reflecting extraction accuracy.

Return ONLY the JSON object, no markdown, no explanation.\
"""


@asynccontextmanager
async def whisper_loader(
    config: AsrAgentConfig,
) -> AsyncGenerator[WhisperModel, None]:
    def _load() -> WhisperModel:
        logger.info(
            "Loading Whisper model '%s' on %s (%s)…",
            config.whisper_model,
            config.whisper_device,
            config.whisper_compute_type,
        )
        model = WhisperModel(
            config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )
        logger.info("Whisper model ready.")
        return model

    model = await asyncio.to_thread(_load)
    try:
        yield model
    finally:
        del model
        logger.info("Whisper model released.")


async def analyze_audio(
    config: AsrAgentConfig,
    model: WhisperModel,
    file_path: str,
) -> tuple[str, str, dict[str, str], dict[str, str], float | None]:
    transcription = await _transcribe(model, file_path)

    if not transcription:
        return "", "", {}, {}, None

    data = await call_llm(
        config,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": transcription},
        ],
        max_tokens=768,
    )

    summary: str = data.get("summary") or ""
    ner = filter_ner(data.get("ner") or {})
    sbar: dict[str, str] = {k: str(v) for k, v in (data.get("sbar") or {}).items()}
    confidence = clamp_confidence(data.get("confidence"))

    return transcription, summary, ner, sbar, confidence


async def _transcribe(model: WhisperModel, file_path: str) -> str:
    def _run() -> str:
        segments, _ = model.transcribe(file_path, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments)

    return await asyncio.to_thread(_run)
