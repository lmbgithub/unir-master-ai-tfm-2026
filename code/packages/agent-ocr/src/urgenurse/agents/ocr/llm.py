import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
)
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)

from urgenurse.agents.agent.llm_utils import call_llm, clamp_confidence, filter_ner

from .config import OcrAgentConfig

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a medical document analysis assistant. Your task is to process medical documents \
(lab results, prescriptions, diagnoses) and extract structured information.

Always respond with a valid JSON object with these exact keys:
- "summary": concise clinical summary (2-4 sentences) highlighting key findings
- "ner": flat dictionary of named medical entities. Include ONLY keys for which the document \
contains explicit information. Omit any key that is absent, unclear, or not mentioned. \
Allowed keys: patient_name, date, doctor_name, institution, diagnosis, \
medications (comma-separated), lab_values (comma-separated key:value pairs), \
allergies, dosage, instructions
- "confidence": float between 0.0 and 1.0 reflecting your confidence in the extraction quality \
    (consider legibility, completeness, and clarity of the document)

Return ONLY the JSON object, no markdown, no explanation.\
"""

_ocr_lock = asyncio.Lock()


@asynccontextmanager
async def ocr_loader(config: OcrAgentConfig) -> AsyncGenerator[DocumentConverter, None]:
    def _load() -> DocumentConverter:
        logger.info("Loading Docling converter (langs=%s)…", config.docling_ocr_langs)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=config.docling_num_threads,
            device=AcceleratorDevice.CPU,
        )
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = RapidOcrOptions(
            lang=config.docling_ocr_langs,
            text_score=config.docling_text_score,
            force_full_page_ocr=config.docling_force_full_page_ocr,
        )
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = (
            TableFormerMode.FAST
            if config.docling_table_former_fast
            else TableFormerMode.ACCURATE
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
            }
        )
        logger.info("Docling converter ready.")
        return converter

    model = await asyncio.to_thread(_load)
    try:
        yield model
    finally:
        logger.info("Docling converter released.")


async def analyze_with_ocr(
    config: OcrAgentConfig,
    model: DocumentConverter,
    file_path: str,
) -> tuple[str, str, dict[str, str], float | None]:
    transcription = await _ocr_file(model, file_path)

    if not transcription:
        return "", "", {}, None

    data = await call_llm(
        config,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": transcription},
        ],
        max_tokens=512,
    )

    summary: str = data.get("summary") or ""
    ner = filter_ner(data.get("ner") or {})
    confidence = clamp_confidence(data.get("confidence"))

    return transcription, summary, ner, confidence


async def _ocr_file(model: DocumentConverter, file_path: str) -> str:
    def _run() -> str:
        result = model.convert(file_path)
        return result.document.export_to_text().strip()

    # Docling pipelines are not thread-safe; serialize concurrent requests
    async with _ocr_lock:
        return await asyncio.to_thread(_run)
