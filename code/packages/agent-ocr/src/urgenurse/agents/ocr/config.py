from urgenurse.agents.agent import LLMAgentConfig


class OcrAgentConfig(LLMAgentConfig):
    # Docling runs CPU-only here; bound the thread pool so a single large page
    # does not saturate every core on the shared host.
    docling_num_threads: int = 4
    # RapidOCR (ONNX) is the lightest engine Docling ships: no PyTorch, models
    # bundled in the wheel (offline), far less RAM than the EasyOCR default.
    # It only supports "english" and "chinese".
    docling_ocr_langs: list[str] = ["english"]
    # Drop low-confidence text regions (mirrors the old recognition threshold).
    docling_text_score: float = 0.5
    # Run OCR on full pages instead of only image regions. Off by default so
    # born-digital PDFs reuse the embedded text layer (faster, more accurate).
    docling_force_full_page_ocr: bool = False
    # FAST TableFormer uses noticeably less memory than ACCURATE while still
    # recovering the tabular lab-result layouts we care about.
    docling_table_former_fast: bool = True
