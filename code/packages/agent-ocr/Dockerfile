FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY agent /agent

COPY agent-ocr/requirements.txt agent-ocr/pyproject.toml ./
RUN pip install uv \
    && uv pip install --system -e /agent \
    && uv pip install --system -r requirements.txt

COPY agent-ocr/src/ src/
RUN uv pip install --system --no-deps -e .

# ── pre-download Docling model cache so first run is offline/fast ────────────
# Only the models our pipeline actually loads: layout + TableFormer + RapidOCR.
# EasyOCR (PyTorch) is left out to keep the image small and CPU-RAM low, and the
# unused code-formula / picture-classifier models are skipped too.
RUN python -c "from docling.utils.model_downloader import download_models; download_models(with_easyocr=False, with_code_formula=False, with_picture_classifier=False)"

# ── dev: source mounted via volume; watchfiles restarts worker on change ──
FROM base AS dev

RUN uv pip install --system watchfiles
CMD ["watchfiles", "--filter", "python", "worker", "src/", "/agent"]

# ── prod ──────────────────────────────────────────────────────────────────
FROM base AS prod

CMD ["worker"]
