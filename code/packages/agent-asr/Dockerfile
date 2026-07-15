FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY agent /agent

COPY agent-asr/requirements.txt agent-asr/pyproject.toml ./
RUN pip install uv \
    && uv pip install --system -e /agent \
    && uv pip install --system -r requirements.txt

COPY agent-asr/src/ src/
RUN uv pip install --system --no-deps -e .

# ── dev: source mounted via volume; watchfiles restarts worker on change ──
FROM base AS dev

RUN uv pip install --system watchfiles
CMD ["watchfiles", "--filter", "python", "worker", "src/", "/agent"]

# ── prod ──────────────────────────────────────────────────────────────────
FROM base AS prod

CMD ["worker"]
