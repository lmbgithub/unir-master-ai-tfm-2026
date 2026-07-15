FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake \
    && rm -rf /var/lib/apt/lists/*

COPY agent/. /agent
COPY agent-triage/requirements.txt agent-triage/pyproject.toml ./
RUN pip install uv \
    && uv pip install --system -e /agent \
    && grep -v "llama-cpp-python" requirements.txt | uv pip install --system -r /dev/stdin

COPY agent-triage/src/ src/
RUN uv pip install --system --no-deps -e .

# ── dev: source mounted via volume; watchfiles restarts worker on change ──
FROM base AS dev

RUN uv pip install --system watchfiles
CMD ["watchfiles", "--filter", "python", "worker", "src/", "/agent"]

# ── prod ──────────────────────────────────────────────────────────────────
FROM base AS prod

CMD ["worker"]
