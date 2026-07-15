FROM python:3.12-slim AS base

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install uv && uv pip install --system -r requirements.txt

COPY alembic/ alembic/
COPY alembic.ini ./
COPY seeders/ seeders/

EXPOSE 8000

# ── dev: source mounted via volume; uvicorn reloads on change ──────────────
FROM base AS dev

COPY src/ src/
RUN uv pip install --system --no-deps -e .

CMD ["sh", "-c", "alembic upgrade head && uvicorn urgenurse.api.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir src"]

# ── prod ───────────────────────────────────────────────────────────────────
FROM base AS prod

COPY src/ src/
RUN uv pip install --system --no-deps -e .

CMD ["sh", "-c", "alembic upgrade head && uvicorn urgenurse.api.main:app --host 0.0.0.0 --port 8000"]
