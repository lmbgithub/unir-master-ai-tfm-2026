from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from urgenurse.api.main import create_app
from urgenurse.api.models import Base  # noqa: F401

import pytest


@asynccontextmanager
async def _no_lifespan(app):  # type: ignore[no-untyped-def]
    yield


@pytest.fixture
async def client(engine: AsyncEngine, test_user: object) -> AsyncGenerator[AsyncClient, None]:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    app = create_app()
    app.router.lifespan_context = _no_lifespan  # type: ignore[assignment]
    app.state.session_factory = session_factory
    app.state.job_manager = AsyncMock()
    app.state.nats = None

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def auth_headers(client: AsyncClient) -> None:
    resp = await client.post("/auth/login", json={"email": "admin@urgenurse.local", "password": "secret"})
    client.cookies.set("auth", resp.cookies["auth"])


async def test_create_case_phase_is_triage(client: AsyncClient, auth_headers: None) -> None:
    resp = await client.post(
        "/cases",
        data={
            "patient_info": '{"name":"Ana","gender":"female","date_of_birth":"1990-01-01","id_number":"11111111","blood_type":"A","blood_rh":true,"blood_pressure_systolic":120,"blood_pressure_diastolic":80,"weight":60,"height":165,"pulse":70}',
            "chief_complaint": "chest pain",
        },
    )
    assert resp.status_code == 201
    assert resp.json()["phase"] == "triage"


async def test_get_nonexistent_case_returns_404(client: AsyncClient, auth_headers: None) -> None:
    resp = await client.get("/cases/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


async def test_update_phase_persisted(client: AsyncClient, auth_headers: None) -> None:
    create_resp = await client.post(
        "/cases",
        data={
            "patient_info": '{"name":"Bob","gender":"male","date_of_birth":"1985-05-15","id_number":"22222222","blood_type":"B","blood_rh":false,"blood_pressure_systolic":130,"blood_pressure_diastolic":85,"weight":80,"height":180,"pulse":75}',
            "chief_complaint": "severe headache",
        },
    )
    case_id = create_resp.json()["id"]

    patch_resp = await client.patch(
        f"/cases/{case_id}/phase",
        json={"phase": "triage_validation"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["phase"] == "triage_validation"

    get_resp = await client.get(f"/cases/{case_id}")
    assert get_resp.json()["phase"] == "triage_validation"


async def test_confirm_triage_sets_pending_care(client: AsyncClient, auth_headers: None) -> None:
    create_resp = await client.post(
        "/cases",
        data={
            "patient_info": '{"name":"Carol","gender":"female","date_of_birth":"2000-03-20","id_number":"33333333","blood_type":"O","blood_rh":true,"blood_pressure_systolic":110,"blood_pressure_diastolic":70,"weight":55,"height":160,"pulse":80}',
            "chief_complaint": "high fever",
        },
    )
    case_id = create_resp.json()["id"]

    await client.patch(
        f"/cases/{case_id}/phase",
        json={"phase": "triage_validation"},
    )

    confirm_resp = await client.post(f"/cases/{case_id}/confirm-triage")
    assert confirm_resp.status_code == 200
    assert confirm_resp.json()["phase"] == "pending_care"
