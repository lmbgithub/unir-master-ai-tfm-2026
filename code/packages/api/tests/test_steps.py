from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from urgenurse.api.main import create_app
from urgenurse.api.models import Base  # noqa: F401


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


@pytest.fixture
async def case_id(client: AsyncClient, auth_headers: None) -> str:
    resp = await client.post(
        "/cases",
        data={
            "patient_info": '{"name":"Test","gender":"male","date_of_birth":"1990-01-01","id_number":"99999999","blood_type":"O","blood_rh":true,"blood_pressure_systolic":120,"blood_pressure_diastolic":80,"weight":70,"height":175,"pulse":72}',
            "chief_complaint": "acute pain",
        },
    )
    return resp.json()["id"]


async def test_add_step_returns_201_with_created_status(client: AsyncClient, auth_headers: None, case_id: str) -> None:
    resp = await client.post(
        f"/cases/{case_id}/steps",
        json={"type": "triage", "description": "Initial triage"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "created"
    assert data["case_id"] == case_id


async def test_submit_step_transitions_to_pending(client: AsyncClient, auth_headers: None, case_id: str) -> None:
    step_resp = await client.post(
        f"/cases/{case_id}/steps",
        json={"type": "regular", "description": "extra step"},
    )
    step_id = step_resp.json()["id"]

    resp = await client.post(f"/cases/{case_id}/steps/{step_id}/submit")
    assert resp.status_code == 204


async def test_submit_step_twice_returns_409(client: AsyncClient, auth_headers: None, case_id: str) -> None:
    step_resp = await client.post(
        f"/cases/{case_id}/steps",
        json={"type": "regular", "description": "extra step"},
    )
    step_id = step_resp.json()["id"]

    await client.post(f"/cases/{case_id}/steps/{step_id}/submit")
    resp = await client.post(f"/cases/{case_id}/steps/{step_id}/submit")
    assert resp.status_code == 409


async def test_add_step_to_nonexistent_case_returns_404(client: AsyncClient, auth_headers: None) -> None:
    resp = await client.post(
        "/cases/00000000-0000-0000-0000-000000000000/steps",
        json={"type": "triage"},
    )
    assert resp.status_code == 404


async def test_add_handoff_step_returns_201(client: AsyncClient, auth_headers: None, case_id: str) -> None:
    resp = await client.post(
        f"/cases/{case_id}/steps",
        json={"type": "handoff", "description": "Transfer to cardiology"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["type"] == "handoff"
    assert data["status"] == "created"
    assert data["description"] == "Transfer to cardiology"


async def test_submit_handoff_step_transitions_to_pending(
    client: AsyncClient, auth_headers: None, case_id: str
) -> None:
    step_resp = await client.post(
        f"/cases/{case_id}/steps",
        json={"type": "handoff", "description": "Handoff to ICU"},
    )
    step_id = step_resp.json()["id"]
    resp = await client.post(f"/cases/{case_id}/steps/{step_id}/submit")
    assert resp.status_code == 204


async def test_update_status_to_in_progress_populates_started_at(
    client: AsyncClient, auth_headers: None, case_id: str
) -> None:
    create_resp = await client.post(
        f"/cases/{case_id}/steps",
        json={"type": "triage"},
    )
    step_id = create_resp.json()["id"]

    patch_resp = await client.patch(
        f"/cases/{case_id}/steps/{step_id}/status",
        json={"status": "in_progress"},
    )
    assert patch_resp.status_code == 200
    data = patch_resp.json()
    assert data["status"] == "in_progress"
    assert data["started_at"] is not None
