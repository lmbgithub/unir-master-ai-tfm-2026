import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from urgenurse.api.utils.nats import NatsClient
from urgenurse.api.main import create_app
from urgenurse.api.models import Base  # noqa: F401


@asynccontextmanager
async def _no_lifespan(app):  # type: ignore[no-untyped-def]
    yield


@pytest.fixture
async def storage_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
async def mock_nats() -> NatsClient:
    client = NatsClient()
    client.publish = AsyncMock()  # type: ignore[method-assign]
    return client


@pytest.fixture
async def client(
    engine: AsyncEngine, storage_dir: str, mock_nats: NatsClient, test_user: object
) -> AsyncGenerator[AsyncClient, None]:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    from urgenurse.api.config import Settings, get_settings
    from urgenurse.api.dependencies import get_nats

    original = get_settings()

    app = create_app()
    app.router.lifespan_context = _no_lifespan  # type: ignore[assignment]
    app.state.session_factory = session_factory
    app.state.job_manager = AsyncMock()
    app.state.nats = mock_nats
    app.dependency_overrides[get_settings] = lambda: Settings(
        database_url=original.database_url,
        nats_url=original.nats_url,
        jwt_secret=original.jwt_secret,
        jwt_algorithm=original.jwt_algorithm,
        admin_user=original.admin_user,
        admin_password=original.admin_password,
        storage_path=storage_dir,
    )
    app.dependency_overrides[get_nats] = lambda: mock_nats

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def auth_headers(client: AsyncClient) -> None:
    resp = await client.post("/auth/login", json={"email": "admin@urgenurse.local", "password": "secret"})
    client.cookies.set("auth", resp.cookies["auth"])


@pytest.fixture
async def step_id(client: AsyncClient, auth_headers: None) -> str:
    case_resp = await client.post(
        "/cases",
        data={
            "patient_info": '{"name":"Test","gender":"male","date_of_birth":"1990-01-01","id_number":"99999999","blood_type":"O","blood_rh":true,"blood_pressure_systolic":120,"blood_pressure_diastolic":80,"weight":70,"height":175,"pulse":72}',
            "chief_complaint": "acute pain",
        },
    )
    case_id = case_resp.json()["id"]
    step_resp = await client.post(
        f"/cases/{case_id}/steps",
        json={"type": "triage"},
    )
    return step_resp.json()["id"]


async def test_upload_image_returns_201(
    client: AsyncClient,
    auth_headers: None,
    step_id: str,
) -> None:
    fake_case_id = "00000000-0000-0000-0000-000000000001"
    resp = await client.post(
        f"/cases/{fake_case_id}/steps/{step_id}/attachments",
        data={"kind": "image"},
        files={"file": ("scan.png", b"data", "image/png")},
    )
    assert resp.status_code == 201
    assert resp.json()["kind"] == "image"


async def test_upload_audio_returns_201(
    client: AsyncClient,
    auth_headers: None,
    step_id: str,
) -> None:
    fake_case_id = "00000000-0000-0000-0000-000000000001"
    resp = await client.post(
        f"/cases/{fake_case_id}/steps/{step_id}/attachments",
        data={"kind": "audio"},
        files={"file": ("handover.wav", b"audio", "audio/wav")},
    )
    assert resp.status_code == 201
    assert resp.json()["kind"] == "audio"
