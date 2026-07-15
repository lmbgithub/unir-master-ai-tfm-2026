import pathlib
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
    app.dependency_overrides[get_nats] = lambda: mock_nats
    app.dependency_overrides[get_settings] = lambda: Settings(
        database_url=original.database_url,
        nats_url=original.nats_url,
        jwt_secret=original.jwt_secret,
        jwt_algorithm=original.jwt_algorithm,
        admin_user=original.admin_user,
        admin_password=original.admin_password,
        storage_path=storage_dir,
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def auth_headers(client: AsyncClient) -> None:
    resp = await client.post("/auth/login", json={"email": "admin@urgenurse.local", "password": "secret"})
    client.cookies.set("auth", resp.cookies["auth"])


@pytest.fixture
async def case_and_step(client: AsyncClient, auth_headers: None) -> tuple[str, str]:
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
    return case_id, step_resp.json()["id"]


@pytest.fixture
async def step_id(case_and_step: tuple[str, str]) -> str:
    return case_and_step[1]


async def test_upload_attachment_returns_201_with_pending_status(
    client: AsyncClient, auth_headers: None, step_id: str, storage_dir: str
) -> None:
    fake_case_id = "00000000-0000-0000-0000-000000000001"
    resp = await client.post(
        f"/cases/{fake_case_id}/steps/{step_id}/attachments",
        data={"kind": "image"},
        files={"file": ("report.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "pending"
    assert data["original_filename"] == "report.txt"
    assert data["kind"] == "image"

    stored = pathlib.Path(data["storage_path"])
    assert stored.exists()
    assert stored.read_bytes() == b"hello world"


async def test_get_attachment_returns_correct_fields(client: AsyncClient, auth_headers: None, step_id: str) -> None:
    fake_case_id = "00000000-0000-0000-0000-000000000001"
    upload_resp = await client.post(
        f"/cases/{fake_case_id}/steps/{step_id}/attachments",
        data={"kind": "pdf"},
        files={"file": ("note.txt", b"data", "text/plain")},
    )
    attachment_id = upload_resp.json()["id"]

    resp = await client.get(
        f"/attachments/{attachment_id}",
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == attachment_id
    assert data["original_filename"] == "note.txt"


async def test_get_nonexistent_attachment_returns_404(client: AsyncClient, auth_headers: None) -> None:
    resp = await client.get(
        "/attachments/00000000-0000-0000-0000-000000000000",
    )
    assert resp.status_code == 404


async def test_download_attachment_returns_file_bytes(client: AsyncClient, auth_headers: None, step_id: str) -> None:
    fake_case_id = "00000000-0000-0000-0000-000000000001"
    upload_resp = await client.post(
        f"/cases/{fake_case_id}/steps/{step_id}/attachments",
        data={"kind": "audio"},
        files={"file": ("voice.wav", b"RIFF....WAV", "audio/wav")},
    )
    attachment_id = upload_resp.json()["id"]

    resp = await client.get(f"/attachments/{attachment_id}/download")
    assert resp.status_code == 200
    assert resp.content == b"RIFF....WAV"
    assert "audio/wav" in resp.headers["content-type"]


async def test_download_nonexistent_attachment_returns_404(client: AsyncClient, auth_headers: None) -> None:
    resp = await client.get("/attachments/00000000-0000-0000-0000-000000000000/download")
    assert resp.status_code == 404


async def test_upload_attachment_to_submitted_step_returns_409(
    client: AsyncClient, auth_headers: None, case_and_step: tuple[str, str]
) -> None:
    case_id, step_id = case_and_step
    await client.post(f"/cases/{case_id}/steps/{step_id}/submit")

    resp = await client.post(
        f"/cases/{case_id}/steps/{step_id}/attachments",
        data={"kind": "image"},
        files={"file": ("report.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 409
