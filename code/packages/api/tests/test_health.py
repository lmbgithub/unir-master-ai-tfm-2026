from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from urgenurse.api.main import create_app
from urgenurse.api.models import Base  # noqa: F401


@asynccontextmanager
async def _no_lifespan(app):  # type: ignore[no-untyped-def]
    yield


@pytest.fixture
async def client(engine: AsyncEngine) -> AsyncGenerator[AsyncClient, None]:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    app = create_app()
    app.router.lifespan_context = _no_lifespan  # type: ignore[assignment]
    app.state.session_factory = session_factory
    app.state.nats = None

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


async def test_health_returns_ok(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_health_protected_without_token_returns_401(client: AsyncClient) -> None:
    resp = await client.get("/health/protected")
    assert resp.status_code == 401


async def test_health_protected_with_token_returns_ok(client: AsyncClient, test_user: object) -> None:
    login = await client.post("/auth/login", json={"email": "admin@urgenurse.local", "password": "secret"})
    auth_cookie = login.cookies["auth"]
    resp = await client.get("/health/protected", cookies={"auth": auth_cookie})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["user"] == "admin@urgenurse.local"
