from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import sessionmaker

from urgenurse.api.main import create_app
from urgenurse.api.models.user import User


@asynccontextmanager
async def _no_lifespan(app):  # type: ignore[no-untyped-def]
    yield


@pytest.fixture
async def client(engine: AsyncEngine, test_user: User) -> AsyncGenerator[AsyncClient, None]:
    app = create_app()
    app.router.lifespan_context = _no_lifespan  # type: ignore[assignment]

    # Inject the test DB session factory into app state
    app.state.session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


async def test_login_correct_credentials(client: AsyncClient) -> None:
    resp = await client.post("/auth/login", json={"email": "admin@urgenurse.local", "password": "secret"})
    assert resp.status_code == 200
    body = resp.json()
    assert "user_id" in body
    assert "email" in body
    assert "role" in body
    assert "access_token" not in body
    cookie = resp.cookies.get("auth")
    assert cookie is not None


async def test_login_wrong_credentials(client: AsyncClient) -> None:
    resp = await client.post("/auth/login", json={"email": "admin@urgenurse.local", "password": "wrong"})
    assert resp.status_code == 401


async def test_login_wrong_email(client: AsyncClient) -> None:
    resp = await client.post("/auth/login", json={"email": "unknown@urgenurse.local", "password": "secret"})
    assert resp.status_code == 401


async def test_login_missing_fields(client: AsyncClient) -> None:
    resp = await client.post("/auth/login", json={"email": "admin@urgenurse.local"})
    assert resp.status_code == 422


async def test_protected_route_without_token(client: AsyncClient) -> None:
    resp = await client.get("/health/protected")
    assert resp.status_code in (401, 403, 404)


async def test_protected_route_with_valid_token(client: AsyncClient) -> None:
    login = await client.post("/auth/login", json={"email": "admin@urgenurse.local", "password": "secret"})
    auth_cookie = login.cookies.get("auth")
    resp = await client.get("/health/protected", cookies={"auth": auth_cookie})
    assert resp.status_code == 200
    assert resp.json()["user"] == "admin@urgenurse.local"


async def test_protected_route_with_invalid_token(client: AsyncClient) -> None:
    resp = await client.get("/health/protected", cookies={"auth": "bad.token|bad.refresh"})
    assert resp.status_code == 401


async def test_health_public(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_refresh_token(client: AsyncClient) -> None:
    login = await client.post("/auth/login", json={"email": "admin@urgenurse.local", "password": "secret"})
    assert login.status_code == 200
    auth_cookie = login.cookies.get("auth")
    resp = await client.post("/auth/refresh", cookies={"auth": auth_cookie})
    assert resp.status_code == 200
    assert resp.cookies.get("auth") is not None


async def test_logout(client: AsyncClient) -> None:
    login = await client.post("/auth/login", json={"email": "admin@urgenurse.local", "password": "secret"})
    auth_cookie = login.cookies.get("auth")
    resp = await client.post("/auth/logout", cookies={"auth": auth_cookie})
    assert resp.status_code == 200


async def test_me_authenticated(client: AsyncClient) -> None:
    login = await client.post("/auth/login", json={"email": "admin@urgenurse.local", "password": "secret"})
    auth_cookie = login.cookies.get("auth")
    resp = await client.get("/auth/me", cookies={"auth": auth_cookie})
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "admin@urgenurse.local"


async def test_me_unauthenticated(client: AsyncClient) -> None:
    resp = await client.get("/auth/me")
    assert resp.status_code == 401
