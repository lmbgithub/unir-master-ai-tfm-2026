import os
import subprocess

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://placeholder:placeholder@localhost/placeholder")
os.environ.setdefault("JWT_SECRET", "test-secret-key")
os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("NATS_URL", "nats://localhost:4222")
os.environ.setdefault("ADMIN_USER", "admin@urgenurse.local")
os.environ.setdefault("ADMIN_PASSWORD", "secret")

# Docker Desktop on macOS uses a non-standard socket path; resolve it via the
# active context so testcontainers can reach the daemon.
if "DOCKER_HOST" not in os.environ:
    try:
        host = subprocess.check_output(
            ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if host:
            os.environ["DOCKER_HOST"] = host
    except Exception:
        pass

import pytest  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from testcontainers.postgres import PostgresContainer  # noqa: E402

from urgenurse.api.models import Base  # noqa: E402, F401
from urgenurse.api.models.user import User  # noqa: E402
from urgenurse.api.services.auth_service import hash_password  # noqa: E402


@pytest.fixture(scope="session")
def pg_url() -> str:
    with PostgresContainer("postgres:16-alpine") as pg:
        url = pg.get_connection_url().replace("psycopg2", "asyncpg")
        os.environ["DATABASE_URL"] = url
        yield url


@pytest.fixture(scope="session")
async def engine(pg_url: str) -> AsyncEngine:
    _engine = create_async_engine(pg_url)
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield _engine
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await _engine.dispose()


@pytest.fixture(scope="session")
async def test_user(engine: AsyncEngine) -> User:
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)  # type: ignore[call-overload]
    async with async_session() as session:
        user = User(
            email="admin@urgenurse.local",
            hashed_password=hash_password("secret"),
            role="admin",
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
    return user
