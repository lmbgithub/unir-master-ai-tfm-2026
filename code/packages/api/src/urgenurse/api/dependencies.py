import uuid
from collections.abc import AsyncIterator
from typing import Annotated
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from .services.auth_service import verify_auth_token
from .utils.nats import NatsClient
from .config import Settings, get_settings
from .utils.job_manager import JobManager
from .models.user import User


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    factory = request.app.state.session_factory
    async with factory() as session:
        yield session


async def get_current_user(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    cookie = request.cookies.get("auth")
    if not cookie:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    auth_token = cookie.split("|")[0]
    try:
        user_id = verify_auth_token(auth_token, settings)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    user = await db.get(User, uuid.UUID(user_id))
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def get_nats(request: Request) -> NatsClient:
    return request.app.state.nats


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


SettingsDep = Annotated[Settings, Depends(get_settings)]
DbSession = Annotated[AsyncSession, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]
NatsDep = Annotated[NatsClient, Depends(get_nats)]
JobManagerDep = Annotated[JobManager, Depends(get_job_manager)]
