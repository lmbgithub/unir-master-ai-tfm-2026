import uuid

from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy import select

from ..dependencies import CurrentUser, DbSession, SettingsDep
from ..models.user import User
from ..schemas.api.auth import LoginRequest, UserResponse
from ..services.auth_service import (
    clear_auth_cookie,
    create_auth_token,
    create_refresh_token,
    set_auth_cookie,
    verify_password,
    verify_refresh_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=UserResponse)
async def login(
    body: LoginRequest,
    response: Response,
    settings: SettingsDep,
    db: DbSession,
) -> UserResponse:
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    auth_token = create_auth_token(str(user.user_id), settings)
    refresh_token = create_refresh_token(str(user.user_id), settings)
    set_auth_cookie(response, auth_token, refresh_token, settings)
    return UserResponse(user_id=user.user_id, email=user.email, role=user.role)


@router.post("/refresh", response_model=UserResponse)
async def refresh(
    request: Request,
    response: Response,
    settings: SettingsDep,
    db: DbSession,
) -> UserResponse:
    cookie = request.cookies.get("auth")
    if not cookie:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    parts = cookie.split("|")
    if len(parts) < 2:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid cookie format")
    try:
        user_id = verify_refresh_token(parts[1], settings)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    user = await db.get(User, uuid.UUID(user_id))
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    auth_token = create_auth_token(str(user.user_id), settings)
    new_refresh_token = create_refresh_token(str(user.user_id), settings)
    set_auth_cookie(response, auth_token, new_refresh_token, settings)
    return UserResponse(user_id=user.user_id, email=user.email, role=user.role)


@router.post("/logout")
async def logout(response: Response) -> dict[str, str]:
    clear_auth_cookie(response)
    return {"detail": "logged out"}


@router.get("/me", response_model=UserResponse)
async def me(current_user: CurrentUser) -> UserResponse:
    return UserResponse(
        user_id=current_user.user_id,
        email=current_user.email,
        role=current_user.role,
    )
