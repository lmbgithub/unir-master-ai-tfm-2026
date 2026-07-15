from datetime import UTC, datetime, timedelta

from fastapi import Response
from jose import JWTError, jwt
from pwdlib import PasswordHash

from ..config import Settings

_pwd = PasswordHash.recommended()

AUTH_COOKIE = "auth"


def hash_password(plain: str) -> str:
    return _pwd.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd.verify(plain, hashed)


def create_auth_token(user_id: str, settings: Settings) -> str:
    payload = {
        "sub": user_id,
        "type": "auth",
        "exp": datetime.now(UTC) + timedelta(minutes=15),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_refresh_token(user_id: str, settings: Settings) -> str:
    payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": datetime.now(UTC) + timedelta(days=7),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _decode_token(token: str, expected_type: str, settings: Settings) -> str:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise ValueError("invalid token") from exc
    user_id: str | None = payload.get("sub")
    if user_id is None:
        raise ValueError("missing sub")
    if payload.get("type") != expected_type:
        raise ValueError("wrong token type")
    return user_id


def verify_auth_token(token: str, settings: Settings) -> str:
    return _decode_token(token, "auth", settings)


def verify_refresh_token(token: str, settings: Settings) -> str:
    return _decode_token(token, "refresh", settings)


def set_auth_cookie(
    response: Response,
    auth_token: str,
    refresh_token: str,
    settings: Settings,
) -> None:
    value = f"{auth_token}|{refresh_token}"
    response.set_cookie(
        key=AUTH_COOKIE,
        value=value,
        httponly=True,
        secure=settings.env == "production",
        samesite="lax",
        max_age=7 * 24 * 3600,
        path="/",
    )


def clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(key=AUTH_COOKIE, path="/")
