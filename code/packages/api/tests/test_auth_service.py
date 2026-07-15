from datetime import UTC, datetime, timedelta

import pytest
from jose import jwt

from urgenurse.api.config import Settings
from urgenurse.api.services.auth_service import (
    create_auth_token,
    create_refresh_token,
    hash_password,
    verify_auth_token,
    verify_password,
    verify_refresh_token,
)

SETTINGS = Settings(jwt_secret="unit-test-secret", admin_user="admin", admin_password="secret")


def test_hash_and_verify_password() -> None:
    hashed = hash_password("mysecret")
    assert verify_password("mysecret", hashed)
    assert not verify_password("wrong", hashed)


def test_create_auth_token_returns_decodable_jwt() -> None:
    token = create_auth_token("user-123", SETTINGS)
    payload = jwt.decode(token, SETTINGS.jwt_secret, algorithms=[SETTINGS.jwt_algorithm])
    assert payload["sub"] == "user-123"
    assert payload["type"] == "auth"


def test_create_auth_token_expires_in_15min() -> None:
    before = datetime.now(UTC)
    token = create_auth_token("user-123", SETTINGS)
    payload = jwt.decode(token, SETTINGS.jwt_secret, algorithms=[SETTINGS.jwt_algorithm])
    exp = datetime.fromtimestamp(payload["exp"], tz=UTC)
    assert timedelta(minutes=14) < (exp - before) <= timedelta(minutes=15, seconds=5)


def test_create_refresh_token_expires_in_7days() -> None:
    before = datetime.now(UTC)
    token = create_refresh_token("user-123", SETTINGS)
    payload = jwt.decode(token, SETTINGS.jwt_secret, algorithms=[SETTINGS.jwt_algorithm])
    exp = datetime.fromtimestamp(payload["exp"], tz=UTC)
    assert timedelta(days=6, hours=23) < (exp - before) <= timedelta(days=7, seconds=5)


def test_verify_auth_token_returns_user_id() -> None:
    token = create_auth_token("user-abc", SETTINGS)
    assert verify_auth_token(token, SETTINGS) == "user-abc"


def test_verify_refresh_token_returns_user_id() -> None:
    token = create_refresh_token("user-abc", SETTINGS)
    assert verify_refresh_token(token, SETTINGS) == "user-abc"


def test_verify_auth_token_rejects_refresh_token() -> None:
    token = create_refresh_token("user-abc", SETTINGS)
    with pytest.raises(ValueError, match="wrong token type"):
        verify_auth_token(token, SETTINGS)


def test_verify_refresh_token_rejects_auth_token() -> None:
    token = create_auth_token("user-abc", SETTINGS)
    with pytest.raises(ValueError, match="wrong token type"):
        verify_refresh_token(token, SETTINGS)


def test_verify_auth_token_raises_on_wrong_secret() -> None:
    bad_settings = Settings(jwt_secret="wrong-secret")
    token = create_auth_token("user-abc", SETTINGS)
    with pytest.raises(ValueError, match="invalid token"):
        verify_auth_token(token, bad_settings)


def test_verify_auth_token_raises_on_expired() -> None:
    payload = {
        "sub": "user-abc",
        "type": "auth",
        "exp": datetime.now(UTC) - timedelta(seconds=1),
    }
    token = jwt.encode(payload, SETTINGS.jwt_secret, algorithm=SETTINGS.jwt_algorithm)
    with pytest.raises(ValueError, match="invalid token"):
        verify_auth_token(token, SETTINGS)


def test_verify_auth_token_raises_on_malformed() -> None:
    with pytest.raises(ValueError, match="invalid token"):
        verify_auth_token("not.a.token", SETTINGS)


def test_verify_auth_token_raises_when_sub_missing() -> None:
    payload = {"type": "auth", "exp": datetime.now(UTC) + timedelta(hours=1)}
    token = jwt.encode(payload, SETTINGS.jwt_secret, algorithm=SETTINGS.jwt_algorithm)
    with pytest.raises(ValueError, match="missing sub"):
        verify_auth_token(token, SETTINGS)
