from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    nats_url: str = "nats://nats:4222"
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    admin_user: str
    admin_password: str
    storage_path: str = "/app/uploads"
    env: str = "development"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # pydantic-settings populates fields from env
