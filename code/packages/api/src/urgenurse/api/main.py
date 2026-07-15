import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from .config import get_settings
from .utils.job_manager import JobManager
from .utils.nats import NatsClient
from .routers import attachments as attachments_router
from .routers import auth as auth_router
from .routers import cases as cases_router
from .routers import health as health_router
from .routers import steps as steps_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()

    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    app.state.session_factory = session_factory

    nats_client = NatsClient()
    await nats_client.connect(settings.nats_url)
    await nats_client.ensure_streams()
    app.state.nats = nats_client

    job_manager = JobManager(nats=nats_client, db=session_factory)
    app.state.job_manager = job_manager
    job_manager_task = asyncio.create_task(job_manager.run())

    try:
        yield
    finally:
        job_manager_task.cancel()
        await job_manager.close()
        await nats_client.drain()
        await engine.dispose()


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    app = FastAPI(
        title="UrgeNurse API",
        version="0.1.0",
        description="Local AI-assisted triage system for emergency departments.",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "auth", "description": "Login and token management"},
            {"name": "cases", "description": "Case lifecycle management"},
            {
                "name": "attachments",
                "description": "File uploads and transcription status",
            },
        ],
        swagger_ui_parameters={"persistAuthorization": True},
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_router.router)
    app.include_router(cases_router.router)
    app.include_router(steps_router.router)
    app.include_router(attachments_router.router)
    app.include_router(health_router.router)

    def custom_openapi() -> dict:  # type: ignore[type-arg]
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            tags=app.openapi_tags,
        )
        schema.setdefault("components", {}).setdefault("securitySchemes", {})["BearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
        for path in schema.get("paths", {}).values():
            for operation in path.values():
                operation.setdefault("security", [{"BearerAuth": []}])
        # Login endpoint is public — remove the security requirement
        schema["paths"].get("/auth/login", {}).get("post", {}).pop("security", None)
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]

    return app


app = create_app()
