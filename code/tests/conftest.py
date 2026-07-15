import asyncio
import os

import httpx
import pytest_asyncio

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "secret")


async def _wait_for_api(url: str, retries: int = 10, delay: float = 2.0) -> None:
    async with httpx.AsyncClient() as client:
        for _ in range(retries):
            try:
                r = await client.get(f"{url}/health")
                if r.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            await asyncio.sleep(delay)
    raise RuntimeError(f"API at {url} did not become ready after {retries} retries")


@pytest_asyncio.fixture(scope="session")
async def api_client():
    await _wait_for_api(API_BASE_URL)
    async with httpx.AsyncClient(base_url=API_BASE_URL) as client:
        yield client
