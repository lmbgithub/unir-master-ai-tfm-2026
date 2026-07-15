import asyncio
import io

import httpx

from .conftest import ADMIN_PASSWORD, ADMIN_USER

POLL_INTERVAL = 2.0
POLL_TIMEOUT = 30.0


async def _poll(
    client: httpx.AsyncClient,
    path: str,
    headers: dict,
    predicate,
    timeout: float = POLL_TIMEOUT,
):
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        r = await client.get(path, headers=headers)
        r.raise_for_status()
        data = r.json()
        if predicate(data):
            return data
        if asyncio.get_event_loop().time() >= deadline:
            raise TimeoutError(f"Condition not met for {path}: last={data}")
        await asyncio.sleep(POLL_INTERVAL)


async def test_happy_path(api_client: httpx.AsyncClient):
    # 1. Login
    r = await api_client.post(
        "/auth/login",
        json={"username": ADMIN_USER, "password": ADMIN_PASSWORD},
    )
    assert r.status_code == 200
    token = r.json()["access_token"]
    auth = {"Authorization": f"Bearer {token}"}

    # 2. Create case
    r = await api_client.post(
        "/cases",
        json={
            "patient_info": {"name": "Test Patient", "dob": "1990-01-01"},
            "chief_complaint": "Chest pain",
        },
        headers=auth,
    )
    assert r.status_code == 201
    case = r.json()
    case_id = case["id"]
    assert case["phase"] == "triage"

    # 3. Add triage step
    r = await api_client.post(
        f"/cases/{case_id}/steps",
        json={"type": "triage", "description": "Initial triage"},
        headers=auth,
    )
    assert r.status_code == 201
    step_id = r.json()["id"]

    # 4. Upload small PNG fixture
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
        b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    r = await api_client.post(
        f"/cases/{case_id}/steps/{step_id}/attachments",
        files={"file": ("fixture.png", io.BytesIO(png_bytes), "image/png")},
        data={"kind": "image"},
        headers=auth,
    )
    assert r.status_code == 201
    attachment = r.json()
    attachment_id = attachment["id"]
    assert attachment["status"] == "pending"

    # 5. Poll attachment until done
    attachment = await _poll(
        api_client,
        f"/attachments/{attachment_id}",
        auth,
        lambda d: d["status"] in ("done", "error"),
    )
    assert attachment["status"] == "done"

    # 6. Poll case until phase=triage_validation
    case = await _poll(
        api_client,
        f"/cases/{case_id}",
        auth,
        lambda d: d["phase"] == "triage_validation",
    )

    # 7. Confirm triage → pending_care
    r = await api_client.post(
        f"/cases/{case_id}/confirm-triage",
        headers=auth,
    )
    assert r.status_code == 200
    assert r.json()["phase"] == "pending_care"
