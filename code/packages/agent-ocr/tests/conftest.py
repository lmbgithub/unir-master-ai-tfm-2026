import pytest

from urgenurse.agents.agent.requests import AgentRequest, AgentRequestPayloadFile


@pytest.fixture
def image_request() -> AgentRequest:
    return AgentRequest(
        id="req-001",
        payload=AgentRequestPayloadFile(
            attachment_id="att-001",
            filename="scan.jpg",
            mime_type="image/jpeg",
            path="/data/attachments/scan.jpg",
        ),
    )


@pytest.fixture
def pdf_request() -> AgentRequest:
    return AgentRequest(
        id="req-002",
        payload=AgentRequestPayloadFile(
            attachment_id="att-002",
            filename="report.pdf",
            mime_type="application/pdf",
            path="/data/attachments/report.pdf",
        ),
    )


@pytest.fixture
def audio_request() -> AgentRequest:
    return AgentRequest(
        id="req-003",
        payload=AgentRequestPayloadFile(
            attachment_id="att-003",
            filename="voice.wav",
            mime_type="audio/wav",
            path="/data/attachments/voice.wav",
        ),
    )
