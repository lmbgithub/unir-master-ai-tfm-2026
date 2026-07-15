import pytest

from urgenurse.agents.agent.requests import AgentRequest, AgentRequestPayloadFile


@pytest.fixture
def audio_request() -> AgentRequest:
    return AgentRequest(
        id="req-asr-001",
        payload=AgentRequestPayloadFile(
            attachment_id="att-001",
            filename="voice.wav",
            mime_type="audio/wav",
            path="/data/attachments/test.wav",
        ),
    )
