import pytest

from urgenurse.agents.agent.requests import AgentRequest, AgentRequestPayloadTriage


@pytest.fixture
def triage_request() -> AgentRequest:
    return AgentRequest(
        id="req-triage-001",
        payload=AgentRequestPayloadTriage(
            case_id="case-001",
            patient={"name": "Test Patient", "dob": "1980-01-01"},
            description="chest pain for 2 hours",
            attachments_transcriptions=None,
        ),
    )
