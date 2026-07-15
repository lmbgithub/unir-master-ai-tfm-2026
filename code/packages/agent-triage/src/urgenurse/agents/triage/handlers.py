from typing import TYPE_CHECKING

from urgenurse.agents.agent import agent_handler
from urgenurse.agents.agent.requests import (
    AgentRequest,
    AgentRequestPayloadTriage,
    AgentResponse,
    AgentResponsePayloadTriage,
    ESILevels,
)

from .llm import evaluate_triage

if TYPE_CHECKING:
    from urgenurse.agents.agent import Agent


@agent_handler
async def handle_triage(agent: "Agent", req: AgentRequest) -> AgentResponse:
    payload: AgentRequestPayloadTriage = req.payload  # type: ignore[assignment]

    attachments = [
        {
            "name": t.name,
            "content": t.content,
            "summary": t.summary,
        }
        for t in (payload.attachments_transcriptions or [])
    ]

    valid, missing_fields, esi_level, analysis = await evaluate_triage(
        config=agent.config,  # type: ignore[arg-type]
        patient=payload.patient,
        chief_complaint=payload.description,
        attachments=attachments,
    )

    return AgentResponse(
        id=req.id,
        ok=True,
        payload=AgentResponsePayloadTriage(
            valid=valid,
            missing_fields=missing_fields or None,
            esi_level=ESILevels(esi_level),
            analysis=analysis,
        ),
    )
