from typing import TYPE_CHECKING

from urgenurse.agents.agent import agent_handler
from urgenurse.agents.agent.requests import (
    AgentRequest,
    AgentRequestPayloadFile,
    AgentResponse,
    AttachmentTranscriptions,
)

from .llm import analyze_audio

if TYPE_CHECKING:
    from urgenurse.agents.agent import Agent


@agent_handler
async def handle_asr(agent: "Agent", req: AgentRequest) -> AgentResponse:
    payload: AgentRequestPayloadFile = req.payload  # type: ignore[assignment]

    transcription, summary, ner, sbar, confidence = await analyze_audio(
        config=agent.config,  # type: ignore[arg-type]
        model=agent._model,
        file_path=payload.path,
    )

    return AgentResponse(
        id=req.id,
        ok=True,
        payload=AttachmentTranscriptions(
            name=payload.filename,
            content=transcription,
            summary=summary,
            ner=ner,
            sbar=sbar,
            confidence=confidence,
        ),
    )
