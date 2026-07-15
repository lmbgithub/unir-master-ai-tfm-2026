import logging

from urgenurse.agents.agent import Agent
from urgenurse.agents.agent.requests import AgentRequest, AgentResponse

from .config import AsrAgentConfig
from .handlers import handle_asr
from .llm import whisper_loader

logger = logging.getLogger(__name__)

config = AsrAgentConfig()
agent = Agent(config, model_loader=whisper_loader)


@agent.subscribe(
    subject="attachment.audio",
    request=AgentRequest,
    response=AgentResponse,
    timeout=120,
)
async def asr_handler(agent: Agent, req: AgentRequest) -> AgentResponse:
    logger.info("asr_handler ======================= %s", req)
    return await handle_asr(agent, req)


def main() -> None:
    logger.info("Start agent asr =======================")
    agent.run()


if __name__ == "__main__":
    main()
