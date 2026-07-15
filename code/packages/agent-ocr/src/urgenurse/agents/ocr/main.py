import logging

from urgenurse.agents.agent import Agent
from urgenurse.agents.agent.requests import AgentRequest, AgentResponse

from .config import OcrAgentConfig
from .handlers import handle_ocr
from .llm import ocr_loader

logger = logging.getLogger(__name__)

config = OcrAgentConfig()
agent = Agent(config, model_loader=ocr_loader)


@agent.subscribe(
    subject="attachment.document",
    request=AgentRequest,
    response=AgentResponse,
    timeout=300,
)
async def ocr_handler(agent: Agent, req: AgentRequest) -> AgentResponse:
    logger.info("ocr_handler ======================= %s", req)
    return await handle_ocr(agent, req)


def main() -> None:
    logger.info("Start agent ocr =======================")
    agent.run()


if __name__ == "__main__":
    main()
