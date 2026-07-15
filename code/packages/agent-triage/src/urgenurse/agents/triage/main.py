from urgenurse.agents.agent import Agent
from urgenurse.agents.agent.requests import AgentRequest, AgentResponse
from .config import TriageAgentConfig
from .handlers import handle_triage
import logging

logger = logging.getLogger(__name__)

config = TriageAgentConfig()
agent = Agent(config)


@agent.subscribe(
    subject="triage.request",
    request=AgentRequest,
    response=AgentResponse,
    timeout=180,
)
async def triage_handler(agent: Agent, req: AgentRequest) -> AgentResponse:
    logger.info("triage_handler req=%s", req)
    return await handle_triage(agent, req)


def main() -> None:
    logger.info("Start agent triage =======================")
    agent.run()


if __name__ == "__main__":
    main()
