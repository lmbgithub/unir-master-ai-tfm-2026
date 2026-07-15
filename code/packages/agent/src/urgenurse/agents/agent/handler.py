import functools
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from .requests import AgentRequest, AgentResponse

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)

_Handler = Callable[["Agent", AgentRequest], Awaitable[AgentResponse]]


def agent_handler(fn: _Handler) -> _Handler:
    """Wrap a worker handler so any failure becomes a uniform error AgentResponse.

    Every worker shares the same contract: build a successful AgentResponse or,
    on any failure, reply with ok=False and the error message instead of letting
    the exception escape (which would leave the requester waiting on a timeout).
    """

    @functools.wraps(fn)
    async def wrapper(agent: "Agent", req: AgentRequest) -> AgentResponse:
        try:
            return await fn(agent, req)
        except Exception as exc:
            logger.exception("Handler %s failed for id=%s", fn.__name__, req.id)
            return AgentResponse(id=req.id, ok=False, error=str(exc))

    return wrapper
