import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any

from pydantic import BaseModel

from urgenurse.agents.agent.config import AgentConfig
from urgenurse.agents.agent.process import ProcessManager
from urgenurse.agents.agent.runtime import run_loop

logger = logging.getLogger(__name__)

_ModelLoader = Callable[[AgentConfig], AsyncGenerator[Any, None]]
_HandlerEntry = tuple[type[BaseModel], type[BaseModel], int, Callable[..., Any]]


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        model_loader: _ModelLoader | None = None,
    ) -> None:
        self.config = config
        self._model: Any = None
        self._handlers: dict[str, _HandlerEntry] = {}
        self._model_loader = model_loader

    def subscribe(
        self,
        subject: str,
        request: type[BaseModel],
        response: type[BaseModel],
        timeout: int = 30,
    ) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self._handlers[subject] = (request, response, timeout, fn)
            return fn

        return decorator

    def run(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%H:%M:%S",
        )

        # Extract primitives so the subprocess closure only pickles plain data,
        # not the Agent instance itself (which holds unpicklable async state).
        config = self.config
        handlers = self._handlers
        model_loader = self._model_loader

        def _run_process() -> None:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt="%H:%M:%S",
            )
            agent = Agent(config, model_loader)
            agent._handlers = handlers
            asyncio.run(_run_with_model(agent))

        pm = ProcessManager(target=_run_process)
        pm.start()
        pm.monitor()


async def _run_with_model(agent: "Agent") -> None:
    if agent._model_loader is not None:
        async with agent._model_loader(agent.config) as model:
            agent._model = model
            await run_loop(agent.config, agent._handlers, agent)
            agent._model = None
    else:
        await run_loop(agent.config, agent._handlers, agent)
