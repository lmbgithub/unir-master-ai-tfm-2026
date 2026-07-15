import asyncio
import json
import logging
import signal
from typing import TYPE_CHECKING, Any

import nats

from urgenurse.agents.agent.config import AgentConfig
from urgenurse.agents.agent.errors import (
    AgentError,
    HandlerTimeoutError,
    MessageParseError,
)

if TYPE_CHECKING:
    from urgenurse.agents.agent.agent import Agent, _HandlerEntry

logger = logging.getLogger(__name__)


async def run_loop(
    config: AgentConfig,
    handlers: "dict[str, _HandlerEntry]",
    agent: "Agent",
) -> None:
    logger.info("run_loop")

    nc = await nats.connect(config.nats_url)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    async def _handle(msg: Any, subject: str, entry: "Any") -> None:
        req_id: str | None = None
        try:
            raw_id = json.loads(msg.data).get("id")
            req_id = str(raw_id) if raw_id is not None else None
        except Exception:
            pass

        try:
            response = await _dispatch(subject, msg.data, entry, agent)
            await msg.respond(response)
            logger.debug("Handled subject=%s response_size=%d", subject, len(response))
        except MessageParseError as exc:
            logger.warning("Parse error on subject=%s: %s", subject, exc)
            await msg.respond(_error_response(req_id, "parse error"))
        except (HandlerTimeoutError, AgentError) as exc:
            logger.error("Handler error on subject=%s: %s", subject, exc)
            await msg.respond(_error_response(req_id, str(exc)))

    for subject, entry in handlers.items():

        async def _cb(msg: Any, _subject: str = subject, _entry: Any = entry) -> None:
            await _handle(msg, _subject, _entry)

        await nc.subscribe(subject, cb=_cb)
        logger.info("Subscribed to subject=%s", subject)

    await stop_event.wait()
    await nc.drain()


async def _dispatch(subject: str, raw: bytes, entry: Any, agent: "Agent") -> bytes:
    req_model, _, timeout, handler = entry

    try:
        parsed = req_model.model_validate_json(raw)
    except Exception as exc:
        raise MessageParseError(f"Failed to parse message on {subject}: {exc}") from exc

    try:
        result = await asyncio.wait_for(handler(agent, parsed), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise HandlerTimeoutError(f"Handler exceeded {timeout}s on {subject}") from exc
    except Exception as exc:
        raise AgentError(f"Handler error on {subject}: {exc}") from exc

    return result.model_dump_json().encode()


def _error_response(req_id: Any, error: str) -> bytes:
    return json.dumps({"id": str(req_id), "ok": False, "error": error}).encode()
