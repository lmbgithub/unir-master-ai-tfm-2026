import json
import logging

import httpx
from json_repair import repair_json

from .config import LLMAgentConfig

logger = logging.getLogger(__name__)

_NER_EMPTY = {"none", "null", "n/a", ""}


async def call_llm(
    config: LLMAgentConfig,
    messages: list[dict],
    max_tokens: int = 512,
) -> dict:
    """POST to /v1/chat/completions, strip markdown fences, parse and return JSON dict."""
    body = {
        "model": config.llm_model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    # Never log request/response bodies: they carry patient clinical data (PII).
    logger.debug(
        "Calling LLM model=%s messages=%d max_tokens=%d",
        config.llm_model,
        len(messages),
        max_tokens,
    )

    try:
        async with httpx.AsyncClient(timeout=config.llm_timeout) as client:
            resp = await client.post(f"{config.llm_url}/v1/chat/completions", json=body)
            resp.raise_for_status()
    except httpx.ConnectError as exc:
        raise ConnectionError(f"Cannot reach LLM at {config.llm_url}: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        raise ConnectionError(
            f"LLM returned {exc.response.status_code}: {exc.response.text}"
        ) from exc

    content = resp.json()["choices"][0]["message"]["content"].strip()

    if content.startswith("```"):
        content = content.split("```", 2)[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.rsplit("```", 1)[0].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Content may contain PII; keep the body itself at debug only.
        logger.warning("LLM returned malformed JSON, attempting repair")
        logger.debug("malformed content: %s", content)

    repaired = repair_json(content)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        logger.error("LLM returned unrepairable JSON")
        logger.debug("unrepairable content: %s", content)
        raise ValueError(f"LLM returned non-JSON: {exc}") from exc


def filter_ner(raw: dict) -> dict[str, str]:
    return {
        k: str(v)
        for k, v in raw.items()
        if v and str(v).strip().lower() not in _NER_EMPTY
    }


def clamp_confidence(raw: object) -> float | None:
    if raw is None:
        return None
    try:
        return max(0.0, min(1.0, float(raw)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
