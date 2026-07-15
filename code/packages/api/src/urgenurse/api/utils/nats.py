import nats
import nats.aio.msg
import nats.js.errors
from nats.js.api import StreamConfig
from pydantic import BaseModel

_STREAMS: list[StreamConfig] = [
    StreamConfig(name="requests", subjects=["request"]),
]


class NatsClient:
    def __init__(self) -> None:
        self._nc: nats.aio.client.Client | None = None

    async def connect(self, url: str) -> None:
        self._nc = await nats.connect(url)

    async def ensure_streams(self) -> None:
        if self._nc is None:
            return
        js = self._nc.jetstream()
        for cfg in _STREAMS:
            try:
                await js.find_stream_name_by_subject(cfg.subjects[0])
            except nats.js.errors.NotFoundError:
                await js.add_stream(config=cfg)

    async def drain(self) -> None:
        if self._nc is not None:
            await self._nc.drain()

    async def publish(self, subject: str, msg: BaseModel) -> None:
        if self._nc is None:
            return
        js = self._nc.jetstream()
        await js.publish(subject, msg.model_dump_json().encode())

    def jetstream(self) -> nats.js.client.JetStreamContext:
        assert self._nc is not None
        return self._nc.jetstream()

    async def request(self, subject: str, payload: bytes, timeout: float) -> nats.aio.msg.Msg:
        assert self._nc is not None
        return await self._nc.request(subject, payload, timeout=timeout)
