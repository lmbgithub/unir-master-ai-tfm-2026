import asyncio
import logging
from collections.abc import Callable
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from urgenurse.agents.agent.requests import (
    AgentRequest,
    AgentRequestPayloadFile,
    AgentRequestPayloadTriage,
    AgentResponse,
    AttachmentTranscriptions,
    AgentResponsePayloadTriage,
)

from .nats import NatsClient
from ..enums import (
    AttachmentKind,
    AttachmentStatus,
    CasePhase,
    CaseStepStatus,
    CaseStepType,
)
from ..models.attachment import Attachment
from ..models.case import Case
from ..models.case_step import CaseStep

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 600.0
MAX_RETRIES = 3


class Request(BaseModel):
    case_id: str
    step_id: str


class JobManager:
    def __init__(self, nats: NatsClient, db: Callable) -> None:
        self._nats = nats
        self._db = db
        self._subject = "request"
        self._stream = "requests"
        self._durable = "request"

    async def close(self) -> None:
        pass

    async def run(self) -> None:
        await self._recover_orphaned()
        while True:
            try:
                await self._run_consumer()
            except Exception:
                logger.exception("job_manager consumer error — restarting in 2s")
                await asyncio.sleep(2)

    async def _recover_orphaned(self) -> None:
        """Reset work left mid-flight by a previous crash or restart to error.

        A worker segfault or a service restart can leave an attachment in
        ``processing`` (or a step in ``in_progress``) with no coroutine driving
        it, so nothing ever transitions it to a terminal state.
        """
        message = "Processing interrupted: worker crashed or service restarted"
        try:
            async with self._db() as db:
                steps = (
                    (
                        await db.execute(
                            select(CaseStep)
                            .where(CaseStep.status == CaseStepStatus.in_progress)
                            .options(selectinload(CaseStep.attachments), selectinload(CaseStep.case))
                        )
                    )
                    .scalars()
                    .all()
                )
                attachments = (
                    (await db.execute(select(Attachment).where(Attachment.status == AttachmentStatus.processing)))
                    .scalars()
                    .all()
                )
                if not steps and not attachments:
                    return

                for attachment in attachments:
                    attachment.status = AttachmentStatus.error
                    if not attachment.transcription:
                        attachment.transcription = message
                for step in steps:
                    step.status = CaseStepStatus.error
                    step.error_message = step.error_message or message
                    if step.case is not None:
                        step.case.phase = CasePhase.error
                await db.commit()
                logger.warning(
                    "Recovered %d orphaned step(s) and %d orphaned attachment(s) to error",
                    len(steps),
                    len(attachments),
                )
        except Exception:
            logger.exception("Failed to recover orphaned work on startup")

    async def queue_request(self, case_id: str, step_id: str) -> None:
        js = self._nats.jetstream()
        request = Request(case_id=case_id, step_id=step_id)
        await js.publish(self._subject, request.model_dump_json().encode(), stream=self._stream)

    async def _run_consumer(self) -> None:
        js = self._nats.jetstream()
        try:
            ci = await js.consumer_info(self._stream, self._durable)
            if ci.config.deliver_subject:
                logger.warning(
                    "Deleting stale push consumer durable=%s (deliver_subject=%s)",
                    self._durable,
                    ci.config.deliver_subject,
                )
                await js.delete_consumer(self._stream, self._durable)
        except Exception:
            pass  # consumer doesn't exist yet — that's fine
        sub = await js.pull_subscribe(self._subject, durable=self._durable, stream=self._stream)
        logger.info(
            "job_manager pull consumer ready stream=%s subject=%s durable=%s",
            self._stream,
            self._subject,
            self._durable,
        )
        while True:
            try:
                msgs = await sub.fetch(1, timeout=5)
            except Exception:
                # fetch timeout or transient error — just loop
                continue
            for raw in msgs:
                request: Request | None = None
                try:
                    request = Request.model_validate_json(raw.data)
                    await self._process_request(request)
                except Exception as exc:
                    logger.exception("Failed to process request message")
                    if request is not None:
                        await self._mark_request_failed(request, exc)
                finally:
                    await raw.ack()

    async def _mark_request_failed(self, request: Request, exc: Exception) -> None:
        """Record a terminal error state when _process_request fails unexpectedly.

        Uses a fresh session so a poisoned session from the original failure
        cannot prevent the error from being persisted.
        """
        try:
            async with self._db() as db:
                result = await db.execute(
                    select(CaseStep)
                    .where(CaseStep.id == UUID(request.step_id))
                    .options(selectinload(CaseStep.attachments), selectinload(CaseStep.case))
                )
                step = result.scalar_one_or_none()
                if step is None:
                    return
                for attachment in step.attachments:
                    if attachment.status == AttachmentStatus.processing:
                        attachment.status = AttachmentStatus.error
                        if not attachment.transcription:
                            attachment.transcription = str(exc)
                step.status = CaseStepStatus.error
                step.error_message = step.error_message or str(exc)
                if step.case is not None:
                    step.case.phase = CasePhase.error
                await db.commit()
        except Exception:
            logger.exception("Failed to mark request %s as error", request.step_id)

    async def _process_request(self, request: Request) -> None:
        async with self._db() as db:
            result = await db.execute(
                select(CaseStep).where(CaseStep.id == UUID(request.step_id)).options(selectinload(CaseStep.attachments))
            )
            step = result.scalar_one_or_none()
            if step is None:
                logger.error("Step %s not found", request.step_id)
                return

            case_result = await db.execute(select(Case).where(Case.id == UUID(request.case_id)))
            case = case_result.scalar_one_or_none()
            if case is None:
                logger.error("Case %s not found", request.case_id)
                return

            if step.type == CaseStepType.triage:
                await self._process_step_triage(case, step, db)
            else:
                await self._process_step_regular(case, step, db)

    async def _process_step_regular(self, case: Case, step: CaseStep, db: AsyncSession) -> bool:
        step.status = CaseStepStatus.in_progress
        await db.commit()
        logger.info(
            "Processing step=%s case=%s attachments=%d",
            step.id,
            case.id,
            len(step.attachments),
        )
        for attachment in step.attachments:
            logger.info(
                "Attachment id=%s filename=%s kind=%s status=%s",
                attachment.id,
                attachment.original_filename,
                attachment.kind,
                attachment.status,
            )
            if attachment.status not in (AttachmentStatus.pending, AttachmentStatus.error):
                logger.warning("Skipping attachment %s — status is %s", attachment.id, attachment.status)
                continue
            attachment.status = AttachmentStatus.pending
            success = await self._process_file(attachment, db)
            if not success:
                step.status = CaseStepStatus.error
                step.error_message = f"Failed to process attachment: {attachment.original_filename}"
                case.phase = CasePhase.error
                await db.commit()
                return False

        step.status = CaseStepStatus.done
        await db.commit()
        return True

    async def _process_step_triage(self, case: Case, step: CaseStep, db: AsyncSession) -> None:
        if not await self._process_step_regular(case, step, db):
            return

        # Keep step as in_progress while waiting for the triage agent response
        step.status = CaseStepStatus.in_progress
        await db.commit()

        transcriptions = [
            AttachmentTranscriptions(
                name=a.original_filename,
                content=a.transcription,
                summary=a.summary,
                ner=a.ner,
                confidence=a.confidence,
            )
            for a in step.attachments
            if a.transcription
        ]
        triage_payload = AgentRequestPayloadTriage(
            case_id=str(case.id),
            patient=case.patient_info,
            description=case.chief_complaint,
            attachments_transcriptions=transcriptions or None,
        )
        triage_request = AgentRequest(id=str(case.id), payload=triage_payload)

        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                raw = await self._nats.request(
                    "triage.request",
                    triage_request.model_dump_json().encode(),
                    timeout=REQUEST_TIMEOUT,
                )
                result = AgentResponse.model_validate_json(raw.data)
                if not result.ok:
                    raise ValueError(result.error or "triage worker returned error")
                if not isinstance(result.payload, AgentResponsePayloadTriage):
                    raise ValueError("unexpected payload type from triage worker")
                payload = result.payload
                step.meta = payload.model_dump(mode="json")
                if payload.valid:
                    step.status = CaseStepStatus.done
                    case.esi_level = payload.esi_level.value if payload.esi_level else None
                    case.phase = CasePhase.pending_care
                else:
                    step.status = CaseStepStatus.error
                    step.error_message = payload.analysis
                    case.phase = CasePhase.triage
                await db.commit()
                return
            except Exception as exc:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    step.status = CaseStepStatus.error
                    step.error_message = str(exc)
                    case.phase = CasePhase.error
                    await db.commit()
                    logger.error("Triage request failed after %d retries: %s", MAX_RETRIES, exc)
                    return
                await asyncio.sleep(2**retry_count)

    async def _process_file(self, attachment: Attachment, db: AsyncSession) -> bool:
        subject = "attachment.audio" if attachment.kind == AttachmentKind.audio else "attachment.document"
        file_payload = AgentRequestPayloadFile(
            attachment_id=str(attachment.id),
            filename=attachment.original_filename,
            mime_type=attachment.mime_type,
            path=attachment.storage_path,
        )
        agent_request = AgentRequest(id=str(attachment.id), payload=file_payload)
        logger.info("Sending attachment id=%s to subject=%s", attachment.id, subject)

        attachment.status = AttachmentStatus.processing
        await db.commit()

        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                response = await self._nats.request(
                    subject,
                    agent_request.model_dump_json().encode(),
                    timeout=REQUEST_TIMEOUT,
                )
                result = AgentResponse.model_validate_json(response.data)
                if not result.ok:
                    raise ValueError(result.error or "worker returned error")
                if isinstance(result.payload, AttachmentTranscriptions):
                    attachment.transcription = result.payload.content
                    attachment.summary = result.payload.summary
                    attachment.ner = result.payload.ner
                    attachment.sbar = result.payload.sbar
                    attachment.confidence = result.payload.confidence
                else:
                    attachment.transcription = None
                attachment.status = AttachmentStatus.done
                await db.commit()
                return True
            except Exception as exc:
                retry_count += 1
                logger.warning(
                    "Attachment request failed (attempt %d/%d) id=%s subject=%s: %s",
                    retry_count,
                    MAX_RETRIES,
                    attachment.id,
                    subject,
                    exc,
                )
                if retry_count >= MAX_RETRIES:
                    attachment.status = AttachmentStatus.error
                    attachment.transcription = str(exc)
                    await db.commit()
                    return False
                await asyncio.sleep(2**retry_count)
        return False
