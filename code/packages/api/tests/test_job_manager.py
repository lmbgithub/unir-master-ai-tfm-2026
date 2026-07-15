import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from urgenurse.api.enums import (
    AttachmentKind,
    AttachmentStatus,
    CasePhase,
    CaseStepStatus,
    CaseStepType,
)
from urgenurse.agents.agent.requests import (
    AgentResponse,
    AgentResponsePayloadTriage,
    AttachmentTranscriptions,
    ESILevels,
)
from urgenurse.api.models.attachment import Attachment
from urgenurse.api.models.case import Case
from urgenurse.api.models.case_step import CaseStep
from urgenurse.api.utils.job_manager import JobManager, MAX_RETRIES, Request


def _make_nats() -> MagicMock:
    nc = MagicMock()
    js = AsyncMock()
    nc.jetstream.return_value = js
    nc.request = AsyncMock()
    return nc


def _make_db_factory(session: AsyncMock) -> MagicMock:
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return factory


_MIME = {
    AttachmentKind.image: "image/jpeg",
    AttachmentKind.pdf: "application/pdf",
    AttachmentKind.audio: "audio/wav",
}


def _attachment(kind: AttachmentKind, status: AttachmentStatus = AttachmentStatus.pending) -> Attachment:
    a = Attachment()
    a.id = uuid.uuid4()
    a.kind = kind
    a.status = status
    a.storage_path = "/tmp/file"
    a.original_filename = "test.file"
    a.mime_type = _MIME[kind]
    a.transcription = None
    return a


def _step(step_type: CaseStepType, attachments: list[Attachment] | None = None) -> CaseStep:
    s = CaseStep()
    s.id = uuid.uuid4()
    s.type = step_type
    s.status = CaseStepStatus.pending
    s.attachments = attachments or []
    return s


def _case() -> Case:
    c = Case()
    c.id = uuid.uuid4()
    c.chief_complaint = "chest pain"
    c.patient_info = {}
    c.phase = CasePhase.triage
    return c


@pytest.fixture
def nats_mock():
    return _make_nats()


@pytest.fixture
def job_manager(nats_mock):
    db_factory = MagicMock()
    return JobManager(nats=nats_mock, db=db_factory)


# ── queue_request ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_queue_request_publishes(job_manager, nats_mock):
    js = nats_mock.jetstream.return_value
    await job_manager.queue_request("case-1", "step-1")
    js.publish.assert_awaited_once()
    subject, payload = js.publish.call_args.args
    assert subject == "request"
    req = Request.model_validate_json(payload)
    assert req.case_id == "case-1"
    assert req.step_id == "step-1"


# ── _process_file ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_process_file_doc_success(job_manager, nats_mock):
    attachment = _attachment(AttachmentKind.image)
    db = AsyncMock()
    response_msg = AgentResponse(
        id=str(attachment.id),
        ok=True,
        payload=AttachmentTranscriptions(name="test.file", content="hello"),
    )
    nats_mock.request.return_value = MagicMock(data=response_msg.model_dump_json().encode())

    result = await job_manager._process_file(attachment, db)

    assert result is True
    assert attachment.status == AttachmentStatus.done
    assert attachment.transcription == "hello"
    nats_mock.request.assert_awaited_once()
    subject = nats_mock.request.call_args.args[0]
    assert subject == "attachment.document"


@pytest.mark.asyncio
async def test_process_file_audio_success(job_manager, nats_mock):
    attachment = _attachment(AttachmentKind.audio)
    db = AsyncMock()
    response_msg = AgentResponse(
        id=str(attachment.id),
        ok=True,
        payload=AttachmentTranscriptions(name="test.file", content="audio text"),
    )
    nats_mock.request.return_value = MagicMock(data=response_msg.model_dump_json().encode())

    result = await job_manager._process_file(attachment, db)

    assert result is True
    subject = nats_mock.request.call_args.args[0]
    assert subject == "attachment.audio"


@pytest.mark.asyncio
async def test_process_file_retries_on_error(job_manager, nats_mock):
    attachment = _attachment(AttachmentKind.pdf)
    db = AsyncMock()
    success_msg = AgentResponse(
        id=str(attachment.id),
        ok=True,
        payload=AttachmentTranscriptions(name="test.file", content="text"),
    )
    nats_mock.request.side_effect = [
        Exception("timeout"),
        Exception("timeout"),
        MagicMock(data=success_msg.model_dump_json().encode()),
    ]

    with patch("urgenurse.api.utils.job_manager.asyncio.sleep", new_callable=AsyncMock):
        result = await job_manager._process_file(attachment, db)

    assert result is True
    assert nats_mock.request.await_count == 3


@pytest.mark.asyncio
async def test_process_file_max_retries_sets_error(job_manager, nats_mock):
    attachment = _attachment(AttachmentKind.image)
    db = AsyncMock()
    nats_mock.request.side_effect = Exception("always fails")

    with patch("urgenurse.api.utils.job_manager.asyncio.sleep", new_callable=AsyncMock):
        result = await job_manager._process_file(attachment, db)

    assert result is False
    assert attachment.status == AttachmentStatus.error
    assert nats_mock.request.await_count == MAX_RETRIES


# ── _process_step_regular ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_process_step_regular_no_attachments_sets_done(job_manager, nats_mock):
    step = _step(CaseStepType.regular, [])
    case = _case()
    db = AsyncMock()

    result = await job_manager._process_step_regular(case, step, db)

    assert result is True
    assert step.status == CaseStepStatus.done
    nats_mock.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_step_regular_handoff_with_audio_sets_done(job_manager, nats_mock):
    attachment = _attachment(AttachmentKind.audio)
    step = _step(CaseStepType.handoff, [attachment])
    case = _case()
    db = AsyncMock()

    success_msg = AgentResponse(
        id=str(attachment.id),
        ok=True,
        payload=AttachmentTranscriptions(name="voice.wav", content="handoff note"),
    )
    nats_mock.request.return_value = MagicMock(data=success_msg.model_dump_json().encode())

    result = await job_manager._process_step_regular(case, step, db)

    assert result is True
    assert step.status == CaseStepStatus.done
    subject = nats_mock.request.call_args.args[0]
    assert subject == "attachment.audio"
    assert attachment.transcription == "handoff note"


@pytest.mark.asyncio
async def test_process_step_regular_attachment_failure_sets_error(job_manager, nats_mock):
    attachment = _attachment(AttachmentKind.image)
    step = _step(CaseStepType.regular, [attachment])
    case = _case()
    db = AsyncMock()
    nats_mock.request.side_effect = Exception("worker unavailable")

    with patch("urgenurse.api.utils.job_manager.asyncio.sleep", new_callable=AsyncMock):
        result = await job_manager._process_step_regular(case, step, db)

    assert result is False
    assert step.status == CaseStepStatus.error
    assert case.phase == CasePhase.error
    assert attachment.original_filename in (step.error_message or "")


@pytest.mark.asyncio
async def test_process_step_regular_skips_non_pending_attachment(job_manager, nats_mock):
    already_done = _attachment(AttachmentKind.image, status=AttachmentStatus.done)
    already_done.transcription = "existing"
    step = _step(CaseStepType.regular, [already_done])
    case = _case()
    db = AsyncMock()

    result = await job_manager._process_step_regular(case, step, db)

    assert result is True
    assert step.status == CaseStepStatus.done
    nats_mock.request.assert_not_awaited()


# ── _process_triage ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_process_triage_all_done(job_manager, nats_mock):
    attachment = _attachment(AttachmentKind.image)
    step = _step(CaseStepType.triage, [attachment])
    case = _case()
    db = AsyncMock()

    success_msg = AgentResponse(
        id=str(attachment.id),
        ok=True,
        payload=AttachmentTranscriptions(name="test.file", content="text"),
    )
    triage_ok = AgentResponse(
        id=str(case.id),
        ok=True,
        payload=AgentResponsePayloadTriage(
            valid=True,
            missing_fields=[],
            esi_level=ESILevels.LEVEL2,
            analysis="stub",
        ),
    )
    # first request = attachment, second = triage
    nats_mock.request.side_effect = [
        MagicMock(data=success_msg.model_dump_json().encode()),
        MagicMock(data=triage_ok.model_dump_json().encode()),
    ]

    await job_manager._process_step_triage(case, step, db)

    assert step.status == CaseStepStatus.done
    assert nats_mock.request.await_count == 2
    triage_subject = nats_mock.request.call_args_list[1].args[0]
    assert triage_subject == "triage.request"


@pytest.mark.asyncio
async def test_process_triage_valid_result_sets_esi_and_phase(job_manager, nats_mock):
    attachment = _attachment(AttachmentKind.image)
    step = _step(CaseStepType.triage, [attachment])
    case = _case()
    db = AsyncMock()

    success_msg = AgentResponse(
        id=str(attachment.id),
        ok=True,
        payload=AttachmentTranscriptions(name="test.file", content="text"),
    )
    triage_ok = AgentResponse(
        id=str(case.id),
        ok=True,
        payload=AgentResponsePayloadTriage(
            valid=True,
            missing_fields=[],
            esi_level=ESILevels.LEVEL3,
            analysis="patient looks stable",
        ),
    )
    nats_mock.request.side_effect = [
        MagicMock(data=success_msg.model_dump_json().encode()),
        MagicMock(data=triage_ok.model_dump_json().encode()),
    ]

    await job_manager._process_step_triage(case, step, db)

    assert step.status == CaseStepStatus.done
    assert case.esi_level == ESILevels.LEVEL3.value
    assert case.phase == CasePhase.pending_care
    assert step.meta is not None
    assert step.meta["valid"] is True
    assert step.meta["analysis"] == "patient looks stable"


@pytest.mark.asyncio
async def test_process_triage_invalid_result_sets_step_error(job_manager, nats_mock):
    attachment = _attachment(AttachmentKind.image)
    step = _step(CaseStepType.triage, [attachment])
    case = _case()
    db = AsyncMock()

    success_msg = AgentResponse(
        id=str(attachment.id),
        ok=True,
        payload=AttachmentTranscriptions(name="test.file", content="text"),
    )
    triage_invalid = AgentResponse(
        id=str(case.id),
        ok=True,
        payload=AgentResponsePayloadTriage(
            valid=False,
            missing_fields=["blood_pressure_systolic", "pulse"],
            esi_level=ESILevels.LEVEL5,
            analysis="missing vital signs",
        ),
    )
    nats_mock.request.side_effect = [
        MagicMock(data=success_msg.model_dump_json().encode()),
        MagicMock(data=triage_invalid.model_dump_json().encode()),
    ]

    await job_manager._process_step_triage(case, step, db)

    assert step.status == CaseStepStatus.error
    assert step.error_message == "missing vital signs"
    assert case.phase == CasePhase.triage
    assert step.meta is not None
    assert step.meta["valid"] is False
    assert step.meta["missing_fields"] == ["blood_pressure_systolic", "pulse"]


@pytest.mark.asyncio
async def test_process_triage_attachment_error_stops(job_manager, nats_mock):
    attachment = _attachment(AttachmentKind.image)
    step = _step(CaseStepType.triage, [attachment])
    case = _case()
    db = AsyncMock()

    nats_mock.request.side_effect = Exception("worker down")

    with patch("urgenurse.api.utils.job_manager.asyncio.sleep", new_callable=AsyncMock):
        await job_manager._process_step_triage(case, step, db)

    assert step.status == CaseStepStatus.error
    assert case.phase == CasePhase.error
    # triage.request must NOT have been called
    for call in nats_mock.request.call_args_list:
        assert call.args[0] != "triage.request"
