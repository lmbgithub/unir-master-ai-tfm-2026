import enum


class CasePhase(str, enum.Enum):
    triage = "triage"
    triage_validation = "triage_validation"
    pending_care = "pending_care"
    in_care = "in_care"
    closed_success = "closed_success"
    closed_death = "closed_death"
    closed_transfer = "closed_transfer"
    error = "error"


class CaseStepType(str, enum.Enum):
    triage = "triage"
    handoff = "handoff"
    regular = "regular"


class CaseStepStatus(str, enum.Enum):
    created = "created"
    pending = "pending"
    in_progress = "in_progress"
    pending_approval = "pending_approval"
    done = "done"
    error = "error"


class AttachmentKind(str, enum.Enum):
    image = "image"
    pdf = "pdf"
    audio = "audio"


class AttachmentStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    error = "error"
    cancelled = "cancelled"
