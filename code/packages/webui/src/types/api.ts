export type CaseStatus =
  | "triage"
  | "triage_validation"
  | "pending_care"
  | "in_care"
  | "closed_success"
  | "closed_Dead"
  | "closed_transfer";

export type AttachmentStatus = "pending" | "processing" | "done" | "error" | "cancelled";
export type AttachmentKind = "image" | "pdf" | "audio";
export type StepStatus = "pending" | "in_progress" | "done" | "error";
export type StepType = "triage" | "handoff" | "regular";

export interface PatientInfo {
  name: string;
  gender: "male" | "female";
  date_of_birth: string;
  id_number: string;
  blood_type: "A" | "B" | "O" | "AB";
  blood_rh: boolean;
  blood_pressure_systolic: number;
  blood_pressure_diastolic: number;
  weight: number;
  height: number;
  pulse: number;
  allergies: string[];
  chronic_conditions: string[];
}

export interface Attachment {
  id: string;
  case_step_id: string;
  original_filename: string;
  mime_type: string;
  storage_path: string;
  kind: AttachmentKind;
  status: AttachmentStatus;
  transcription: string | null;
  summary: string | null;
  ner: Record<string, string> | null;
  sbar: Record<string, string> | null;
  confidence: number | null;
  created_at: string;
}

export interface TriageMeta {
  valid: boolean;
  esi_level: number | null;
  analysis: string;
  missing_fields: string[] | null;
}

export interface CaseStep {
  id: string;
  case_id: string;
  type: StepType;
  status: StepStatus;
  assigned_to: string | null;
  description: string | null;
  error_message: string | null;
  meta: TriageMeta | null;
  created_at: string;
  started_at: string | null;
  updated_at: string;
  attachments: Attachment[];
}

export interface Case {
  id: string;
  patient_info: PatientInfo;
  chief_complaint: string;
  esi_level: number | null;
  phase: CaseStatus;
  created_at: string;
  updated_at: string;
  case_steps: CaseStep[];
}

export interface PaginatedCases {
  items: Case[];
  total: number;
  page: number;
  page_size: number;
}

export interface DashboardStats {
  total: number;
  open: number;
  completed: number;
  error: number;
}

export interface UserResponse {
  user_id: string;
  email: string;
  role: string;
}

export interface CreateCasePayload {
  patient_info: PatientInfo;
  chief_complaint: string;
}

export interface UpdateCasePayload {
  patient_info?: PatientInfo;
  chief_complaint?: string;
}

export interface CreateStepPayload {
  type: StepType;
  description?: string;
  assigned_to?: string;
}
