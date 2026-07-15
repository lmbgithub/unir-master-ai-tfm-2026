import { apiFetch } from "@/lib/api";
import type {
  Attachment,
  AttachmentKind,
  Case,
  CaseStatus,
  CaseStep,
  CreateCasePayload,
  UpdateCasePayload,
  DashboardStats,
  PaginatedCases,
} from "@/types/api";

function baseUrl() {
  return typeof window !== "undefined" ? "/api" : (process.env.API_URL ?? "http://localhost:8000");
}

class CaseService {
  async getAll(params: {
    page: number;
    page_size: number;
    search?: string;
    status?: CaseStatus;
  }): Promise<PaginatedCases> {
    const query = new URLSearchParams({
      page: String(params.page),
      page_size: String(params.page_size),
    });
    if (params.search) query.set("search", params.search);
    if (params.status) query.set("status", params.status);
    return apiFetch<PaginatedCases>(`/cases?${query}`);
  }

  async get(id: string): Promise<Case> {
    return apiFetch<Case>(`/cases/${id}`);
  }

  async create(data: CreateCasePayload): Promise<Case> {
    const formData = new FormData();
    formData.append("patient_info", JSON.stringify(data.patient_info));
    formData.append("chief_complaint", data.chief_complaint);
    const response = await fetch(`${baseUrl()}/cases`, {
      method: "POST",
      credentials: "include",
      body: formData,
    });
    if (!response.ok) {
      const text = await response.text().catch(() => response.statusText);
      throw new Error(text);
    }
    return response.json() as Promise<Case>;
  }

  async updateStatus(id: string, status: CaseStatus): Promise<Case> {
    return apiFetch<Case>(`/cases/${id}/phase`, {
      method: "PATCH",
      body: JSON.stringify({ phase: status }),
    });
  }

  async delete(id: string): Promise<void> {
    return apiFetch<void>(`/cases/${id}`, { method: "DELETE" });
  }

  async getStats(): Promise<DashboardStats> {
    return apiFetch<DashboardStats>("/cases/stats");
  }

  async submitStep(caseId: string, stepId: string): Promise<void> {
    const response = await fetch(`${baseUrl()}/cases/${caseId}/steps/${stepId}/submit`, {
      method: "POST",
      credentials: "include",
    });
    if (!response.ok) {
      const text = await response.text().catch(() => response.statusText);
      throw new Error(text);
    }
  }

  async updateCase(id: string, data: UpdateCasePayload): Promise<Case> {
    return apiFetch<Case>(`/cases/${id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    });
  }

  async deleteAttachment(caseId: string, stepId: string, attachmentId: string): Promise<void> {
    return apiFetch<void>(`/cases/${caseId}/steps/${stepId}/attachments/${attachmentId}`, {
      method: "DELETE",
    });
  }

  async retryStep(caseId: string, stepId: string): Promise<Case> {
    await apiFetch<void>(`/cases/${caseId}/steps/${stepId}/retry`, { method: "POST" });
    return this.get(caseId);
  }

  async createStep(caseId: string, data: { type: "handoff" | "regular"; description?: string }): Promise<CaseStep> {
    const response = await fetch(`${baseUrl()}/cases/${caseId}/steps`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!response.ok) {
      const text = await response.text().catch(() => response.statusText);
      throw new Error(text);
    }
    return response.json() as Promise<CaseStep>;
  }

  async uploadAttachment(caseId: string, stepId: string, file: File, kind: AttachmentKind): Promise<Attachment> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("kind", kind);
    const response = await fetch(`${baseUrl()}/cases/${caseId}/steps/${stepId}/attachments`, {
      method: "POST",
      credentials: "include",
      body: formData,
    });
    if (!response.ok) {
      const text = await response.text().catch(() => response.statusText);
      throw new Error(text);
    }
    return response.json() as Promise<Attachment>;
  }
}

export const caseService = new CaseService();
