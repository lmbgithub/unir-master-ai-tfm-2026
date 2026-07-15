import { apiFetch } from "@/lib/api";
import type { CaseStep, CreateStepPayload, StepStatus } from "@/types/api";

class CaseLinkService {
  async create(caseId: string, data: CreateStepPayload): Promise<CaseStep> {
    return apiFetch<CaseStep>(`/cases/${caseId}/chain`, {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async get(caseId: string, stepId: string): Promise<CaseStep> {
    return apiFetch<CaseStep>(`/cases/${caseId}/chain/${stepId}`);
  }

  async updateStatus(caseId: string, stepId: string, status: StepStatus): Promise<CaseStep> {
    return apiFetch<CaseStep>(`/cases/${caseId}/chain/${stepId}/status`, {
      method: "PATCH",
      body: JSON.stringify({ status }),
    });
  }
}

export const caseLinkService = new CaseLinkService();
