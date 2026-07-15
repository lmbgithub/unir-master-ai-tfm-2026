import { ApiError } from "@/lib/api";
import type { Attachment, AttachmentKind } from "@/types/api";

function baseUrl() {
  if (typeof window !== "undefined") return "/api";
  return process.env.API_URL ?? "http://localhost:8000";
}

async function authedFetch(path: string, options: RequestInit = {}) {
  const response = await fetch(`${baseUrl()}${path}`, {
    ...options,
    credentials: "include",
  });
  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText);
    throw new ApiError(response.status, text);
  }
  return response;
}

class AttachmentService {
  async create(stepId: string, file: File, kind: AttachmentKind): Promise<Attachment> {
    const form = new FormData();
    form.append("file", file);
    form.append("kind", kind);
    const response = await authedFetch(`/attachments?step_id=${stepId}`, {
      method: "POST",
      body: form,
    });
    return response.json() as Promise<Attachment>;
  }

  async get(id: string): Promise<Attachment> {
    const response = await authedFetch(`/attachments/${id}`);
    return response.json() as Promise<Attachment>;
  }

  async download(id: string): Promise<Blob> {
    const response = await authedFetch(`/attachments/${id}/download`);
    return response.blob();
  }
}

export const attachmentService = new AttachmentService();
