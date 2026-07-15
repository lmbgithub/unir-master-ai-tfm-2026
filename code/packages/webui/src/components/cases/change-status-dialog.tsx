"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { caseService } from "@/services/CaseService";
import type { CaseStatus } from "@/types/api";

const STATUS_OPTIONS: { value: CaseStatus; label: string }[] = [
  { value: "triage", label: "Triage" },
  { value: "triage_validation", label: "Triage Validation" },
  { value: "pending_care", label: "Waiting attention" },
  { value: "in_care", label: "In attention" },
  { value: "closed_success", label: "Discharged" },
  { value: "closed_Dead", label: "Dead" },
  { value: "closed_transfer", label: "Transfered" },
];

interface ChangeStatusDialogProps {
  id: string;
  currentStatus: CaseStatus;
  onClose: () => void;
}

export function ChangeStatusDialog({ id, currentStatus, onClose }: ChangeStatusDialogProps) {
  const router = useRouter();
  const [selected, setSelected] = useState<CaseStatus>(currentStatus);
  const [loading, setLoading] = useState(false);

  async function handleSave() {
    if (selected === currentStatus) {
      onClose();
      return;
    }
    setLoading(true);
    try {
      await caseService.updateStatus(id, selected);
      onClose();
      router.refresh();
    } catch {
      alert("Failed to update status. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} aria-hidden />
      <div className="relative z-10 w-full max-w-md rounded-lg border bg-card p-6 shadow-lg">
        <h2 className="text-lg font-semibold">Change Case Status</h2>
        <div className="mt-4">
          <select
            value={selected}
            onChange={(e) => setSelected(e.target.value as CaseStatus)}
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
          >
            {STATUS_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        <div className="mt-6 flex justify-end gap-3">
          <button
            type="button"
            onClick={onClose}
            disabled={loading}
            className="rounded-md border border-border px-4 py-2 text-sm font-medium hover:bg-muted disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSave}
            disabled={loading}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {loading ? "Saving…" : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}
