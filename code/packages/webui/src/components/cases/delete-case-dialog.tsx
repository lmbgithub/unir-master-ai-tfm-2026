"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { caseService } from "@/services/CaseService";

interface DeleteCaseDialogProps {
  id: string;
  onClose: () => void;
}

export function DeleteCaseDialog({ id, onClose }: DeleteCaseDialogProps) {
  const router = useRouter();
  const [loading, setLoading] = useState(false);

  async function handleConfirm() {
    setLoading(true);
    try {
      await caseService.delete(id);
      onClose();
      router.refresh();
    } catch {
      alert("Failed to delete case. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} aria-hidden />
      <div className="relative z-10 w-full max-w-md rounded-lg border bg-card p-6 shadow-lg">
        <h2 className="text-lg font-semibold">Delete Case</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          This action cannot be undone. Case #{id.slice(0, 8)} and all its data will be permanently deleted.
        </p>
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
            onClick={handleConfirm}
            disabled={loading}
            className="rounded-md bg-destructive px-4 py-2 text-sm font-medium text-destructive-foreground hover:bg-destructive/90 disabled:opacity-50"
          >
            {loading ? "Deleting…" : "Delete"}
          </button>
        </div>
      </div>
    </div>
  );
}
