"use client";

import { useEffect, useRef, useState } from "react";
import { caseService } from "@/services/CaseService";
import { CaseTopBar } from "./case-top-bar";
import { Timeline } from "./timeline";
import { AttachmentSidebar } from "./attachment-sidebar";
import { EditTriageDialog } from "@/components/cases/new-case-dialog";
import type { Case, CaseStep } from "@/types/api";

const CLOSED_PHASES = new Set(["closed_success", "closed_Dead", "closed_transfer"]);

function shouldPoll(c: Case): boolean {
  if (CLOSED_PHASES.has(c.phase)) return false;

  const steps = c.case_steps;
  if (steps.length === 0) return false;
  const last = [...steps].sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()).at(-1)!;
  return last.status !== "done" && last.status !== "error";
}

export function CaseDetailClient({ initial }: { initial: Case }) {
  const [caseData, setCaseData] = useState<Case>(initial);
  const [editTriageOpen, setEditTriageOpen] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  function startPolling(id: string) {
    if (timerRef.current !== null) return;
    timerRef.current = setInterval(async () => {
      try {
        const fresh = await caseService.get(id);
        setCaseData(fresh);
        if (!shouldPoll(fresh)) {
          clearInterval(timerRef.current!);
          timerRef.current = null;
        }
      } catch {
        // silently ignore transient errors
      }
    }, 5000);
  }

  useEffect(() => {
    const s = shouldPoll(caseData);
    timerRef.current = null;
    if (s) startPolling(caseData.id);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [caseData]);

  const triageStep = caseData.case_steps.find((s) => s.type === "triage") as CaseStep | undefined;
  const hasTriageError = triageStep?.status === "error";

  async function handleRetry(step: CaseStep) {
    try {
      const updated = await caseService.retryStep(caseData.id, step.id);
      applyUpdate(updated);
    } catch {
      // silently ignore — polling will reflect the new state
    }
  }

  function applyUpdate(updated: Case) {
    setCaseData(updated);
    if (shouldPoll(updated)) startPolling(updated.id);
  }

  return (
    <div>
      <CaseTopBar case={caseData} onNewStep={(updated) => setCaseData(updated)} />
      <Timeline
        steps={caseData.case_steps}
        patientInfo={caseData.patient_info}
        chiefComplaint={caseData.chief_complaint}
        onEditTriage={hasTriageError ? () => setEditTriageOpen(true) : undefined}
        onRetry={handleRetry}
      />
      <AttachmentSidebar />
      {hasTriageError && triageStep && (
        <EditTriageDialog
          open={editTriageOpen}
          onOpenChange={setEditTriageOpen}
          case_={caseData}
          triageStep={triageStep}
          onSuccess={applyUpdate}
        />
      )}
    </div>
  );
}
