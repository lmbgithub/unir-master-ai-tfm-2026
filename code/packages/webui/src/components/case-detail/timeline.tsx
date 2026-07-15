"use client";

import { useEffect, useRef, useState } from "react";
import type { CaseStep, PatientInfo, StepStatus } from "@/types/api";
import { TimelineBlock } from "./timeline-block";

function StatusDot({ status }: { status: StepStatus }) {
  if (status === "in_progress") {
    return (
      <div className="absolute -left-[calc(2rem)] top-4 w-4 h-4 bg-foreground rounded-full flex items-center justify-center">
        <span className="w-2 h-2 rounded-full bg-background animate-ping" />
      </div>
    );
  }
  if (status === "done") {
    return (
      <div className="absolute -left-[calc(2rem)] top-4 w-4 h-4 bg-foreground rounded-full flex items-center justify-center">
        <svg
          className="w-2.5 h-2.5 text-background"
          viewBox="0 0 12 12"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="2,6 5,9 10,3" />
        </svg>
      </div>
    );
  }
  if (status === "error") {
    return (
      <div className="absolute -left-[calc(2rem)] top-4 w-4 h-4 bg-destructive rounded-full flex items-center justify-center">
        <svg
          className="w-2.5 h-2.5 text-destructive-foreground"
          viewBox="0 0 12 12"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
        >
          <line x1="3" y1="3" x2="9" y2="9" />
          <line x1="9" y1="3" x2="3" y2="9" />
        </svg>
      </div>
    );
  }
  // pending
  return <div className="absolute -left-[calc(2rem)] top-4 w-4 h-4 bg-muted-foreground rounded-full" />;
}

interface TimelineProps {
  steps: CaseStep[];
  patientInfo: PatientInfo;
  chiefComplaint: string;
  onEditTriage?: () => void;
  onRetry?: (step: CaseStep) => void;
}

export function Timeline({ steps, patientInfo, chiefComplaint, onEditTriage, onRetry }: TimelineProps) {
  const sorted = [...steps].sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());

  const lastId = sorted.at(-1)?.id ?? null;
  const [expandedId, setExpandedId] = useState<string | null>(lastId);
  const prevCountRef = useRef(sorted.length);
  const stepRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  useEffect(() => {
    if (sorted.length > prevCountRef.current && lastId) {
      setExpandedId(lastId);
      const el = stepRefs.current.get(lastId);
      el?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    prevCountRef.current = sorted.length;
  }, [sorted.length, lastId]);

  if (sorted.length === 0) {
    return <p className="text-sm text-muted-foreground px-6 py-4">No steps yet.</p>;
  }

  return (
    <div className="border-l border-border ml-4 px-6 py-4">
      <div className="space-y-6">
        {sorted.map((step) => (
          <div
            key={step.id}
            className="relative"
            ref={(el) => {
              if (el) stepRefs.current.set(step.id, el);
              else stepRefs.current.delete(step.id);
            }}
          >
            <StatusDot status={step.status} />
            <TimelineBlock
              step={step}
              open={expandedId === step.id}
              onToggle={() => setExpandedId((prev) => (prev === step.id ? null : step.id))}
              patientInfo={step.type === "triage" ? patientInfo : undefined}
              chiefComplaint={step.type === "triage" ? chiefComplaint : undefined}
              onEditTriage={step.type === "triage" && step.status === "error" ? onEditTriage : undefined}
              onRetry={step.status === "error" && onRetry ? () => onRetry(step) : undefined}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
