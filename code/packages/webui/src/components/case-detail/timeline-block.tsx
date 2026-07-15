"use client";

import { ChevronDown, AlertCircle, MoreHorizontal, Pencil, RotateCcw } from "lucide-react";
import type { CaseStep, PatientInfo } from "@/types/api";
import { StepStatusBadge } from "./step-status-badge";
import { AttachmentList } from "./attachment-list";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

function formatDate(iso: string) {
  const d = new Date(iso);
  const day = String(d.getDate()).padStart(2, "0");
  const month = d.toLocaleString("en", { month: "short" });
  const year = d.getFullYear();
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${day} ${month} ${year} · ${hh}:${mm}`;
}

function formatDob(iso: string) {
  const [y, m, d] = iso.split("-");
  const month = new Date(Number(y), Number(m) - 1).toLocaleString("en", { month: "short" });
  return `${d} ${month} ${y}`;
}

const typeLabel: Record<CaseStep["type"], string> = {
  triage: "TRIAGE",
  handoff: "HANDOFF",
  regular: "REGULAR",
};

interface InfoRowProps {
  label: string;
  value: string | number;
}

function InfoRow({ label, value }: InfoRowProps) {
  return (
    <tr className="border-b last:border-0">
      <td className="py-1.5 pr-4 text-xs font-medium text-muted-foreground whitespace-nowrap">{label}</td>
      <td className="py-1.5 text-xs">{value}</td>
    </tr>
  );
}

function PatientInfoTable({ info, chiefComplaint }: { info: PatientInfo; chiefComplaint?: string }) {
  const bloodType = `${info.blood_type} ${info.blood_rh ? "Rh+" : "Rh−"}`;
  const bp = `${info.blood_pressure_systolic} / ${info.blood_pressure_diastolic} mmHg`;
  const allergies = info.allergies.length ? info.allergies.join(", ") : "None";
  const conditions = info.chronic_conditions.length ? info.chronic_conditions.join(", ") : "None";

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">Patient</p>
          <table className="w-full">
            <tbody>
              <InfoRow label="Full Name" value={info.name} />
              <InfoRow label="ID / Document" value={info.id_number} />
              <InfoRow label="Date of Birth" value={formatDob(info.date_of_birth)} />
              <InfoRow label="Gender" value={info.gender.charAt(0).toUpperCase() + info.gender.slice(1)} />
              <InfoRow label="Allergies" value={allergies} />
              <InfoRow label="Chronic Conditions" value={conditions} />
            </tbody>
          </table>
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">Vitals</p>
          <table className="w-full">
            <tbody>
              <InfoRow label="Blood Type" value={bloodType} />
              <InfoRow label="Blood Pressure" value={bp} />
              <InfoRow label="Pulse" value={`${info.pulse} bpm`} />
              <InfoRow label="Weight" value={`${info.weight} kg`} />
              <InfoRow label="Height" value={`${info.height} cm`} />
            </tbody>
          </table>
        </div>
      </div>

      {chiefComplaint && (
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1">
            Main complaint / Symptoms
          </p>
          <p className="text-sm bg-muted/40 rounded-md px-3 py-2">{chiefComplaint}</p>
        </div>
      )}
    </div>
  );
}

const ESI_COLORS: Record<number, string> = {
  1: "bg-red-600 text-white",
  2: "bg-orange-500 text-white",
  3: "bg-yellow-400 text-black",
  4: "bg-green-500 text-white",
  5: "bg-blue-400 text-white",
};

const ESI_LABELS: Record<number, string> = {
  1: "ESI 1 — Immediate",
  2: "ESI 2 — Emergent",
  3: "ESI 3 — Urgent",
  4: "ESI 4 — Less Urgent",
  5: "ESI 5 — Non-Urgent",
};

interface TimelineBlockProps {
  step: CaseStep;
  open: boolean;
  onToggle: () => void;
  patientInfo?: PatientInfo;
  chiefComplaint?: string;
  onEditTriage?: () => void;
  onRetry?: () => void;
}

export function TimelineBlock({
  step,
  open,
  onToggle,
  patientInfo,
  chiefComplaint,
  onEditTriage,
  onRetry,
}: TimelineBlockProps) {
  const isError = step.status === "error";
  const triageMeta = step.type === "triage" ? step.meta : null;

  return (
    <div className="bg-white rounded-xl shadow-sm border">
      <div className="flex items-center gap-3 px-4 py-3">
        <button type="button" onClick={onToggle} className="flex items-center gap-3 flex-1 text-left">
          <span className="font-semibold text-sm tracking-wide">{typeLabel[step.type]}</span>
          <span className="text-xs text-muted-foreground">{formatDate(step.created_at)}</span>
          <StepStatusBadge status={step.status} />
          <ChevronDown
            className={`ml-auto w-4 h-4 text-muted-foreground transition-transform duration-200 ${
              open ? "rotate-180" : ""
            }`}
          />
        </button>
        {isError && (onEditTriage || onRetry) && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button size="sm" variant="outline" className="shrink-0 px-2">
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {onEditTriage && (
                <DropdownMenuItem onClick={onEditTriage}>
                  <Pencil className="mr-2 h-3.5 w-3.5" />
                  Edit
                </DropdownMenuItem>
              )}
              {onRetry && (
                <DropdownMenuItem onClick={onRetry}>
                  <RotateCcw className="mr-2 h-3.5 w-3.5" />
                  Retry
                </DropdownMenuItem>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>

      {isError && step.error_message && (
        <div className="mx-4 mb-3 space-y-2">
          <div className="flex gap-2 items-start rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <div className="space-y-1">
              <p>{step.error_message}</p>
              {step.meta?.missing_fields && step.meta.missing_fields.length > 0 && (
                <p className="text-xs font-medium">Missing fields: {step.meta.missing_fields.join(", ")}</p>
              )}
            </div>
          </div>
        </div>
      )}

      <div className={`overflow-hidden transition-all duration-200 ${open ? "max-h-[2000px]" : "max-h-0"}`}>
        <div className="px-4 pb-4 space-y-4">
          {step.type === "triage" ? (
            <div className="grid grid-cols-3 gap-4">
              <div className="col-span-2">
                <Tabs defaultValue="analysis">
                  <TabsList className="w-full justify-start gap-1 rounded-none border-b bg-transparent p-0 h-auto mb-3">
                    <TabsTrigger
                      value="analysis"
                      className="rounded-none border-b-2 border-transparent px-3 pb-2 pt-0 text-xs font-semibold uppercase tracking-wide text-muted-foreground data-[state=active]:border-foreground data-[state=active]:text-foreground data-[state=active]:shadow-none bg-transparent"
                    >
                      Analysis
                    </TabsTrigger>
                    <TabsTrigger
                      value="patient"
                      className="rounded-none border-b-2 border-transparent px-3 pb-2 pt-0 text-xs font-semibold uppercase tracking-wide text-muted-foreground data-[state=active]:border-foreground data-[state=active]:text-foreground data-[state=active]:shadow-none bg-transparent"
                    >
                      Patient
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="analysis" className="mt-0 space-y-3">
                    {triageMeta && step.status === "done" && (
                      <>
                        {triageMeta.esi_level && (
                          <div className="flex items-center gap-2">
                            <span
                              className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-bold ${ESI_COLORS[triageMeta.esi_level] ?? "bg-muted text-foreground"}`}
                            >
                              {ESI_LABELS[triageMeta.esi_level] ?? `ESI ${triageMeta.esi_level}`}
                            </span>
                          </div>
                        )}
                        <div>
                          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1">
                            Triage Analysis
                          </p>
                          <p className="text-sm bg-muted/40 rounded-md px-3 py-2">{triageMeta.analysis}</p>
                        </div>
                      </>
                    )}
                    {chiefComplaint && (
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1">
                          Main complaint / Symptoms
                        </p>
                        <p className="text-sm bg-muted/40 rounded-md px-3 py-2">{chiefComplaint}</p>
                      </div>
                    )}
                  </TabsContent>

                  <TabsContent value="patient" className="mt-0">
                    {patientInfo && <PatientInfoTable info={patientInfo} chiefComplaint={chiefComplaint} />}
                  </TabsContent>
                </Tabs>
              </div>

              <div className="col-span-1">
                <AttachmentList attachments={step.attachments} active={open} />
              </div>
            </div>
          ) : (
            <>
              {patientInfo && <PatientInfoTable info={patientInfo} chiefComplaint={chiefComplaint} />}

              {!patientInfo && (
                <div className="grid grid-cols-3 gap-4">
                  <div className="col-span-2 space-y-3">
                    {step.assigned_to && (
                      <p className="text-sm">
                        <span className="font-medium">Assigned to:</span> {step.assigned_to}
                      </p>
                    )}
                    {step.description && (
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1">
                          Description
                        </p>
                        <p className="text-sm bg-muted/40 rounded-md px-3 py-2">{step.description}</p>
                      </div>
                    )}
                    {step.type === "handoff" && step.attachments.filter((a) => a.summary).length > 0 && (
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">
                          Transcriptions
                        </p>
                        <div className="space-y-2">
                          {step.attachments
                            .filter((a) => a.transcription)
                            .map((a) => (
                              <div key={a.id} className="bg-muted/40 rounded-md px-3 py-2">
                                <p className="text-xs font-medium text-muted-foreground mb-0.5 truncate">
                                  {a.original_filename}
                                </p>
                                <p className="text-sm">{a.transcription}</p>
                              </div>
                            ))}
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="col-span-1">
                    <AttachmentList attachments={step.attachments} active={open} />
                  </div>
                </div>
              )}

              {patientInfo && step.attachments.length > 0 && (
                <div className="border-t pt-3">
                  <AttachmentList attachments={step.attachments} active={open} />
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
