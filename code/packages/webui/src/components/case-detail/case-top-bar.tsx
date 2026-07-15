import { StatusBadge } from "@/components/cases/status-badge";
import { ChangeStatusButton } from "./change-status-button";
import { NewStepButton } from "./new-step-button";
import type { Case } from "@/types/api";

const ESI_CLASSES: Record<number, string> = {
  1: "bg-red-100 text-red-700",
  2: "bg-orange-100 text-orange-700",
  3: "bg-yellow-100 text-yellow-700",
  4: "bg-green-100 text-green-700",
  5: "bg-blue-100 text-blue-700",
};

const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function formatCreated(iso: string): string {
  const d = new Date(iso);
  const day = String(d.getDate()).padStart(2, "0");
  const month = MONTH_NAMES[d.getMonth()];
  const year = d.getFullYear();
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${day} ${month} ${year} · ${hh}:${mm}`;
}

interface CaseTopBarProps {
  case: Case;
  onNewStep?: (updated: Case) => void;
}

export function CaseTopBar({ case: c, onNewStep }: CaseTopBarProps) {
  const shortId = c.id.slice(0, 8);
  const created = formatCreated(c.created_at);

  const isButtonDisabled = c.phase === "closed_success" || c.phase === "closed_Dead" || c.phase === "closed_transfer";

  return (
    <div className="bg-white border-b px-6 py-2 sticky top-0 z-10">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-row items-center gap-3 text-sm">
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <Chip label="Case" value={`#${shortId}`} />
            <Chip label="Patient" value={c.patient_info.name} />
            <div className="flex items-center gap-1.5">
              <span className="text-muted-foreground">Status</span>
              <StatusBadge status={c.phase} />
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-muted-foreground">ESI</span>
              {c.esi_level !== null ? (
                <span
                  className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${ESI_CLASSES[c.esi_level] ?? ""}`}
                >
                  {c.esi_level}
                </span>
              ) : (
                <span className="text-muted-foreground">—</span>
              )}
            </div>
          </div>
          <div>
            <Chip label="Created" value={created} />
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <NewStepButton caseId={c.id} onSuccess={onNewStep} disabled={isButtonDisabled} />
          <ChangeStatusButton case={c} disabled={isButtonDisabled} />
        </div>
      </div>
    </div>
  );
}

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}
