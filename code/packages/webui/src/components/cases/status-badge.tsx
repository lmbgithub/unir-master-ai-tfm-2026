import { cn } from "@/lib/utils";
import type { CaseStatus } from "@/types/api";

const STATUS_LABELS: Record<CaseStatus, string> = {
  triage: "Triage",
  triage_validation: "Triage Validation",
  pending_care: "Waiting attention",
  in_care: "In attention",
  closed_success: "Discharged",
  closed_Dead: "Dead",
  closed_transfer: "Transfered",
};

const STATUS_CLASSES: Record<CaseStatus, string> = {
  triage: "border border-border bg-background text-foreground",
  triage_validation: "bg-secondary text-secondary-foreground",
  pending_care: "bg-primary text-primary-foreground",
  in_care: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
  closed_success: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
  closed_Dead: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
  closed_transfer: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
};

interface StatusBadgeProps {
  status: CaseStatus;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
        STATUS_CLASSES[status],
        className
      )}
    >
      {STATUS_LABELS[status]}
    </span>
  );
}
