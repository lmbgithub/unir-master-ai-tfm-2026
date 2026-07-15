import type { StepStatus } from "@/types/api";

const variants: Record<StepStatus, string> = {
  pending: "border border-border text-muted-foreground bg-transparent",
  in_progress: "bg-yellow-100 text-yellow-800",
  done: "bg-green-100 text-green-800",
  error: "bg-red-100 text-red-800",
};

const labels: Record<StepStatus, string> = {
  pending: "Pending",
  in_progress: "In Progress",
  done: "Done",
  error: "Error",
};

export function StepStatusBadge({ status }: { status: StepStatus }) {
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${variants[status]}`}>
      {labels[status]}
    </span>
  );
}
