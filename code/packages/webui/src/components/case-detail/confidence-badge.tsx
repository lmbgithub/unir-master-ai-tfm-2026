"use client";

import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface ConfidenceBadgeProps {
  confidence: number;
  size?: "sm" | "md";
}

export function ConfidenceBadge({ confidence, size = "md" }: ConfidenceBadgeProps) {
  const pct = Math.round(confidence * 100);
  const cls =
    confidence >= 0.8
      ? "bg-green-100 text-green-800"
      : confidence >= 0.5
        ? "bg-yellow-100 text-yellow-800"
        : "bg-red-100 text-red-800";
  const padding = size === "sm" ? "px-1.5 py-0.5" : "px-2 py-0.5";
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span className={`inline-flex items-center ${padding} rounded text-xs font-medium ${cls}`}>{pct}%</span>
        </TooltipTrigger>
        <TooltipContent>Confidence</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
