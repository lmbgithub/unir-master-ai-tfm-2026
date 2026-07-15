"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { StatusBadge } from "./status-badge";
import { RowActions } from "./row-actions";
import type { Case, PaginatedCases } from "@/types/api";

function relativeDate(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function shortId(id: string): string {
  return id.slice(0, 8);
}

function truncate(text: string, max: number): string {
  return text.length <= max ? text : text.slice(0, max) + "…";
}

interface CasesTableProps {
  data: PaginatedCases;
}

export function CasesTable({ data }: CasesTableProps) {
  const router = useRouter();
  const searchParams = useSearchParams();

  const totalPages = Math.max(1, Math.ceil(data.total / data.page_size));

  function goToPage(page: number) {
    const params = new URLSearchParams(searchParams.toString());
    params.set("page", String(page));
    router.replace(`?${params.toString()}`);
  }

  if (data.items.length === 0) {
    return (
      <div className="rounded-lg border bg-card">
        <div className="flex h-48 items-center justify-center text-sm text-muted-foreground">No cases found</div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="rounded-lg border bg-card">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50 text-left text-xs font-medium text-muted-foreground uppercase tracking-wide">
              <th className="px-4 py-3">ID</th>
              <th className="px-4 py-3">Patient</th>
              <th className="px-4 py-3">Main complaint</th>
              <th className="px-4 py-3">ESI</th>
              <th className="px-4 py-3">Status</th>
              <th className="px-4 py-3">Created</th>
              <th className="px-4 py-3 w-10" />
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {data.items.map((c: Case) => (
              <tr key={c.id} className="hover:bg-muted/30 transition-colors">
                <td className="px-4 py-3 font-mono">
                  <Link href={`/cases/${c.id}`} className="text-primary hover:underline">
                    {shortId(c.id)}
                  </Link>
                </td>
                <td className="px-4 py-3 font-medium">{c.patient_info.name}</td>
                <td className="px-4 py-3 text-muted-foreground">{truncate(c.chief_complaint, 60)}</td>
                <td className="px-4 py-3">
                  {c.esi_level !== null ? (
                    <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-foreground">
                      {c.esi_level}
                    </span>
                  ) : (
                    <span className="text-muted-foreground">—</span>
                  )}
                </td>
                <td className="px-4 py-3">
                  <StatusBadge status={c.phase} />
                </td>
                <td className="px-4 py-3 text-muted-foreground">{relativeDate(c.created_at)}</td>
                <td className="px-4 py-3">
                  <RowActions case={c} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between px-1 text-sm text-muted-foreground">
        <span>
          Page {data.page} of {totalPages} · {data.total} result
          {data.total !== 1 ? "s" : ""}
        </span>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => goToPage(data.page - 1)}
            disabled={data.page <= 1}
            className="flex h-8 w-8 items-center justify-center rounded border border-border hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
            aria-label="Previous page"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => goToPage(data.page + 1)}
            disabled={data.page >= totalPages}
            className="flex h-8 w-8 items-center justify-center rounded border border-border hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
            aria-label="Next page"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}

export function CasesTableSkeleton() {
  return (
    <div className="rounded-lg border bg-card">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b bg-muted/50 text-left text-xs font-medium text-muted-foreground uppercase tracking-wide">
            <th className="px-4 py-3">ID</th>
            <th className="px-4 py-3">Patient</th>
            <th className="px-4 py-3">Main complaint</th>
            <th className="px-4 py-3">ESI</th>
            <th className="px-4 py-3">Status</th>
            <th className="px-4 py-3">Created</th>
            <th className="px-4 py-3 w-10" />
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {Array.from({ length: 5 }).map((_, i) => (
            <tr key={i}>
              {Array.from({ length: 7 }).map((_, j) => (
                <td key={j} className="px-4 py-3">
                  <div className="h-4 animate-pulse rounded bg-muted" />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
