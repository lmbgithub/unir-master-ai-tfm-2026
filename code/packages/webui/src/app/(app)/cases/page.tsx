"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { apiFetch } from "@/lib/api";
import { CasesFilters } from "@/components/cases/cases-filters";
import { CasesTable, CasesTableSkeleton } from "@/components/cases/cases-table";
import { NewCaseButton } from "@/components/cases/new-case-dialog";
import type { Case, CaseStatus, PaginatedCases } from "@/types/api";

const VALID_STATUSES: CaseStatus[] = [
  "triage",
  "triage_validation",
  "pending_care",
  "in_care",
  "closed_success",
  "closed_Dead",
  "closed_transfer",
];

export default function CasesPage() {
  const searchParams = useSearchParams();
  const searchParamsKey = searchParams.toString();
  const [data, setData] = useState<PaginatedCases | null>(null);

  useEffect(() => {
    const page = Number(searchParams.get("page") ?? 1);
    const page_size = Number(searchParams.get("page_size") ?? 20);
    const query = new URLSearchParams({ page: String(page), page_size: String(page_size) });
    const search = searchParams.get("search");
    const status = searchParams.get("status") as CaseStatus | null;
    if (search) query.set("search", search);
    if (status && VALID_STATUSES.includes(status)) query.set("status", status);

    apiFetch<PaginatedCases | Case[]>(`/cases?${query}`)
      .then((raw) => {
        setData(Array.isArray(raw) ? { items: raw, total: raw.length, page, page_size } : raw);
      })
      .catch(() => {
        setData({ items: [], total: 0, page, page_size });
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParamsKey]);

  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Cases</h1>
        <NewCaseButton />
      </div>
      <CasesFilters />
      {data === null ? <CasesTableSkeleton /> : <CasesTable data={data} />}
    </div>
  );
}
