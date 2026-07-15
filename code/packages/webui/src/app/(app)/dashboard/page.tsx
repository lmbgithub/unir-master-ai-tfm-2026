import { Suspense } from "react";
import { cookies } from "next/headers";
import { apiFetch } from "@/lib/api";
import { StatCard, StatCardSkeleton } from "@/components/dashboard/stat-card";
import type { DashboardStats } from "@/types/api";

async function fetchStats(cookieHeader: string): Promise<DashboardStats> {
  // Falls back to zeros if the API is unreachable
  try {
    return await apiFetch<DashboardStats>("/cases/stats", {
      headers: { Cookie: cookieHeader },
    });
  } catch {
    return { total: 0, open: 0, completed: 0, error: 0 };
  }
}

async function StatsGrid() {
  const authCookie = (await cookies()).get("auth")?.value ?? "";
  const stats = await fetchStats(`auth=${authCookie}`);

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <StatCard label="Total Cases" value={stats.total} />
      <StatCard label="Open" value={stats.open} />
      <StatCard label="Completed" value={stats.completed} />
      <StatCard label="Error" value={stats.error} />
    </div>
  );
}

function StatsSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <StatCardSkeleton />
      <StatCardSkeleton />
      <StatCardSkeleton />
      <StatCardSkeleton />
    </div>
  );
}

export default function DashboardPage() {
  return (
    <div className="space-y-6  p-4">
      <h1 className="text-2xl font-semibold">Dashboard</h1>
      <Suspense fallback={<StatsSkeleton />}>
        <StatsGrid />
      </Suspense>
    </div>
  );
}
