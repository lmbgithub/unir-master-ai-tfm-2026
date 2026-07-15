import { TrendingUpIcon } from "lucide-react";

import { Card, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";

interface StatCardProps {
  label: string;
  value: number | string;
  description?: string;
}

export function StatCard({ label, value, description }: StatCardProps) {
  return (
    <Card>
      <CardHeader className="relative">
        <CardDescription>{label}</CardDescription>
        <CardTitle className="text-3xl font-semibold tabular-nums">{value}</CardTitle>
      </CardHeader>
      {description && (
        <CardFooter className="flex-col items-start gap-1 text-sm">
          <div className="flex gap-2 font-medium text-muted-foreground">
            <TrendingUpIcon className="size-4" />
            {description}
          </div>
        </CardFooter>
      )}
    </Card>
  );
}

export function StatCardSkeleton() {
  return (
    <Card>
      <CardHeader>
        <div className="h-4 w-24 animate-pulse rounded bg-muted" />
        <div className="mt-2 h-10 w-16 animate-pulse rounded bg-muted" />
      </CardHeader>
    </Card>
  );
}
