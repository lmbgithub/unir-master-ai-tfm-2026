import { cookies } from "next/headers";
import { apiFetch } from "@/lib/api";
import { CaseDetailClient } from "@/components/case-detail/case-detail-client";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import type { Case } from "@/types/api";

interface PageProps {
  params: Promise<{ id: string }>;
}

export default async function CaseDetailPage({ params }: PageProps) {
  const { id } = await params;
  const authCookie = (await cookies()).get("auth")?.value ?? "";

  let caseData: Case | null = null;
  try {
    caseData = await apiFetch<Case>(`/cases/${id}`, {
      headers: { Cookie: `auth=${authCookie}` },
    });
  } catch {
    // case not found or fetch error — show inline alert
  }

  if (!caseData) {
    return (
      <div className="p-4">
        <Alert variant="destructive">
          <AlertTitle>Case not found</AlertTitle>
          <AlertDescription>
            No case with ID <span className="font-mono">{id}</span> exists.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return <CaseDetailClient initial={caseData} />;
}
