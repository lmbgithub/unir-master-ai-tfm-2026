"use client";

import { useState } from "react";
import { ChangeStatusDialog } from "@/components/cases/change-status-dialog";
import type { Case } from "@/types/api";
import { Button } from "@/components/ui/button";

interface ChangeStatusButtonProps {
  case: Case;
  disabled: boolean;
}

export function ChangeStatusButton({ case: caseData, disabled }: ChangeStatusButtonProps) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <Button size="sm" variant="outline" type="button" onClick={() => setOpen(true)} disabled={disabled}>
        Change Status
      </Button>
      {open && <ChangeStatusDialog id={caseData.id} currentStatus={caseData.phase} onClose={() => setOpen(false)} />}
    </>
  );
}
