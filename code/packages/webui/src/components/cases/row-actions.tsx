"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { MoreHorizontal } from "lucide-react";
import { DeleteCaseDialog } from "./delete-case-dialog";
import { ChangeStatusDialog } from "./change-status-dialog";
import type { Case } from "@/types/api";

interface RowActionsProps {
  case: Case;
}

export function RowActions({ case: c }: RowActionsProps) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [dialog, setDialog] = useState<"delete" | "status" | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, [open]);

  function closeDialog() {
    setDialog(null);
  }

  return (
    <>
      <div ref={menuRef} className="relative">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex h-7 w-7 items-center justify-center rounded hover:bg-muted"
          aria-label="Actions"
        >
          <MoreHorizontal className="h-4 w-4" />
        </button>

        {open && (
          <div className="absolute right-0 z-10 mt-1 w-44 rounded-md border border-border bg-card shadow-md">
            <button
              type="button"
              className="flex w-full items-center px-3 py-2 text-sm hover:bg-muted"
              onClick={() => {
                setOpen(false);
                router.push(`/cases/${c.id}`);
              }}
            >
              See detail
            </button>
            <button
              type="button"
              className="flex w-full items-center px-3 py-2 text-sm hover:bg-muted"
              onClick={() => {
                setOpen(false);
                setDialog("status");
              }}
            >
              Change status
            </button>
            <button
              type="button"
              className="flex w-full items-center px-3 py-2 text-sm text-destructive hover:bg-muted"
              onClick={() => {
                setOpen(false);
                setDialog("delete");
              }}
            >
              Delete
            </button>
          </div>
        )}
      </div>

      {dialog === "delete" && <DeleteCaseDialog id={c.id} onClose={closeDialog} />}
      {dialog === "status" && <ChangeStatusDialog id={c.id} currentStatus={c.phase} onClose={closeDialog} />}
    </>
  );
}
