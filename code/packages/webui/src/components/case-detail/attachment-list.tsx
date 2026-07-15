"use client";

import { FileText, FileAudio, FileImage, File, Loader2 } from "lucide-react";
import { useAppDispatch } from "@/store/hooks";
import { openAttachment } from "@/store/uiSlice";
import type { Attachment, AttachmentStatus } from "@/types/api";
import { AudioPlayer } from "./audio-player";
import { ConfidenceBadge } from "./confidence-badge";

function AttachmentStatusBadge({ status }: { status: AttachmentStatus }) {
  if (status === "processing") {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-yellow-100 text-yellow-800">
        <Loader2 className="w-3 h-3 animate-spin" />
        processing
      </span>
    );
  }
  const cls: Record<Exclude<AttachmentStatus, "processing">, string> = {
    pending: "border border-border text-muted-foreground bg-transparent",
    done: "bg-green-100 text-green-800",
    error: "bg-red-100 text-red-800",
    cancelled: "bg-muted text-muted-foreground",
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${cls[status]}`}>{status}</span>
  );
}

function FileIcon({ mimeType }: { mimeType: string }) {
  if (mimeType.startsWith("audio/")) return <FileAudio className="w-4 h-4 shrink-0" />;
  if (mimeType.startsWith("image/")) return <FileImage className="w-4 h-4 shrink-0" />;
  if (mimeType === "application/pdf") return <FileText className="w-4 h-4 shrink-0" />;
  return <File className="w-4 h-4 shrink-0" />;
}

interface AttachmentListProps {
  attachments: Attachment[];
  active?: boolean;
}

export function AttachmentList({ attachments, active = true }: AttachmentListProps) {
  const dispatch = useAppDispatch();

  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-sm font-medium">Attachments</span>
        <span className="inline-flex items-center px-1.5 py-0.5 rounded bg-muted text-muted-foreground text-xs">
          {attachments.length}
        </span>
      </div>
      {attachments.length === 0 ? (
        <p className="text-sm text-muted-foreground">No attachments</p>
      ) : (
        <ul className="space-y-1">
          {attachments.map((att) => (
            <li key={att.id} className="flex items-center gap-1">
              {att.mime_type.startsWith("audio/") && <AudioPlayer attachmentId={att.id} size="sm" active={active} />}
              <button
                type="button"
                onClick={() => dispatch(openAttachment(att.id))}
                className="flex items-center gap-2 flex-1 text-left px-2 py-1.5 rounded hover:bg-muted transition-colors text-sm min-w-0"
              >
                <FileIcon mimeType={att.mime_type} />
                <span className="truncate text-xs">{att.original_filename}</span>
                <span className="flex-1" />
                <AttachmentStatusBadge status={att.status} />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
