"use client";

import { useEffect, useState } from "react";
import { X, Download, Eye, Loader2 } from "lucide-react";
import { useAppSelector, useAppDispatch } from "@/store/hooks";
import { closeAttachment } from "@/store/uiSlice";
import { attachmentService } from "@/services/AttachmentService";
import type { Attachment, AttachmentKind, AttachmentStatus } from "@/types/api";
import { AudioPlayer } from "./audio-player";
import { ConfidenceBadge } from "./confidence-badge";

const KIND_LABELS: Record<AttachmentKind, string> = {
  image: "Image",
  pdf: "PDF",
  audio: "Audio",
};

const STATUS_CLASSES: Record<AttachmentStatus, string> = {
  pending: "border border-border text-muted-foreground bg-transparent",
  processing: "bg-yellow-100 text-yellow-800",
  done: "bg-green-100 text-green-800",
  error: "bg-red-100 text-red-800",
  cancelled: "bg-muted text-muted-foreground",
};

function Badge({ label, className }: { label: string; className: string }) {
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${className}`}>{label}</span>
  );
}

function Skeleton() {
  return (
    <div className="p-6 space-y-4 animate-pulse">
      <div className="h-5 bg-muted rounded w-2/3" />
      <div className="flex gap-2">
        <div className="h-4 bg-muted rounded w-16" />
        <div className="h-4 bg-muted rounded w-16" />
      </div>
      <div className="h-4 bg-muted rounded w-1/3" />
      <div className="mt-6 space-y-2">
        <div className="h-4 bg-muted rounded w-1/4" />
        <div className="h-32 bg-muted rounded" />
      </div>
    </div>
  );
}

export function AttachmentSidebar() {
  const dispatch = useAppDispatch();
  const id = useAppSelector((s) => s.ui.selectedAttachmentId);
  const open = id !== null;

  const [data, setData] = useState<Attachment | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [imagePreviewOpen, setImagePreviewOpen] = useState(false);

  useEffect(() => {
    if (!id) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setData(null);
      setImagePreviewUrl(null);
      setImagePreviewOpen(false);
      return;
    }
    let cancelled = false;
    attachmentService
      .get(id)
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch(() => {
        if (!cancelled) setData(null);
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  async function handleDownload() {
    if (!data) return;
    setDownloading(true);
    try {
      const blob = await attachmentService.download(data.id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = data.original_filename;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setDownloading(false);
    }
  }

  async function handlePreview() {
    if (!data) return;
    if (!imagePreviewUrl) {
      const blob = await attachmentService.download(data.id);
      setImagePreviewUrl(URL.createObjectURL(blob));
    }
    setImagePreviewOpen(true);
  }

  function handleClosePreview() {
    setImagePreviewOpen(false);
  }

  return (
    <>
      {imagePreviewOpen && imagePreviewUrl && (
        <div
          className="fixed inset-0 z-[100] bg-black/90 flex items-center justify-center"
          onClick={handleClosePreview}
        >
          <button
            type="button"
            onClick={handleClosePreview}
            className="absolute top-4 right-4 p-2 rounded-full bg-white/10 hover:bg-white/20 text-white transition-colors"
            title="Close"
          >
            <X className="w-5 h-5" />
          </button>
          <img
            src={imagePreviewUrl}
            alt={data?.original_filename ?? "preview"}
            className="max-h-[90vh] max-w-[90vw] object-contain rounded shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}

      {open && <div className="fixed inset-0 bg-black/30 z-40" onClick={() => dispatch(closeAttachment())} />}

      <div
        className={`fixed inset-y-0 right-0 z-50 w-[480px] max-w-full bg-white shadow-2xl border-l flex flex-col transition-transform duration-300 ease-in-out ${
          open ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between px-4 py-3 border-b sticky top-0 bg-white">
          <span className="font-semibold text-sm truncate flex-1">{data?.original_filename ?? "Loading…"}</span>
          <div className="flex items-center gap-2 ml-2 shrink-0">
            {data?.kind === "image" && (
              <button
                type="button"
                onClick={handlePreview}
                className="p-1.5 rounded hover:bg-muted transition-colors"
                title="View image"
              >
                <Eye className="w-4 h-4" />
              </button>
            )}
            <button
              type="button"
              onClick={handleDownload}
              disabled={!data || downloading}
              className="p-1.5 rounded hover:bg-muted transition-colors disabled:opacity-50"
              title="Download"
            >
              {downloading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
            </button>
            <button
              type="button"
              onClick={() => dispatch(closeAttachment())}
              className="p-1.5 rounded hover:bg-muted transition-colors"
              title="Close"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {open && !data ? (
          <Skeleton />
        ) : data ? (
          <div className="flex-1 overflow-y-auto p-6 space-y-1">
            <div className="flex items-center gap-2 flex-wrap">
              <Badge label={KIND_LABELS[data.kind]} className="bg-muted text-muted-foreground" />
              <Badge label={data.status} className={STATUS_CLASSES[data.status]} />
              {data.kind === "audio" && <AudioPlayer attachmentId={data.id} size="md" active={open} />}
            </div>
            <p className="text-xs text-muted-foreground pt-1">{new Date(data.created_at).toLocaleString()}</p>

            {data.summary && (
              <div className="pt-6">
                <p className="text-sm font-medium mb-2">Summary</p>
                <p className="text-sm text-foreground bg-muted/40 rounded p-3">{data.summary}</p>
              </div>
            )}

            <div className="pt-4">
              <p className="text-sm font-medium mb-2">Transcription</p>
              {data.transcription ? (
                <pre className="text-sm whitespace-pre-wrap font-sans text-foreground bg-muted/40 rounded p-3 overflow-auto max-h-[40vh]">
                  {data.transcription}
                </pre>
              ) : (
                <p className="text-sm text-muted-foreground">No transcription available</p>
              )}
            </div>

            {data.sbar && Object.keys(data.sbar).length > 0 && (
              <div className="pt-4">
                <p className="text-sm font-medium mb-2">SBAR</p>
                <table className="w-full text-sm border-collapse">
                  <tbody>
                    {(["situation", "background", "assessment", "recommendation"] as const).map((key) => {
                      const value = (data.sbar as Record<string, string>)[key];
                      if (!value) return null;
                      return (
                        <tr key={key} className="border-b border-border last:border-0">
                          <td className="py-1.5 pr-3 text-muted-foreground font-medium capitalize align-top w-2/5">
                            {key}
                          </td>
                          <td className="py-1.5 text-foreground align-top">{value}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {data.kind !== "audio" && data.ner && Object.keys(data.ner).length > 0 && (
              <div className="pt-4">
                <p className="text-sm font-medium mb-2">Extracted Entities</p>
                <table className="w-full text-sm border-collapse">
                  <tbody>
                    {Object.entries(data.ner).map(([key, value]) => (
                      <tr key={key} className="border-b border-border last:border-0">
                        <td className="py-1.5 pr-3 text-muted-foreground font-medium capitalize align-top w-2/5">
                          {key.replace(/_/g, " ")}
                        </td>
                        <td className="py-1.5 text-foreground align-top">{value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ) : null}
      </div>
    </>
  );
}
