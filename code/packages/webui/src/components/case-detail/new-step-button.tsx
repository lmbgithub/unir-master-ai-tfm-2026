"use client";

import { useRef, useState } from "react";
import { Upload, X, FileText, ImageIcon, Music, Plus, Mic, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { caseService } from "@/services/CaseService";
import { useAudioRecorder, MAX_RECORDING_SECONDS } from "@/hooks/use-audio-recorder";
import type { AttachmentKind, Case } from "@/types/api";

const MAX_FILE_SIZE = 5 * 1024 * 1024;

const ALL_TYPES: Record<string, AttachmentKind> = {
  "image/jpeg": "image",
  "image/png": "image",
  "image/gif": "image",
  "application/pdf": "pdf",
  "audio/wav": "audio",
  "audio/mpeg": "audio",
  "audio/mp3": "audio",
};

const AUDIO_ONLY_TYPES: Record<string, AttachmentKind> = {
  "audio/wav": "audio",
  "audio/mpeg": "audio",
  "audio/mp3": "audio",
};

type StepKind = "handoff" | "regular";

interface PendingFile {
  file: File;
  kind: AttachmentKind;
}

function kindIcon(kind: AttachmentKind) {
  if (kind === "pdf") return <FileText className="h-4 w-4 text-red-500" />;
  if (kind === "audio") return <Music className="h-4 w-4 text-blue-500" />;
  return <ImageIcon className="h-4 w-4 text-green-500" />;
}

function formatTime(seconds: number) {
  const mm = Math.floor(seconds / 60);
  const ss = seconds % 60;
  return `${mm}:${ss.toString().padStart(2, "0")}`;
}

interface NewStepButtonProps {
  caseId: string;
  onSuccess?: (updated: Case) => void;
}

export function NewStepButton({ caseId, onSuccess, disabled }: NewStepButtonProps & { disabled: boolean }) {
  const [open, setOpen] = useState(false);
  const [stepType, setStepType] = useState<StepKind>("regular");
  const [description, setDescription] = useState("");
  const [pendingFiles, setPendingFiles] = useState<PendingFile[]>([]);
  const [fileError, setFileError] = useState<string | null>(null);
  const [serverError, setServerError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recorder = useAudioRecorder();

  const acceptedTypes = stepType === "handoff" ? AUDIO_ONLY_TYPES : ALL_TYPES;

  function addFiles(files: File[]) {
    setFileError(null);
    const next: PendingFile[] = [];
    for (const file of files) {
      const kind = acceptedTypes[file.type];
      if (!kind) {
        setFileError(
          stepType === "handoff"
            ? `Unsupported file type: ${file.name}. Handoff steps only accept WAV and MP3.`
            : `Unsupported file type: ${file.name}`
        );
        continue;
      }
      if (file.size > MAX_FILE_SIZE) {
        setFileError(`${file.name} exceeds 5 MB limit`);
        continue;
      }
      next.push({ file, kind });
    }
    setPendingFiles((prev) => [...prev, ...next]);
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    addFiles(Array.from(e.target.files ?? []));
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function handleDrop(e: React.DragEvent<HTMLButtonElement>) {
    e.preventDefault();
    setDragging(false);
    addFiles(Array.from(e.dataTransfer.files));
  }

  function handleStepTypeChange(value: StepKind) {
    setStepType(value);
    // Clear files that don't match the new type
    if (value === "handoff") {
      setPendingFiles((prev) => prev.filter((f) => f.kind === "audio"));
    }
    setFileError(null);
  }

  async function handleStopRecording() {
    const file = await recorder.stop();
    if (file) addFiles([file]);
  }

  function handleClose(nextOpen: boolean) {
    if (!nextOpen) {
      recorder.cancel();
      setStepType("regular");
      setDescription("");
      setPendingFiles([]);
      setFileError(null);
      setServerError(null);
    }
    setOpen(nextOpen);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setServerError(null);
    try {
      const step = await caseService.createStep(caseId, {
        type: stepType,
        description: description.trim() || undefined,
      });

      if (pendingFiles.length > 0) {
        await Promise.all(
          pendingFiles.map(({ file, kind }) => caseService.uploadAttachment(caseId, step.id, file, kind))
        );
      }

      await caseService.submitStep(caseId, step.id);
      const updated = await caseService.get(caseId);
      setOpen(false);
      onSuccess?.(updated);
    } catch {
      setServerError("Something went wrong while creating the step. Please try again.");
    } finally {
      setSubmitting(false);
    }
  }

  const acceptAttr =
    stepType === "handoff"
      ? ".wav,.mp3,audio/wav,audio/mpeg"
      : ".jpg,.jpeg,.png,.gif,.pdf,.wav,.mp3,audio/mpeg,audio/wav,image/jpeg,image/png,image/gif,application/pdf";

  const uploadLabel =
    stepType === "handoff"
      ? "Upload or drag audio files (WAV, MP3 · max 5 MB each)"
      : "Upload or drag files (JPG, PNG, GIF, PDF, WAV, MP3 · max 5 MB each)";

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogTrigger asChild>
        <Button size="sm" variant="outline" className="gap-1.5" disabled={disabled}>
          <Plus className="h-4 w-4" />
          New Step
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-lg max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>New Step</DialogTitle>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-5 mt-2">
          {serverError && (
            <div className="rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">{serverError}</div>
          )}

          <div className="space-y-1.5">
            <Label>Step Type</Label>
            <Select value={stepType} onValueChange={handleStepTypeChange}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="regular">Regular</SelectItem>
                <SelectItem value="handoff">Handoff</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="description">Description</Label>
            <Textarea
              id="description"
              placeholder="Describe this step…"
              rows={4}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />
          </div>

          <section className="space-y-3">
            <p className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
              Attachments (optional)
            </p>

            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              className={`flex w-full items-center justify-center gap-2 rounded-md border-2 border-dashed py-4 text-sm transition-colors ${
                dragging
                  ? "border-primary bg-primary/5 text-primary"
                  : "border-border text-muted-foreground hover:border-primary hover:text-primary"
              }`}
            >
              <Upload className="h-4 w-4" />
              {dragging ? "Drop files here" : uploadLabel}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept={acceptAttr}
              className="hidden"
              onChange={handleFileChange}
            />

            {recorder.supported &&
              (recorder.status === "recording" ? (
                <div className="flex items-center gap-3 rounded-md border bg-muted/40 px-3 py-2">
                  <span className="relative flex h-3 w-3">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-500 opacity-75" />
                    <span className="relative inline-flex h-3 w-3 rounded-full bg-red-500" />
                  </span>
                  <span className="flex-1 text-sm tabular-nums text-muted-foreground">
                    {formatTime(recorder.seconds)}
                    <span className="opacity-60"> / {formatTime(MAX_RECORDING_SECONDS)}</span>
                  </span>
                  <Button type="button" size="sm" variant="ghost" onClick={recorder.cancel}>
                    Cancel
                  </Button>
                  <Button type="button" size="sm" onClick={handleStopRecording} className="gap-1.5">
                    <Square className="h-3.5 w-3.5 fill-current" />
                    Stop
                  </Button>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={recorder.start}
                  disabled={recorder.status === "requesting"}
                  className="flex w-full items-center justify-center gap-2 rounded-md border py-2.5 text-sm text-muted-foreground transition-colors hover:border-primary hover:text-primary disabled:opacity-60"
                >
                  <Mic className="h-4 w-4" />
                  {recorder.status === "requesting" ? "Requesting microphone…" : "Record audio note"}
                </button>
              ))}

            {recorder.error && <p className="text-xs text-destructive">{recorder.error}</p>}

            {fileError && <p className="text-xs text-destructive">{fileError}</p>}

            {pendingFiles.length > 0 && (
              <ul className="space-y-1.5">
                {pendingFiles.map(({ file, kind }, i) => (
                  <li key={i} className="flex items-center gap-2 rounded-md border bg-muted/40 px-3 py-2 text-sm">
                    {kindIcon(kind)}
                    <span className="flex-1 truncate">{file.name}</span>
                    <span className="text-xs text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                    <button
                      type="button"
                      onClick={() => setPendingFiles((prev) => prev.filter((_, idx) => idx !== i))}
                      className="ml-1 rounded-sm opacity-60 hover:opacity-100"
                      aria-label="Remove file"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <div className="flex justify-end gap-2 pt-1">
            <Button type="button" variant="outline" onClick={() => handleClose(false)} disabled={submitting}>
              Cancel
            </Button>
            <Button type="submit" disabled={submitting}>
              {submitting ? "Creating…" : "Create Step"}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
