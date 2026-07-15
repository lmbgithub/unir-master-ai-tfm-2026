"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { format, parse, isValid } from "date-fns";
import { CalendarIcon, Plus, Upload, X, FileText, ImageIcon, Music } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { caseService } from "@/services/CaseService";
import type { Attachment, AttachmentKind, Case, CaseStep } from "@/types/api";

const MAX_FILE_SIZE = 5 * 1024 * 1024;

const ACCEPTED_TYPES: Record<string, AttachmentKind> = {
  "image/jpeg": "image",
  "image/png": "image",
  "image/gif": "image",
  "application/pdf": "pdf",
  "audio/wav": "audio",
  "audio/mpeg": "audio",
  "audio/mp3": "audio",
};

const schema = z.object({
  name: z.string({ error: "Required" }).min(2, "At least 2 characters"),
  gender: z.string({ error: "Required" }).min(1, "Required"),
  date_of_birth: z.string({ error: "Required" }).min(1, "Required"),
  id_number: z.string({ error: "Required" }).min(1, "Required"),
  blood_type: z.string({ error: "Required" }).min(1, "Required"),
  blood_rh: z.string({ error: "Required" }).min(1, "Required"),
  blood_pressure_systolic: z.number({ error: "Required" }).int().min(40, "Min 40").max(300, "Max 300"),
  blood_pressure_diastolic: z.number({ error: "Required" }).int().min(20, "Min 20").max(200, "Max 200"),
  weight: z.number({ error: "Required" }).positive("Must be positive").max(500),
  height: z.number({ error: "Required" }).positive("Must be positive").max(300),
  pulse: z.number({ error: "Required" }).int().min(20, "Min 20").max(300, "Max 300"),
  allergies: z.string().optional(),
  chronic_conditions: z.string().optional(),
  chief_complaint: z.string({ error: "Required" }).min(5, "Describe symptoms (at least 5 characters)"),
});

type FormValues = z.infer<typeof schema>;

function splitComma(value: string | undefined): string[] {
  return value
    ? value
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean)
    : [];
}

function kindIcon(kind: AttachmentKind) {
  if (kind === "pdf") return <FileText className="h-4 w-4 text-red-500" />;
  if (kind === "audio") return <Music className="h-4 w-4 text-blue-500" />;
  return <ImageIcon className="h-4 w-4 text-green-500" />;
}

interface PendingFile {
  file: File;
  kind: AttachmentKind;
}

function FieldError({ message }: { message?: string }) {
  if (!message) return null;
  return <p className="text-xs text-destructive mt-1">{message}</p>;
}

function caseToFormValues(c: Case): Partial<FormValues> {
  const p = c.patient_info;
  return {
    name: p.name,
    gender: p.gender,
    date_of_birth: p.date_of_birth,
    id_number: p.id_number,
    blood_type: p.blood_type,
    blood_rh: String(p.blood_rh),
    blood_pressure_systolic: p.blood_pressure_systolic,
    blood_pressure_diastolic: p.blood_pressure_diastolic,
    weight: p.weight,
    height: p.height,
    pulse: p.pulse,
    allergies: p.allergies.join(", "),
    chronic_conditions: p.chronic_conditions.join(", "),
    chief_complaint: c.chief_complaint,
  };
}

// ─── New Case Button ──────────────────────────────────────────────────────────

export function NewCaseButton() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [pendingFiles, setPendingFiles] = useState<PendingFile[]>([]);
  const [fileError, setFileError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [serverError, setServerError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const {
    register,
    handleSubmit,
    reset,
    control,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    mode: "onChange",
  });

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    setFileError(null);
    const files = Array.from(e.target.files ?? []);
    const next: PendingFile[] = [];
    for (const file of files) {
      const kind = ACCEPTED_TYPES[file.type];
      if (!kind) {
        setFileError(`Unsupported file type: ${file.name}`);
        continue;
      }
      if (file.size > MAX_FILE_SIZE) {
        setFileError(`${file.name} exceeds 5 MB limit`);
        continue;
      }
      next.push({ file, kind });
    }
    setPendingFiles((prev) => [...prev, ...next]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function handleDrop(e: React.DragEvent<HTMLButtonElement>) {
    e.preventDefault();
    setDragging(false);
    const files = Array.from(e.dataTransfer.files);
    const next: PendingFile[] = [];
    for (const file of files) {
      const kind = ACCEPTED_TYPES[file.type];
      if (!kind) {
        setFileError(`Unsupported file type: ${file.name}`);
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

  function handleClose(nextOpen: boolean) {
    if (!nextOpen) {
      reset();
      setPendingFiles([]);
      setFileError(null);
      setServerError(null);
    }
    setOpen(nextOpen);
  }

  async function onSubmit(values: FormValues) {
    setSubmitting(true);
    setServerError(null);
    try {
      const created = await caseService.create({
        patient_info: {
          name: values.name,
          gender: values.gender as "male" | "female",
          date_of_birth: values.date_of_birth,
          id_number: values.id_number,
          blood_type: values.blood_type as "A" | "B" | "O" | "AB",
          blood_rh: values.blood_rh === "true",
          blood_pressure_systolic: values.blood_pressure_systolic,
          blood_pressure_diastolic: values.blood_pressure_diastolic,
          weight: values.weight,
          height: values.height,
          pulse: values.pulse,
          allergies: splitComma(values.allergies),
          chronic_conditions: splitComma(values.chronic_conditions),
        },
        chief_complaint: values.chief_complaint,
      });

      const triageStep = created.case_steps.find((s) => s.type === "triage");
      if (triageStep) {
        if (pendingFiles.length > 0) {
          await Promise.all(
            pendingFiles.map(({ file, kind }) => caseService.uploadAttachment(created.id, triageStep.id, file, kind))
          );
        }
        await caseService.submitStep(created.id, triageStep.id);
      }

      setOpen(false);
      reset();
      setPendingFiles([]);
      router.push(`/cases/${created.id}`);
    } catch {
      setServerError("Something went wrong while creating the case. Please check your information and try again.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogTrigger asChild>
        <Button size="sm" className="gap-1.5">
          <Plus className="h-4 w-4" />
          New Case
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>New Case</DialogTitle>
        </DialogHeader>
        <CaseForm
          errors={errors}
          register={register}
          control={control}
          pendingFiles={pendingFiles}
          fileError={fileError}
          serverError={serverError}
          submitting={submitting}
          dragging={dragging}
          fileInputRef={fileInputRef}
          onFileChange={handleFileChange}
          onDrop={handleDrop}
          onDragOver={() => setDragging(true)}
          onDragLeave={() => setDragging(false)}
          onRemoveFile={(i) => setPendingFiles((prev) => prev.filter((_, idx) => idx !== i))}
          onCancel={() => handleClose(false)}
          onSubmit={handleSubmit(onSubmit)}
          submitLabel={submitting ? "Creating…" : "Create Case"}
        />
      </DialogContent>
    </Dialog>
  );
}

// ─── Edit Triage Dialog ───────────────────────────────────────────────────────

interface EditTriageDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  case_: Case;
  triageStep: CaseStep;
  onSuccess: (updated: Case) => void;
}

export function EditTriageDialog({ open, onOpenChange, case_, triageStep, onSuccess }: EditTriageDialogProps) {
  const [pendingFiles, setPendingFiles] = useState<PendingFile[]>([]);
  const [existingAttachments, setExistingAttachments] = useState<Attachment[]>(triageStep.attachments);
  const [fileError, setFileError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [serverError, setServerError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const {
    register,
    handleSubmit,
    reset,
    control,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    mode: "onChange",
    defaultValues: caseToFormValues(case_),
  });

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    setFileError(null);
    const files = Array.from(e.target.files ?? []);
    const next: PendingFile[] = [];
    for (const file of files) {
      const kind = ACCEPTED_TYPES[file.type];
      if (!kind) {
        setFileError(`Unsupported file type: ${file.name}`);
        continue;
      }
      if (file.size > MAX_FILE_SIZE) {
        setFileError(`${file.name} exceeds 5 MB limit`);
        continue;
      }
      next.push({ file, kind });
    }
    setPendingFiles((prev) => [...prev, ...next]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function handleDrop(e: React.DragEvent<HTMLButtonElement>) {
    e.preventDefault();
    setDragging(false);
    const files = Array.from(e.dataTransfer.files);
    const next: PendingFile[] = [];
    for (const file of files) {
      const kind = ACCEPTED_TYPES[file.type];
      if (!kind) {
        setFileError(`Unsupported file type: ${file.name}`);
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

  async function removeExistingAttachment(attachment: Attachment) {
    try {
      await caseService.deleteAttachment(case_.id, triageStep.id, attachment.id);
      setExistingAttachments((prev) => prev.filter((a) => a.id !== attachment.id));
    } catch {
      setServerError("Failed to remove attachment.");
    }
  }

  function handleClose(nextOpen: boolean) {
    if (!nextOpen) {
      reset(caseToFormValues(case_));
      setPendingFiles([]);
      setExistingAttachments(triageStep.attachments);
      setFileError(null);
      setServerError(null);
    }
    onOpenChange(nextOpen);
  }

  async function onSubmit(values: FormValues) {
    setSubmitting(true);
    setServerError(null);
    try {
      await caseService.updateCase(case_.id, {
        patient_info: {
          name: values.name,
          gender: values.gender as "male" | "female",
          date_of_birth: values.date_of_birth,
          id_number: values.id_number,
          blood_type: values.blood_type as "A" | "B" | "O" | "AB",
          blood_rh: values.blood_rh === "true",
          blood_pressure_systolic: values.blood_pressure_systolic,
          blood_pressure_diastolic: values.blood_pressure_diastolic,
          weight: values.weight,
          height: values.height,
          pulse: values.pulse,
          allergies: splitComma(values.allergies),
          chronic_conditions: splitComma(values.chronic_conditions),
        },
        chief_complaint: values.chief_complaint,
      });

      if (pendingFiles.length > 0) {
        await Promise.all(
          pendingFiles.map(({ file, kind }) => caseService.uploadAttachment(case_.id, triageStep.id, file, kind))
        );
      }

      const updated = await caseService.retryStep(case_.id, triageStep.id);
      onOpenChange(false);
      onSuccess(updated);
    } catch {
      setServerError("Something went wrong. Please check your information and try again.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Edit &amp; Retry Triage</DialogTitle>
        </DialogHeader>
        <CaseForm
          errors={errors}
          register={register}
          control={control}
          pendingFiles={pendingFiles}
          existingAttachments={existingAttachments}
          fileError={fileError}
          serverError={serverError}
          submitting={submitting}
          dragging={dragging}
          fileInputRef={fileInputRef}
          onFileChange={handleFileChange}
          onDrop={handleDrop}
          onDragOver={() => setDragging(true)}
          onDragLeave={() => setDragging(false)}
          onRemoveFile={(i) => setPendingFiles((prev) => prev.filter((_, idx) => idx !== i))}
          onRemoveExistingAttachment={removeExistingAttachment}
          onCancel={() => handleClose(false)}
          onSubmit={handleSubmit(onSubmit)}
          submitLabel={submitting ? "Saving…" : "Save & Retry Triage"}
        />
      </DialogContent>
    </Dialog>
  );
}

// ─── Shared Form ──────────────────────────────────────────────────────────────

interface CaseFormProps {
  errors: ReturnType<typeof useForm<FormValues>>["formState"]["errors"];
  register: ReturnType<typeof useForm<FormValues>>["register"];
  control: ReturnType<typeof useForm<FormValues>>["control"];
  pendingFiles: PendingFile[];
  existingAttachments?: Attachment[];
  fileError: string | null;
  serverError: string | null;
  submitting: boolean;
  dragging: boolean;
  fileInputRef: React.RefObject<HTMLInputElement>;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onDrop: (e: React.DragEvent<HTMLButtonElement>) => void;
  onDragOver: () => void;
  onDragLeave: () => void;
  onRemoveFile: (index: number) => void;
  onRemoveExistingAttachment?: (attachment: Attachment) => void;
  onCancel: () => void;
  onSubmit: (e: React.FormEvent) => void;
  submitLabel: string;
}

function CaseForm({
  errors,
  register,
  control,
  pendingFiles,
  existingAttachments,
  fileError,
  serverError,
  submitting,
  dragging,
  fileInputRef,
  onFileChange,
  onDrop,
  onDragOver,
  onDragLeave,
  onRemoveFile,
  onRemoveExistingAttachment,
  onCancel,
  onSubmit,
  submitLabel,
}: CaseFormProps) {
  return (
    <form onSubmit={onSubmit} className="space-y-6 mt-2">
      {serverError && (
        <div className="rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">{serverError}</div>
      )}

      {/* Patient Info */}
      <section className="space-y-3">
        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">Patient Information</h3>

        <div className="grid grid-cols-2 gap-3">
          <div className="col-span-2">
            <Label htmlFor="name">Full Name</Label>
            <Input id="name" placeholder="Jane Doe" className="mt-1" {...register("name")} />
            <FieldError message={errors.name?.message} />
          </div>

          <div>
            <Label htmlFor="id_number">ID / Document Number</Label>
            <Input id="id_number" placeholder="12345678" className="mt-1" {...register("id_number")} />
            <FieldError message={errors.id_number?.message} />
          </div>

          <div>
            <Label>Date of Birth</Label>
            <Controller
              control={control}
              name="date_of_birth"
              render={({ field }) => {
                const selected = field.value ? parse(field.value, "yyyy-MM-dd", new Date()) : undefined;
                return (
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button variant="outline" className="mt-1 w-full justify-start text-left font-normal">
                        <CalendarIcon className="mr-2 h-4 w-4 opacity-50" />
                        {selected && isValid(selected) ? (
                          format(selected, "MM/dd/yyyy")
                        ) : (
                          <span className="text-muted-foreground">MM/DD/YYYY</span>
                        )}
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-0" align="start">
                      <Calendar
                        mode="single"
                        selected={selected && isValid(selected) ? selected : undefined}
                        onSelect={(date) => field.onChange(date ? format(date, "yyyy-MM-dd") : "")}
                        captionLayout="dropdown"
                        startMonth={new Date(1900, 0)}
                        endMonth={new Date()}
                      />
                    </PopoverContent>
                  </Popover>
                );
              }}
            />
            <FieldError message={errors.date_of_birth?.message} />
          </div>

          <div>
            <Label>Gender</Label>
            <Controller
              control={control}
              name="gender"
              render={({ field }) => (
                <Select onValueChange={field.onChange} value={field.value ?? ""}>
                  <SelectTrigger className="mt-1">
                    <SelectValue placeholder="Select…" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="male">Male</SelectItem>
                    <SelectItem value="female">Female</SelectItem>
                  </SelectContent>
                </Select>
              )}
            />
            <FieldError message={errors.gender?.message} />
          </div>

          <div>
            <Label>Blood Type</Label>
            <div className="flex gap-2 mt-1">
              <Controller
                control={control}
                name="blood_type"
                render={({ field }) => (
                  <Select onValueChange={field.onChange} value={field.value ?? ""}>
                    <SelectTrigger>
                      <SelectValue placeholder="Type…" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="A">A</SelectItem>
                      <SelectItem value="B">B</SelectItem>
                      <SelectItem value="O">O</SelectItem>
                      <SelectItem value="AB">AB</SelectItem>
                    </SelectContent>
                  </Select>
                )}
              />
              <Controller
                control={control}
                name="blood_rh"
                render={({ field }) => (
                  <Select onValueChange={field.onChange} value={field.value ?? ""}>
                    <SelectTrigger>
                      <SelectValue placeholder="Rh…" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="true">+ (Positive)</SelectItem>
                      <SelectItem value="false">− (Negative)</SelectItem>
                    </SelectContent>
                  </Select>
                )}
              />
            </div>
            <FieldError message={errors.blood_type?.message ?? errors.blood_rh?.message} />
          </div>

          <div>
            <Label>Blood Pressure (mmHg)</Label>
            <div className="flex items-center gap-2 mt-1">
              <Input
                type="number"
                placeholder="Systolic"
                {...register("blood_pressure_systolic", { valueAsNumber: true })}
                className="w-28"
              />
              <span className="text-muted-foreground">/</span>
              <Input
                type="number"
                placeholder="Diastolic"
                {...register("blood_pressure_diastolic", { valueAsNumber: true })}
                className="w-28"
              />
            </div>
            <FieldError message={errors.blood_pressure_systolic?.message ?? errors.blood_pressure_diastolic?.message} />
          </div>

          <div>
            <Label htmlFor="weight">Weight (kg)</Label>
            <Input
              id="weight"
              type="number"
              step="0.1"
              placeholder="70"
              className="mt-1"
              {...register("weight", { valueAsNumber: true })}
            />
            <FieldError message={errors.weight?.message} />
          </div>

          <div>
            <Label htmlFor="height">Height (cm)</Label>
            <Input
              id="height"
              type="number"
              placeholder="170"
              className="mt-1"
              {...register("height", { valueAsNumber: true })}
            />
            <FieldError message={errors.height?.message} />
          </div>

          <div>
            <Label htmlFor="pulse">Pulse (bpm)</Label>
            <Input
              id="pulse"
              type="number"
              placeholder="80"
              className="mt-1"
              {...register("pulse", { valueAsNumber: true })}
            />
            <FieldError message={errors.pulse?.message} />
          </div>

          <div>
            <Label htmlFor="allergies">Allergies</Label>
            <Input
              id="allergies"
              placeholder="Penicillin, latex (comma-separated)"
              className="mt-1"
              {...register("allergies")}
            />
          </div>

          <div>
            <Label htmlFor="chronic_conditions">Chronic Conditions</Label>
            <Input
              id="chronic_conditions"
              placeholder="Diabetes, hypertension (comma-separated)"
              className="mt-1"
              {...register("chronic_conditions")}
            />
          </div>
        </div>
      </section>

      {/* Triage step */}
      <section className="space-y-3">
        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">Triage — Symptoms</h3>
        <div>
          <Label htmlFor="chief_complaint">Main complaint / Symptoms</Label>
          <Textarea
            id="chief_complaint"
            placeholder="Describe the patient's symptoms…"
            rows={3}
            className="mt-1"
            {...register("chief_complaint")}
          />
          <FieldError message={errors.chief_complaint?.message} />
        </div>
      </section>

      {/* Attachments */}
      <section className="space-y-3">
        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">Attachments (optional)</h3>

        {existingAttachments && existingAttachments.length > 0 && (
          <ul className="space-y-1.5">
            {existingAttachments.map((att) => (
              <li key={att.id} className="flex items-center gap-2 rounded-md border bg-muted/40 px-3 py-2 text-sm">
                {kindIcon(att.kind)}
                <span className="flex-1 truncate">{att.original_filename}</span>
                <span className="text-xs text-muted-foreground capitalize">{att.status}</span>
                {onRemoveExistingAttachment && (
                  <button
                    type="button"
                    onClick={() => onRemoveExistingAttachment(att)}
                    className="ml-1 rounded-sm opacity-60 hover:opacity-100"
                    aria-label="Remove attachment"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}

        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            onDragOver();
          }}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          className={`flex w-full items-center justify-center gap-2 rounded-md border-2 border-dashed py-4 text-sm transition-colors ${dragging ? "border-primary bg-primary/5 text-primary" : "border-border text-muted-foreground hover:border-primary hover:text-primary"}`}
        >
          <Upload className="h-4 w-4" />
          {dragging ? "Drop files here" : "Upload or drag files (JPG, PNG, GIF, PDF, WAV, MP3 · max 5 MB each)"}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".jpg,.jpeg,.png,.gif,.pdf,.wav,.mp3,audio/mpeg,audio/wav,image/jpeg,image/png,image/gif,application/pdf"
          className="hidden"
          onChange={onFileChange}
        />

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
                  onClick={() => onRemoveFile(i)}
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

      <div className="flex justify-end gap-2 pt-2">
        <Button type="button" variant="outline" onClick={onCancel} disabled={submitting}>
          Cancel
        </Button>
        <Button type="submit" disabled={submitting}>
          {submitLabel}
        </Button>
      </div>
    </form>
  );
}
