"use client";

import * as React from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import {
  UploadCloud,
  FileImage,
  FileVideo,
  X,
  Play,
  RotateCcw,
  ImageIcon,
  Video,
  Loader2,
  AlertTriangle,
} from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { DetectionMode, ModelOption } from "@/lib/types";

const ACCEPTED = {
  "image/jpeg": [".jpg", ".jpeg"],
  "image/png": [".png"],
  "video/mp4": [".mp4"],
  "video/x-msvideo": [".avi"],
  "video/quicktime": [".mov"],
};

// Kept in sync manually with attention_extractor.DEFAULT_HF_DEEPFAKE_MODEL.
// The actual model name used for a given request is echoed back in the real
// /infer/* response body (`model` field) -- this is only a display default
// shown before the first run.
const MODEL_OPTIONS: { value: ModelOption; label: string }[] = [
  {
    value: "hamzenium/ViT-Deepfake-Classifier",
    label: "ViT-Base/16 — hamzenium/ViT-Deepfake-Classifier",
  },
];

export type UploadStatus = "idle" | "running" | "done" | "error";

export function UploadPanel({
  file,
  previewUrl,
  mode,
  status,
  errorMessage,
  onFileSelect,
  onModeChange,
  onRun,
  onReset,
}: {
  file: File | null;
  previewUrl: string | null;
  mode: DetectionMode;
  status: UploadStatus;
  errorMessage?: string | null;
  onFileSelect: (file: File) => void;
  onModeChange: (m: DetectionMode) => void;
  onRun: () => void;
  onReset: () => void;
}) {
  const isVideo = mode === "video";

  const onDrop = React.useCallback(
    (accepted: File[]) => {
      if (accepted[0]) {
        onFileSelect(accepted[0]);
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    multiple: false,
    disabled: status === "running",
  });

  return (
    <GlassCard className="flex h-full flex-col gap-6 p-6">
      <div className="flex items-center justify-between">
        <h3 className="font-display text-lg font-semibold text-white">Input Media</h3>
        <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[11px] font-medium uppercase tracking-wider text-slate-400">
          Step 01
        </span>
      </div>

      {!previewUrl ? (
        <div
          {...getRootProps()}
          className={cn(
            "group relative flex min-h-[220px] flex-1 cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-white/15 bg-white/[0.02] p-8 text-center transition-all duration-300",
            isDragActive && "border-primary/70 bg-primary/5",
            status === "running" && "pointer-events-none opacity-50"
          )}
        >
          <input {...getInputProps()} />
          <motion.div
            animate={{ y: isDragActive ? -6 : 0 }}
            className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/20 to-secondary/20 text-blue-300 ring-1 ring-white/10"
          >
            <UploadCloud className="h-8 w-8" />
          </motion.div>
          <p className="font-medium text-slate-200">
            {isDragActive ? "Drop the file here" : "Drag & drop your media, or click to browse"}
          </p>
          <p className="text-xs text-slate-500">
            Supports JPG, JPEG, PNG, MP4, AVI, MOV
          </p>
          <div className="mt-2 flex items-center gap-4 text-slate-500">
            <div className="flex items-center gap-1.5 text-xs">
              <FileImage className="h-3.5 w-3.5" /> Images
            </div>
            <div className="flex items-center gap-1.5 text-xs">
              <FileVideo className="h-3.5 w-3.5" /> Videos
            </div>
          </div>
        </div>
      ) : (
        <div className="relative flex-1 overflow-hidden rounded-2xl border border-white/10 bg-black/40">
          <button
            onClick={onReset}
            disabled={status === "running"}
            className="absolute right-3 top-3 z-10 flex h-8 w-8 items-center justify-center rounded-full bg-black/60 text-slate-200 backdrop-blur transition-colors hover:bg-black/80 disabled:opacity-40"
          >
            <X className="h-4 w-4" />
          </button>
          {isVideo ? (
            <video
              src={previewUrl}
              controls
              className="h-full max-h-[280px] w-full object-contain"
            />
          ) : (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={previewUrl}
              alt="Uploaded preview"
              className="h-full max-h-[280px] w-full object-contain"
            />
          )}
          <div className="flex items-center gap-2 border-t border-white/10 bg-white/[0.03] px-4 py-2 text-xs text-slate-400">
            {isVideo ? <FileVideo className="h-3.5 w-3.5" /> : <FileImage className="h-3.5 w-3.5" />}
            <span className="truncate">{file?.name}</span>
            {file && (
              <span className="ml-auto shrink-0 text-slate-500">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </span>
            )}
          </div>
        </div>
      )}

      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Detection Model <span className="text-emerald-400">(live, calibrated)</span>
          </label>
          <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2.5 text-sm text-slate-200">
            <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-400" />
            <span className="truncate">{MODEL_OPTIONS[0].label}</span>
          </div>
        </div>

        <div className="flex flex-col gap-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Detection Mode
          </label>
          <div className="grid grid-cols-2 gap-2 rounded-xl border border-white/10 bg-white/[0.03] p-1">
            {(
              [
                { value: "image" as DetectionMode, label: "Image", icon: ImageIcon },
                { value: "video" as DetectionMode, label: "Video", icon: Video },
              ]
            ).map((opt) => {
              const Icon = opt.icon;
              const active = mode === opt.value;
              return (
                <button
                  key={opt.value}
                  onClick={() => onModeChange(opt.value)}
                  disabled={status === "running"}
                  className={cn(
                    "relative flex items-center justify-center gap-2 rounded-lg py-2.5 text-sm font-medium transition-colors",
                    active ? "text-white" : "text-slate-400 hover:text-slate-200"
                  )}
                >
                  {active && (
                    <motion.span
                      layoutId="mode-active"
                      className="absolute inset-0 rounded-lg bg-gradient-to-r from-primary to-secondary"
                      transition={{ type: "spring", duration: 0.4 }}
                    />
                  )}
                  <span className="relative z-10 flex items-center gap-1.5">
                    <Icon className="h-3.5 w-3.5" /> {opt.label}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {status === "error" && errorMessage && (
        <div className="flex items-start gap-2 rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-300">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      <div className="flex items-center gap-3 pt-1">
        <Button
          className="flex-1"
          size="lg"
          onClick={onRun}
          disabled={!file || status === "running"}
        >
          <AnimatePresence mode="wait" initial={false}>
            {status === "running" ? (
              <motion.span
                key="running"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center gap-2"
              >
                <Loader2 className="h-4 w-4 animate-spin" /> Processing...
              </motion.span>
            ) : (
              <motion.span
                key="run"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center gap-2"
              >
                <Play className="h-4 w-4" /> Run Detection
              </motion.span>
            )}
          </AnimatePresence>
        </Button>
        <Button variant="outline" size="lg" onClick={onReset} disabled={status === "running"}>
          <RotateCcw className="h-4 w-4" />
        </Button>
      </div>
    </GlassCard>
  );
}
