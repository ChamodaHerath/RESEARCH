"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ScanSearch, FileSearch } from "lucide-react";
import { SectionBadge } from "@/components/ui/glass-card";
import { UploadPanel, type UploadStatus } from "@/components/demo/upload-panel";
import { TerminalConsole } from "@/components/demo/terminal-console";
import { ResultsPanel } from "@/components/demo/results-panel";
import { ForensicDeepDive } from "@/components/demo/forensic-deep-dive";
import { cn } from "@/lib/utils";
import type { DetectionMode, DetectionResponse } from "@/lib/types";

type DemoPanel = "detection" | "forensic";

export default function DemoPage() {
  const [file, setFile] = React.useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = React.useState<string | null>(null);
  const [mode, setMode] = React.useState<DetectionMode>("image");
  const [status, setStatus] = React.useState<UploadStatus>("idle");
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null);
  const [logs, setLogs] = React.useState<string[]>([]);
  const [result, setResult] = React.useState<DetectionResponse | null>(null);
  const [pendingResult, setPendingResult] = React.useState<DetectionResponse | null>(null);
  const [runId, setRunId] = React.useState(0);
  const [activePanel, setActivePanel] = React.useState<DemoPanel>("detection");

  const isFake = result?.prediction === "Fake";
  const forensicAvailable = Boolean(result && isFake);

  const handleFileSelect = (selected: File) => {
    setFile(selected);
    setResult(null);
    setPendingResult(null);
    setLogs([]);
    setStatus("idle");
    setErrorMessage(null);
    setActivePanel("detection");
    const url = URL.createObjectURL(selected);
    setPreviewUrl(url);
    if (selected.type.startsWith("video/")) {
      setMode("video");
    } else if (selected.type.startsWith("image/")) {
      setMode("image");
    }
  };

  const handleReset = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setPendingResult(null);
    setLogs([]);
    setStatus("idle");
    setErrorMessage(null);
    setActivePanel("detection");
  };

  const handleRun = async () => {
    if (!file) return;
    setStatus("running");
    setResult(null);
    setPendingResult(null);
    setErrorMessage(null);
    setLogs([]);
    setActivePanel("detection");

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("mode", mode);

      const res = await fetch("/api/detect", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok || data?.error) {
        setStatus("error");
        setErrorMessage(
          data?.error ?? "The real detection backend returned an error for this file."
        );
        return;
      }

      setPendingResult(data as DetectionResponse);
      setLogs((data as DetectionResponse).terminal_logs);
      setRunId((id) => id + 1);
    } catch {
      setStatus("error");
      setErrorMessage(
        "Could not reach the detection backend. Make sure inference_server.py is running."
      );
    }
  };

  const handleTerminalComplete = React.useCallback(() => {
    setResult(pendingResult);
    setStatus("done");
  }, [pendingResult]);

  // If result flips away from Fake, force Panel 1
  React.useEffect(() => {
    if (!forensicAvailable && activePanel === "forensic") {
      setActivePanel("detection");
    }
  }, [forensicAvailable, activePanel]);

  return (
    <div className="container-app flex flex-col gap-5 pb-10 pt-6 sm:pt-8">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="flex flex-col gap-3"
      >
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div className="flex flex-col gap-1.5">
            <SectionBadge>Live Forensic Demonstration</SectionBadge>
            <h1 className="font-display text-2xl font-bold text-white sm:text-3xl">
              Explainable Deepfake Detection Console
            </h1>
            <p className="max-w-2xl text-sm text-slate-400">
              Panel 1 runs the live pipeline. When the verdict is Fake, switch to Panel 2 for the
              full why-Fake forensic report.
            </p>
          </div>

          {/* Panel switch — Panel 2 only when Fake */}
          <div
            className={cn(
              "inline-flex shrink-0 rounded-xl border border-white/10 bg-white/[0.03] p-1",
              !forensicAvailable && "opacity-90"
            )}
            role="tablist"
            aria-label="Demo panels"
          >
            <button
              type="button"
              role="tab"
              aria-selected={activePanel === "detection"}
              onClick={() => setActivePanel("detection")}
              className={cn(
                "flex items-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors",
                activePanel === "detection"
                  ? "bg-gradient-to-r from-primary to-secondary text-white shadow"
                  : "text-slate-400 hover:text-slate-200"
              )}
            >
              <ScanSearch className="h-3.5 w-3.5" />
              Panel 1 · Detection
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activePanel === "forensic"}
              disabled={!forensicAvailable}
              title={
                forensicAvailable
                  ? "Full Fake forensic deep-dive"
                  : "Available only when classification is Fake"
              }
              onClick={() => forensicAvailable && setActivePanel("forensic")}
              className={cn(
                "flex items-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors",
                activePanel === "forensic"
                  ? "bg-gradient-to-r from-red-600 to-rose-500 text-white shadow"
                  : forensicAvailable
                    ? "text-slate-400 hover:text-slate-200"
                    : "cursor-not-allowed text-slate-600"
              )}
            >
              <FileSearch className="h-3.5 w-3.5" />
              Panel 2 · Why Fake
              {!forensicAvailable && (
                <span className="hidden text-[10px] font-normal uppercase tracking-wide sm:inline">
                  (Fake only)
                </span>
              )}
            </button>
          </div>
        </div>
      </motion.div>

      <AnimatePresence mode="wait">
        {activePanel === "detection" ? (
          <motion.div
            key="panel-detection"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25 }}
            className="flex flex-col gap-4"
          >
            {/* Top row: upload + terminal — compact, no tall empty columns */}
            <div className="grid grid-cols-1 items-stretch gap-4 lg:grid-cols-[340px_1fr]">
              <UploadPanel
                file={file}
                previewUrl={previewUrl}
                mode={mode}
                status={status}
                errorMessage={errorMessage}
                onFileSelect={handleFileSelect}
                onModeChange={setMode}
                onRun={handleRun}
                onReset={handleReset}
              />
              <TerminalConsole
                key={runId}
                logs={logs}
                runId={runId}
                onComplete={handleTerminalComplete}
              />
            </div>

            {/* Results span full width in a denser card grid */}
            <ResultsPanel result={result} isProcessing={status === "running"} />

            {result && result.prediction === "Fake" && (
              <p className="text-center text-xs text-slate-500">
                Fake verdict ready — use{" "}
                <button
                  type="button"
                  onClick={() => setActivePanel("forensic")}
                  className="font-medium text-red-400 underline-offset-2 hover:underline"
                >
                  Panel 2 · Why Fake
                </button>{" "}
                for the full forensic report.
              </p>
            )}
          </motion.div>
        ) : (
          result &&
          result.prediction === "Fake" && (
            <motion.div
              key="panel-forensic"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
            >
              <ForensicDeepDive result={result} />
            </motion.div>
          )
        )}
      </AnimatePresence>
    </div>
  );
}
