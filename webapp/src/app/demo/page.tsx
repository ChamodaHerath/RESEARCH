"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ScanSearch, FileSearch, ClipboardCheck } from "lucide-react";
import { SectionBadge } from "@/components/ui/glass-card";
import { UploadPanel, type UploadStatus } from "@/components/demo/upload-panel";
import { TerminalConsole } from "@/components/demo/terminal-console";
import { ResultsPanel } from "@/components/demo/results-panel";
import { ForensicDeepDive } from "@/components/demo/forensic-deep-dive";
import { ExplanationEvaluationPanel } from "@/components/demo/explanation-evaluation-panel";
import { cn } from "@/lib/utils";
import type { DetectionMode, DetectionResponse } from "@/lib/types";

type DemoPanel = "detection" | "forensic" | "evaluation";

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
  const fakePanelsAvailable = Boolean(result && isFake);

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

  React.useEffect(() => {
    if (!fakePanelsAvailable && activePanel !== "detection") {
      setActivePanel("detection");
    }
  }, [fakePanelsAvailable, activePanel]);

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
              Panel 1: detection · Panel 2: why Fake forensic report · Panel 3:
              per-explanation accuracy evaluation.
            </p>
          </div>

          <div
            className={cn(
              "inline-flex shrink-0 flex-wrap rounded-xl border border-white/10 bg-white/[0.03] p-1",
              !fakePanelsAvailable && "opacity-90"
            )}
            role="tablist"
            aria-label="Demo panels"
          >
            <PanelTab
              active={activePanel === "detection"}
              onClick={() => setActivePanel("detection")}
              icon={<ScanSearch className="h-3.5 w-3.5" />}
              label="Panel 1 · Detection"
              activeClass="bg-gradient-to-r from-primary to-secondary text-white shadow"
            />
            <PanelTab
              active={activePanel === "forensic"}
              disabled={!fakePanelsAvailable}
              onClick={() => fakePanelsAvailable && setActivePanel("forensic")}
              icon={<FileSearch className="h-3.5 w-3.5" />}
              label="Panel 2 · Why Fake"
              activeClass="bg-gradient-to-r from-red-600 to-rose-500 text-white shadow"
              disabledHint="Fake only"
            />
            <PanelTab
              active={activePanel === "evaluation"}
              disabled={!fakePanelsAvailable}
              onClick={() => fakePanelsAvailable && setActivePanel("evaluation")}
              icon={<ClipboardCheck className="h-3.5 w-3.5" />}
              label="Panel 3 · Evaluation"
              activeClass="bg-gradient-to-r from-violet-600 to-purple-500 text-white shadow"
              disabledHint="Fake only"
            />
          </div>
        </div>
      </motion.div>

      <AnimatePresence mode="wait">
        {activePanel === "detection" && (
          <motion.div
            key="panel-detection"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25 }}
            className="flex flex-col gap-4"
          >
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

            <ResultsPanel result={result} isProcessing={status === "running"} />

            {result && result.prediction === "Fake" && (
              <p className="text-center text-xs text-slate-500">
                Fake verdict ready — open{" "}
                <button
                  type="button"
                  onClick={() => setActivePanel("forensic")}
                  className="font-medium text-red-400 underline-offset-2 hover:underline"
                >
                  Panel 2
                </button>{" "}
                for the forensic report or{" "}
                <button
                  type="button"
                  onClick={() => setActivePanel("evaluation")}
                  className="font-medium text-violet-400 underline-offset-2 hover:underline"
                >
                  Panel 3
                </button>{" "}
                to verify each explanation.
              </p>
            )}
          </motion.div>
        )}

        {activePanel === "forensic" && result && result.prediction === "Fake" && (
          <motion.div
            key="panel-forensic"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25 }}
          >
            <ForensicDeepDive result={result} />
          </motion.div>
        )}

        {activePanel === "evaluation" && result && result.prediction === "Fake" && (
          <motion.div
            key="panel-evaluation"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25 }}
          >
            <ExplanationEvaluationPanel result={result} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function PanelTab({
  active,
  disabled,
  onClick,
  icon,
  label,
  activeClass,
  disabledHint,
}: {
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  activeClass: string;
  disabledHint?: string;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
        active
          ? activeClass
          : disabled
            ? "cursor-not-allowed text-slate-600"
            : "text-slate-400 hover:text-slate-200"
      )}
    >
      {icon}
      <span className="hidden sm:inline">{label}</span>
      <span className="sm:hidden">{label.split("·")[0]?.trim()}</span>
      {disabled && disabledHint && (
        <span className="hidden text-[10px] font-normal uppercase tracking-wide lg:inline">
          ({disabledHint})
        </span>
      )}
    </button>
  );
}
