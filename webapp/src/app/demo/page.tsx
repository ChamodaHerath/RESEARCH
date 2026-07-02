"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { SectionBadge } from "@/components/ui/glass-card";
import { UploadPanel, type UploadStatus } from "@/components/demo/upload-panel";
import { TerminalConsole } from "@/components/demo/terminal-console";
import { ResultsPanel } from "@/components/demo/results-panel";
import { ForensicReport } from "@/components/demo/forensic-report";
import type { DetectionMode, DetectionResponse } from "@/lib/types";

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

  const handleFileSelect = (selected: File) => {
    setFile(selected);
    setResult(null);
    setPendingResult(null);
    setLogs([]);
    setStatus("idle");
    setErrorMessage(null);
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
  };

  const handleRun = async () => {
    if (!file) return;
    setStatus("running");
    setResult(null);
    setPendingResult(null);
    setErrorMessage(null);
    setLogs([]);

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

  return (
    <div className="section-pad container-app flex flex-col gap-10">
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="flex flex-col items-center gap-4 text-center"
      >
        <SectionBadge>Live Forensic Demonstration</SectionBadge>
        <h1 className="font-display text-3xl font-bold text-white sm:text-4xl">
          Explainable Deepfake Detection Console
        </h1>
        <p className="max-w-2xl text-sm text-slate-400 sm:text-base">
          Upload an image or video to run it through the full Vision Transformer forensic
          pipeline. Watch the live processing console, then review the prediction, semantic
          explanations, attention heatmap, and forensic report.
        </p>
      </motion.div>

      <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-[380px_1fr_420px]">
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

        <TerminalConsole key={runId} logs={logs} runId={runId} onComplete={handleTerminalComplete} />

        <ResultsPanel result={result} isProcessing={status === "running"} />
      </div>

      {result && <ForensicReport result={result} />}
    </div>
  );
}
