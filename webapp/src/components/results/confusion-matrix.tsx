"use client";

import { GlassCard } from "@/components/ui/glass-card";
import { DATASETS, deriveConfusionMatrix } from "@/lib/real-eval";
import { useState } from "react";
import { cn } from "@/lib/utils";

const CELL_META = [
  { key: "tp" as const, label: "True Positive", sub: "Correctly detected Fake", color: "#22c55e" },
  { key: "fp" as const, label: "False Positive", sub: "Real misclassified as Fake", color: "#f59e0b" },
  { key: "fn" as const, label: "False Negative", sub: "Fake misclassified as Real", color: "#ef4444" },
  { key: "tn" as const, label: "True Negative", sub: "Correctly detected Real", color: "#3b82f6" },
];

export function ConfusionMatrix() {
  const [datasetIdx, setDatasetIdx] = useState(0);
  const dataset = DATASETS[datasetIdx];
  const matrix = deriveConfusionMatrix(dataset.eval.classification_metrics);
  const total =
    dataset.eval.classification_metrics.n_fake + dataset.eval.classification_metrics.n_real;

  return (
    <GlassCard className="flex flex-col gap-5 p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="font-display text-base font-semibold text-white">
          Confusion Matrix — real evaluation run
        </h3>
        <div className="flex gap-1.5 rounded-lg border border-white/10 bg-white/[0.03] p-1">
          {DATASETS.map((d, i) => (
            <button
              key={d.key}
              onClick={() => setDatasetIdx(i)}
              className={cn(
                "rounded-md px-3 py-1 text-xs font-medium transition-colors",
                i === datasetIdx ? "bg-primary text-white" : "text-slate-400 hover:text-slate-200"
              )}
            >
              {d.name}
            </button>
          ))}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        {CELL_META.map((cell) => {
          const value = matrix[cell.key];
          return (
            <div
              key={cell.label}
              className="flex flex-col gap-1 rounded-xl border border-white/10 p-4"
              style={{ background: `${cell.color}14` }}
            >
              <span className="font-display text-2xl font-bold" style={{ color: cell.color }}>
                {value.toLocaleString()}
              </span>
              <span className="text-xs font-semibold text-slate-200">{cell.label}</span>
              <span className="text-[11px] text-slate-500">{cell.sub}</span>
              <span className="text-[11px] font-medium text-slate-400">
                {((value / total) * 100).toFixed(1)}% of total
              </span>
            </div>
          );
        })}
      </div>
      <p className="text-xs text-slate-500">
        Evaluated on the real {dataset.name} test partition ({total.toLocaleString()} labeled
        frames). Counts are the pipeline&apos;s own measured TP/FP/TN/FN from
        evaluate_pipeline.py — not independently fabricated or derived.
      </p>
    </GlassCard>
  );
}
