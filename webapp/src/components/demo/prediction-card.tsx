"use client";

import { motion } from "framer-motion";
import { ShieldAlert, ShieldCheck, AlertTriangle } from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import { ConfidenceGauge } from "./confidence-gauge";
import type { ConfidenceBreakdown } from "@/lib/types";

export function PredictionCard({
  prediction,
  confidence,
  riskLevel,
  processingTime,
  breakdown,
}: {
  prediction: "Real" | "Fake";
  confidence: number;
  riskLevel: string;
  processingTime: string;
  breakdown: ConfidenceBreakdown[];
}) {
  const isFake = prediction === "Fake";
  const color = isFake ? "#ef4444" : "#22c55e";

  return (
    <GlassCard glow className="flex flex-col gap-6 p-6">
      <div className="flex items-center justify-between">
        <h3 className="font-display text-lg font-semibold text-white">Detection Result</h3>
        <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[11px] font-medium text-slate-400">
          {processingTime}
        </span>
      </div>

      <div className="flex flex-col items-center gap-5 sm:flex-row sm:justify-between">
        <motion.div
          initial={{ opacity: 0, scale: 0.85 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="flex flex-col items-center gap-3"
        >
          <div
            className="flex items-center gap-2 rounded-2xl px-6 py-3 text-2xl font-extrabold tracking-wide"
            style={{
              color,
              background: isFake ? "rgba(239,68,68,0.12)" : "rgba(34,197,94,0.12)",
              boxShadow: `0 0 30px -6px ${color}66, inset 0 0 0 1px ${color}44`,
            }}
          >
            {isFake ? <ShieldAlert className="h-7 w-7" /> : <ShieldCheck className="h-7 w-7" />}
            {prediction.toUpperCase()}
          </div>
          <div className="flex items-center gap-1.5 text-xs text-slate-400">
            <AlertTriangle className="h-3.5 w-3.5" style={{ color }} />
            Risk Level:{" "}
            <span className="font-semibold" style={{ color }}>
              {riskLevel}
            </span>
          </div>
        </motion.div>

        <ConfidenceGauge value={confidence} color={color} />
      </div>

      <div className="flex flex-col gap-3 border-t border-white/10 pt-5">
        {breakdown.map((item, i) => (
          <div key={item.metric} className="flex items-center gap-3">
            <span className="w-40 shrink-0 truncate text-xs text-slate-400">{item.metric}</span>
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/[0.06]">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${item.value}%` }}
                transition={{ duration: 0.9, delay: i * 0.08, ease: "easeOut" }}
                className="h-full rounded-full bg-gradient-to-r from-primary to-secondary"
              />
            </div>
            <span className="w-9 shrink-0 text-right text-xs font-medium text-slate-300">
              {item.value}%
            </span>
          </div>
        ))}
      </div>
    </GlassCard>
  );
}
