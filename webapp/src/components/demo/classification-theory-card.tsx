"use client";

import { motion } from "framer-motion";
import { BookOpen, FlaskConical } from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import type { ClassificationReasoning } from "@/lib/types";

export function ClassificationTheoryCard({
  reasoning,
  prediction,
}: {
  reasoning: ClassificationReasoning;
  prediction: "Real" | "Fake";
}) {
  const accent = prediction === "Fake" ? "#ef4444" : "#22c55e";

  return (
    <GlassCard className="flex flex-col gap-5 p-6">
      <div className="flex items-start justify-between gap-3">
        <h3 className="flex items-center gap-2 font-display text-lg font-semibold text-white">
          <BookOpen className="h-5 w-5 text-primary" />
          Why {prediction}?
        </h3>
        <span
          className="shrink-0 rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide"
          style={{ borderColor: `${accent}44`, color: accent, background: `${accent}14` }}
        >
          Theoretical basis
        </span>
      </div>

      <p className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm leading-relaxed text-slate-300">
        {reasoning.summary}
      </p>

      <div className="flex flex-col gap-3">
        {reasoning.steps.map((step, i) => (
          <motion.div
            key={step.stage}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: i * 0.06 }}
            className="rounded-xl border border-white/10 bg-white/[0.02] p-4"
          >
            <div className="flex items-start gap-3">
              <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary">
                <FlaskConical className="h-3.5 w-3.5" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                  <h4 className="text-sm font-semibold text-slate-100">{step.stage}</h4>
                  <span className="rounded-md bg-white/[0.06] px-2 py-0.5 font-mono text-[11px] text-cyan-300">
                    {step.value}
                  </span>
                </div>
                {step.formula && (
                  <p className="mt-1.5 font-mono text-[11px] text-slate-500">{step.formula}</p>
                )}
                <p className="mt-2 text-xs leading-relaxed text-slate-400">{step.interpretation}</p>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </GlassCard>
  );
}
