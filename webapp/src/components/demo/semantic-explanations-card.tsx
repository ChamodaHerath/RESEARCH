"use client";

import { motion } from "framer-motion";
import {
  MessageSquareText,
  AlertCircle,
  Info,
  CheckCircle2,
} from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import type { SemanticExplanation } from "@/lib/types";

const SEVERITY_CONFIG = {
  high: { color: "#ef4444", icon: AlertCircle, label: "High" },
  medium: { color: "#f59e0b", icon: Info, label: "Medium" },
  low: { color: "#22c55e", icon: CheckCircle2, label: "Low" },
};

export function SemanticExplanationsCard({
  explanations,
}: {
  explanations: SemanticExplanation[];
}) {
  if (explanations.length === 0) {
    return (
      <GlassCard className="flex flex-col gap-4 p-6">
        <h3 className="flex items-center gap-2 font-display text-lg font-semibold text-white">
          <MessageSquareText className="h-5 w-5 text-primary" />
          Semantic Explanations
        </h3>
        <p className="text-sm leading-relaxed text-slate-400">
          No zone exceeded its calibrated threshold on this Fake verdict, so no
          natural-language claims were emitted.
        </p>
      </GlassCard>
    );
  }

  return (
    <GlassCard className="flex flex-col gap-5 p-6">
      <div className="flex items-center justify-between">
        <h3 className="flex items-center gap-2 font-display text-lg font-semibold text-white">
          <MessageSquareText className="h-5 w-5 text-primary" />
          Semantic Explanations
        </h3>
        <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[11px] font-medium text-slate-400">
          {explanations.length} findings
        </span>
      </div>

      <div className="flex flex-col gap-3">
        {explanations.map((exp, i) => {
          const config = SEVERITY_CONFIG[exp.severity];
          const Icon = config.icon;

          return (
            <motion.div
              key={exp.id}
              initial={{ opacity: 0, x: -16 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4, delay: i * 0.08 }}
              className="rounded-xl border border-white/10 bg-white/[0.03] p-4"
            >
              <div className="flex gap-3">
                <div
                  className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg"
                  style={{ background: `${config.color}22`, color: config.color }}
                >
                  <Icon className="h-4 w-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <h4 className="text-sm font-semibold text-slate-100">
                      {exp.title}
                    </h4>
                    <span
                      className="rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide"
                      style={{ background: `${config.color}1a`, color: config.color }}
                    >
                      {config.label}
                    </span>
                  </div>
                  <p className="mt-1.5 text-xs leading-relaxed text-slate-300">
                    {exp.description}
                  </p>
                  <div className="mt-2 flex items-center gap-2">
                    <div className="h-1 flex-1 overflow-hidden rounded-full bg-white/[0.06]">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${exp.score * 100}%` }}
                        transition={{ duration: 0.8, delay: i * 0.08 + 0.2 }}
                        className="h-full rounded-full"
                        style={{ background: config.color }}
                      />
                    </div>
                    <span className="text-[10px] font-medium text-slate-500">
                      {(exp.score * 100).toFixed(0)}% match
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>
    </GlassCard>
  );
}
