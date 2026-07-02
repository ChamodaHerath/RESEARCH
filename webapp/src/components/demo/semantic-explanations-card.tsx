"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageSquareText,
  AlertCircle,
  Info,
  CheckCircle2,
  ChevronDown,
  Lightbulb,
} from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import { cn } from "@/lib/utils";
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
  const [expandedId, setExpandedId] = React.useState<string | null>(
    explanations[0]?.id ?? null
  );
  // Reset the expanded card whenever a new set of explanations arrives (e.g.
  // a new detection result). Adjusting state during render -- rather than in
  // a useEffect -- avoids the extra cascading render React warns about for
  // "reset state when a prop changes" (see react.dev/learn/you-might-not-need-an-effect).
  const [prevExplanations, setPrevExplanations] = React.useState(explanations);
  if (explanations !== prevExplanations) {
    setPrevExplanations(explanations);
    setExpandedId(explanations[0]?.id ?? null);
  }

  if (explanations.length === 0) {
    return (
      <GlassCard className="flex flex-col gap-4 p-6">
        <h3 className="flex items-center gap-2 font-display text-lg font-semibold text-white">
          <MessageSquareText className="h-5 w-5 text-primary" />
          Semantic Explanations
        </h3>
        <p className="text-sm leading-relaxed text-slate-400">
          No zones crossed the calibrated anomaly threshold for this frame, so the
          explainability layer did not emit zone-level concept vectors. The Real/Fake
          verdict above comes from the ViT classifier probability alone.
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
          const isOpen = expandedId === exp.id;

          return (
            <motion.div
              key={exp.id}
              initial={{ opacity: 0, x: -16 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4, delay: i * 0.08 }}
              className="overflow-hidden rounded-xl border border-white/10 bg-white/[0.03]"
            >
              <button
                type="button"
                onClick={() => setExpandedId(isOpen ? null : exp.id)}
                className="flex w-full gap-3 p-4 text-left transition-colors hover:bg-white/[0.02]"
              >
                <div
                  className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg"
                  style={{ background: `${config.color}22`, color: config.color }}
                >
                  <Icon className="h-4 w-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <h4 className="text-sm font-semibold text-slate-100">{exp.title}</h4>
                    <div className="flex items-center gap-2">
                      <span
                        className={cn(
                          "shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide"
                        )}
                        style={{ background: `${config.color}1a`, color: config.color }}
                      >
                        {config.label}
                      </span>
                      <ChevronDown
                        className={cn(
                          "h-4 w-4 shrink-0 text-slate-500 transition-transform",
                          isOpen && "rotate-180"
                        )}
                      />
                    </div>
                  </div>
                  <p className="mt-1 text-xs leading-relaxed text-slate-400">{exp.description}</p>
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
              </button>

              <AnimatePresence initial={false}>
                {isOpen && exp.theory && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.25 }}
                    className="overflow-hidden"
                  >
                    <div className="border-t border-white/10 bg-black/20 px-4 pb-4 pt-3">
                      <div className="mb-3 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-cyan-400/90">
                        <Lightbulb className="h-3.5 w-3.5" />
                        Why this explanation?
                      </div>
                      <p className="mb-3 text-xs leading-relaxed text-slate-300">
                        {exp.theory.summary}
                      </p>
                      <div className="mb-3 flex flex-wrap gap-2">
                        <ScoreChip label="R(z)" value={exp.theory.scores.R} />
                        {exp.theory.scores.Q !== undefined && (
                          <ScoreChip label="Q(z)" value={exp.theory.scores.Q} />
                        )}
                      </div>
                      <ol className="flex flex-col gap-2.5">
                        {exp.theory.steps.map((step, stepIdx) => (
                          <li
                            key={step.label}
                            className="rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2.5"
                          >
                            <p className="text-[11px] font-semibold text-slate-200">
                              {stepIdx + 1}. {step.label}
                            </p>
                            <p className="mt-1 text-[11px] leading-relaxed text-slate-400">
                              {step.detail}
                            </p>
                          </li>
                        ))}
                      </ol>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>
    </GlassCard>
  );
}

function ScoreChip({ label, value }: { label: string; value: number }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-white/[0.04] px-2 py-1 font-mono text-[10px] text-slate-300">
      <span className="text-slate-500">{label}</span>
      {value.toFixed(4)}
    </span>
  );
}
