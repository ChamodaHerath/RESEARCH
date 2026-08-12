"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { ScanFace } from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import type { FacialRegionScore } from "@/lib/types";

function scoreColor(score: number) {
  if (score > 0.65) return "#ef4444";
  if (score > 0.35) return "#f59e0b";
  return "#22c55e";
}

function corruptionLabel(score: number) {
  if (score > 0.65) return "High corruption";
  if (score > 0.35) return "Moderate";
  return "Low";
}

/** Approximate face-map positions for the 9 SAEDF zones. */
const ZONE_SHAPES: Record<string, { cx: number; cy: number; rx: number; ry: number }> = {
  forehead: { cx: 200, cy: 105, rx: 78, ry: 36 },
  left_eye: { cx: 155, cy: 175, rx: 32, ry: 18 },
  right_eye: { cx: 245, cy: 175, rx: 32, ry: 18 },
  nose_bridge: { cx: 200, cy: 205, rx: 18, ry: 28 },
  nose_tip: { cx: 200, cy: 245, rx: 24, ry: 18 },
  left_cheek: { cx: 130, cy: 240, rx: 34, ry: 36 },
  right_cheek: { cx: 270, cy: 240, rx: 34, ry: 36 },
  mouth: { cx: 200, cy: 290, rx: 46, ry: 20 },
  chin_jaw: { cx: 200, cy: 335, rx: 88, ry: 32 },
  // Legacy 5-region aggregates (older API responses)
  eyes: { cx: 200, cy: 175, rx: 88, ry: 22 },
  nose: { cx: 200, cy: 228, rx: 26, ry: 40 },
  jaw: { cx: 200, cy: 330, rx: 92, ry: 40 },
};

export function FacialRegionsCard({ regions }: { regions: FacialRegionScore[] }) {
  const [active, setActive] = React.useState<string | null>(null);
  const ranked = [...regions].sort((a, b) => b.anomalyScore - a.anomalyScore);

  return (
    <GlassCard className="flex flex-col gap-5 p-6">
      <div className="flex flex-col gap-1">
        <h3 className="flex items-center gap-2 font-display text-lg font-semibold text-white">
          <ScanFace className="h-5 w-5 text-accent" />
          All 9 Facial Region Scores
        </h3>
        <p className="text-xs leading-relaxed text-slate-400">
          Every zone is scored. Higher score ⇒ more likely corrupted / manipulated.
          “Claimed” zones cleared the calibrated threshold and have a semantic explanation.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 sm:items-start">
        <svg viewBox="0 0 400 400" className="mx-auto w-full max-w-[220px]">
          <ellipse
            cx="200"
            cy="215"
            rx="115"
            ry="150"
            fill="#0f172a"
            stroke="rgba(255,255,255,0.15)"
            strokeWidth="1.5"
          />
          {ranked.map((r) => {
            const shape = ZONE_SHAPES[r.region];
            if (!shape) return null;
            const color = scoreColor(r.anomalyScore);
            const isActive = active === r.region;
            return (
              <motion.ellipse
                key={r.region}
                cx={shape.cx}
                cy={shape.cy}
                rx={shape.rx}
                ry={shape.ry}
                fill={color}
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.15 + r.anomalyScore * 0.55 }}
                transition={{ duration: 0.6 }}
                stroke={r.flagged ? "#fff" : color}
                strokeWidth={isActive || r.flagged ? 2.4 : 1.1}
                style={{
                  cursor: "pointer",
                  filter: isActive ? `drop-shadow(0 0 10px ${color})` : undefined,
                }}
                onMouseEnter={() => setActive(r.region)}
                onMouseLeave={() => setActive(null)}
              />
            );
          })}
        </svg>

        <div className="flex flex-col gap-2">
          {ranked.map((r) => {
            const color = scoreColor(r.anomalyScore);
            return (
              <div
                key={r.region}
                onMouseEnter={() => setActive(r.region)}
                onMouseLeave={() => setActive(null)}
                className="flex items-center gap-3 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2 transition-colors hover:bg-white/[0.06]"
              >
                <span
                  className="h-2.5 w-2.5 shrink-0 rounded-full"
                  style={{ background: color, boxShadow: `0 0 8px ${color}` }}
                />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="truncate text-sm text-slate-200">{r.label}</span>
                    {r.flagged && (
                      <span className="shrink-0 rounded bg-red-500/20 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide text-red-300">
                        Claimed
                      </span>
                    )}
                  </div>
                  <div className="mt-1 flex items-center gap-2">
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/[0.08]">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.min(r.anomalyScore, 1) * 100}%` }}
                        transition={{ duration: 0.8 }}
                        className="h-full rounded-full"
                        style={{ background: color }}
                      />
                    </div>
                    <span className="w-10 shrink-0 text-right text-xs font-semibold text-slate-300">
                      {(r.anomalyScore * 100).toFixed(0)}%
                    </span>
                  </div>
                  <p className="mt-0.5 text-[10px] text-slate-500">
                    {corruptionLabel(r.anomalyScore)}
                    {r.Q != null ? ` · Q=${r.Q.toFixed(3)}` : ""}
                    {r.R != null ? ` · R=${r.R.toFixed(3)}` : ""}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </GlassCard>
  );
}
