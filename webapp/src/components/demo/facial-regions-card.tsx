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

const REGION_SHAPES: Record<
  FacialRegionScore["region"],
  { cx: number; cy: number; rx: number; ry: number }
> = {
  forehead: { cx: 200, cy: 120, rx: 78, ry: 44 },
  eyes: { cx: 200, cy: 188, rx: 88, ry: 26 },
  nose: { cx: 200, cy: 228, rx: 26, ry: 40 },
  mouth: { cx: 200, cy: 285, rx: 46, ry: 20 },
  jaw: { cx: 200, cy: 320, rx: 92, ry: 46 },
};

export function FacialRegionsCard({ regions }: { regions: FacialRegionScore[] }) {
  const [active, setActive] = React.useState<string | null>(null);

  return (
    <GlassCard className="flex flex-col gap-5 p-6">
      <div className="flex items-center justify-between">
        <h3 className="flex items-center gap-2 font-display text-lg font-semibold text-white">
          <ScanFace className="h-5 w-5 text-accent" />
          Detected Facial Regions
        </h3>
      </div>

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 sm:items-center">
        <svg viewBox="0 0 400 400" className="mx-auto w-full max-w-[220px]">
          <ellipse cx="200" cy="215" rx="115" ry="150" fill="#0f172a" stroke="rgba(255,255,255,0.15)" strokeWidth="1.5" />
          {regions.map((r) => {
            const shape = REGION_SHAPES[r.region];
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
                animate={{ opacity: 0.18 + r.anomalyScore * 0.5 }}
                transition={{ duration: 0.6 }}
                stroke={color}
                strokeWidth={isActive ? 2.5 : 1.2}
                style={{ cursor: "pointer", filter: isActive ? `drop-shadow(0 0 10px ${color})` : undefined }}
                onMouseEnter={() => setActive(r.region)}
                onMouseLeave={() => setActive(null)}
              />
            );
          })}
        </svg>

        <div className="flex flex-col gap-2.5">
          {regions.map((r) => {
            const color = scoreColor(r.anomalyScore);
            return (
              <div
                key={r.region}
                onMouseEnter={() => setActive(r.region)}
                onMouseLeave={() => setActive(null)}
                className="flex items-center gap-3 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2 transition-colors hover:bg-white/[0.06]"
              >
                <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: color, boxShadow: `0 0 8px ${color}` }} />
                <span className="w-20 shrink-0 text-sm text-slate-200">{r.label}</span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/[0.08]">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${r.anomalyScore * 100}%` }}
                    transition={{ duration: 0.8 }}
                    className="h-full rounded-full"
                    style={{ background: color }}
                  />
                </div>
                <span className="w-10 shrink-0 text-right text-xs font-medium text-slate-400">
                  {(r.anomalyScore * 100).toFixed(0)}%
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </GlassCard>
  );
}
