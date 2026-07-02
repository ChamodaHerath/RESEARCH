"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { Film, AlertTriangle } from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import type { TimelineMarker } from "@/lib/types";
import { cn } from "@/lib/utils";

function markerColor(suspicion: number) {
  if (suspicion > 0.65) return "#ef4444";
  if (suspicion > 0.35) return "#f59e0b";
  return "#22c55e";
}

export function TimelineCard({ markers }: { markers: TimelineMarker[] }) {
  const [selected, setSelected] = React.useState(0);
  if (markers.length === 0) return null;
  const active = markers[selected];
  const maxFrame = markers[markers.length - 1]?.frame ?? 1;

  return (
    <GlassCard className="flex flex-col gap-5 p-6">
      <div className="flex items-center justify-between">
        <h3 className="flex items-center gap-2 font-display text-lg font-semibold text-white">
          <Film className="h-5 w-5 text-primary" />
          Frame Timeline
        </h3>
        <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[11px] font-medium text-slate-400">
          {markers.length} markers
        </span>
      </div>

      <div className="relative pt-2">
        <div className="relative h-1.5 rounded-full bg-white/[0.08]">
          <div className="absolute inset-0 rounded-full bg-gradient-to-r from-emerald-500/40 via-amber-500/40 to-red-500/40" />
        </div>
        <div className="absolute inset-x-0 top-0 flex h-1.5 items-center">
          {markers.map((m, i) => {
            const left = (m.frame / maxFrame) * 100;
            const color = markerColor(m.suspicion);
            const isActive = i === selected;
            return (
              <button
                key={m.frame}
                onClick={() => setSelected(i)}
                className="absolute -translate-x-1/2"
                style={{ left: `${left}%` }}
              >
                <motion.span
                  animate={{ scale: isActive ? 1.5 : 1 }}
                  className="block h-3.5 w-3.5 rounded-full border-2 border-[#050816]"
                  style={{ background: color, boxShadow: isActive ? `0 0 12px 3px ${color}` : `0 0 6px 1px ${color}88` }}
                />
              </button>
            );
          })}
        </div>
      </div>

      <motion.div
        key={active.frame}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-start gap-3 rounded-xl border border-white/10 bg-white/[0.03] p-4"
      >
        <div
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg"
          style={{ background: `${markerColor(active.suspicion)}22`, color: markerColor(active.suspicion) }}
        >
          <AlertTriangle className="h-4 w-4" />
        </div>
        <div className="flex-1">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm font-semibold text-slate-100">
            Frame {active.frame}
            <span className="text-xs font-normal text-slate-500">t = {active.timeSec}s</span>
            <span
              className={cn("rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase")}
              style={{
                background: `${markerColor(active.suspicion)}1a`,
                color: markerColor(active.suspicion),
              }}
            >
              {(active.suspicion * 100).toFixed(0)}% suspicion
            </span>
          </div>
          <p className="mt-1 text-xs leading-relaxed text-slate-400">{active.note}</p>
        </div>
      </motion.div>
    </GlassCard>
  );
}
