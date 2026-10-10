"use client";

import * as React from "react";
import { Flame, MoveHorizontal } from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import { clamp } from "@/lib/utils";

export function HeatmapCard({
  beforeImage,
  afterImage,
}: {
  /** Real 224x224 aligned face crop that was actually fed to the model. */
  beforeImage: string;
  /** Real grad-weighted attention rollout rendered on the same face crop. */
  afterImage: string;
}) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const [percent, setPercent] = React.useState(50);
  const draggingRef = React.useRef(false);

  const updateFromClientX = React.useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setPercent(clamp(pct, 0, 100));
  }, []);

  React.useEffect(() => {
    function onMove(e: PointerEvent) {
      if (!draggingRef.current) return;
      updateFromClientX(e.clientX);
    }
    function onUp() {
      draggingRef.current = false;
    }
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [updateFromClientX]);

  return (
    <GlassCard className="flex flex-col gap-5 p-6">
      <div className="flex items-center justify-between">
        <h3 className="flex items-center gap-2 font-display text-lg font-semibold text-white">
          <Flame className="h-5 w-5 text-secondary" />
          Attention Heatmap
        </h3>
        <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[11px] font-medium text-slate-400">
          Drag to compare
        </span>
      </div>

      <div
        ref={containerRef}
        className="relative aspect-square w-full max-w-sm mx-auto select-none overflow-hidden rounded-2xl border border-white/10"
        onPointerDown={(e) => {
          draggingRef.current = true;
          updateFromClientX(e.clientX);
        }}
      >
        {/* Before: real 224x224 aligned face crop fed to the model */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={beforeImage}
          alt="Aligned face crop"
          className="absolute inset-0 h-full w-full object-cover"
          draggable={false}
        />

        {/* After: real grad-weighted attention rollout overlay, revealed based on percent */}
        <div
          className="absolute inset-0 h-full w-full overflow-hidden"
          style={{ clipPath: `inset(0 ${100 - percent}% 0 0)` }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={afterImage}
            alt="Attention heatmap overlay"
            className="absolute inset-0 h-full w-full object-cover"
            draggable={false}
          />
          <div className="absolute left-2 top-2 rounded-md bg-black/60 px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-cyan-300">
            After
          </div>
        </div>

        <div className="absolute left-2 bottom-2 rounded-md bg-black/60 px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-slate-300">
          Before
        </div>

        {/* Divider handle */}
        <div
          className="absolute top-0 bottom-0 z-10 w-0.5 bg-white/80 shadow-[0_0_10px_2px_rgba(255,255,255,0.5)]"
          style={{ left: `${percent}%` }}
        >
          <div className="absolute top-1/2 left-1/2 flex h-9 w-9 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize items-center justify-center rounded-full border border-white/40 bg-[#0B1120] shadow-lg">
            <MoveHorizontal className="h-4 w-4 text-white" />
          </div>
        </div>
      </div>

      <input
        type="range"
        min={0}
        max={100}
        value={percent}
        onChange={(e) => setPercent(Number(e.target.value))}
        className="h-1.5 w-full max-w-sm mx-auto cursor-pointer appearance-none rounded-full bg-white/10 accent-primary"
      />
      <p className="text-center text-xs text-slate-500">
        Real grad-weighted attention rollout for this exact image — computed from an actual
        forward + backward pass of the ViT classifier, not a pre-rendered graphic.
      </p>
    </GlassCard>
  );
}
