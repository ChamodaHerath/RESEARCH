"use client";

import { HelpCircle } from "lucide-react";
import { getConfidenceMetricHelp } from "@/lib/confidence-breakdown-help";
import type { ConfidenceBreakdown } from "@/lib/types";

export function ConfidenceBreakdownList({
  items,
}: {
  items: ConfidenceBreakdown[];
}) {
  if (items.length === 0) {
    return (
      <p className="text-sm text-slate-500">No breakdown metrics returned.</p>
    );
  }

  return (
    <ul className="flex flex-col gap-2">
      {items.map((c) => {
        const help = getConfidenceMetricHelp(c.metric);
        return (
          <li
            key={c.metric}
            className="group relative flex items-center justify-between gap-3 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2"
          >
            <span
              className={
                help
                  ? "flex min-w-0 cursor-help items-start gap-1.5 text-xs text-slate-400"
                  : "min-w-0 text-xs text-slate-400"
              }
              title={help ? undefined : c.metric}
            >
              <span className="min-w-0">{c.metric}</span>
              {help ? (
                <HelpCircle
                  className="mt-0.5 h-3.5 w-3.5 shrink-0 text-slate-500 transition-colors group-hover:text-sky-400"
                  aria-hidden
                />
              ) : null}
            </span>
            <span className="shrink-0 font-mono text-sm font-semibold text-white">
              {c.value}%
            </span>
            {help ? (
              <div
                role="tooltip"
                className="pointer-events-none absolute bottom-full left-3 right-3 z-30 mb-1.5 hidden rounded-lg border border-white/15 bg-slate-900/95 px-3 py-2 text-[11px] leading-relaxed text-slate-300 shadow-xl backdrop-blur-sm group-hover:block"
              >
                {help}
              </div>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}
