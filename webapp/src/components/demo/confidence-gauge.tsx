"use client";

import * as React from "react";
import { motion, useMotionValue, animate } from "framer-motion";

export function ConfidenceGauge({
  value,
  color,
  size = 168,
  strokeWidth = 12,
}: {
  value: number;
  color: string;
  size?: number;
  strokeWidth?: number;
}) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const [display, setDisplay] = React.useState(0);
  const progress = useMotionValue(0);

  React.useEffect(() => {
    const controls = animate(progress, value, {
      duration: 1.4,
      ease: "easeOut",
      onUpdate: (v) => setDisplay(v),
    });
    return controls.stop;
  }, [value, progress]);

  const offset = circumference - (display / 100) * circumference;

  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth={strokeWidth}
        />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ filter: `drop-shadow(0 0 8px ${color}aa)` }}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="font-display text-3xl font-bold text-white">
          {display.toFixed(1)}
          <span className="text-lg text-slate-400">%</span>
        </span>
        <span className="text-[10px] font-medium uppercase tracking-widest text-slate-500">
          Confidence
        </span>
      </div>
    </div>
  );
}
