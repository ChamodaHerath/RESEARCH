"use client";

import { motion } from "framer-motion";
import { ScanFace, Eye, Fingerprint, Cpu, Sparkles } from "lucide-react";

const ORBIT_ITEMS = [
  { icon: Eye, label: "Eye Region", angle: 0, radius: 165, color: "#3B82F6" },
  { icon: Fingerprint, label: "Texture Map", angle: 120, radius: 165, color: "#8B5CF6" },
  { icon: Cpu, label: "Attention Head", angle: 240, radius: 165, color: "#22D3EE" },
];

export function HeroIllustration() {
  return (
    <div className="relative mx-auto flex h-[420px] w-[420px] max-w-full items-center justify-center sm:h-[480px] sm:w-[480px]">
      {/* Outer rotating rings */}
      <div className="absolute inset-0 rounded-full border border-primary/20 animate-spin-slow" />
      <div className="absolute inset-6 rounded-full border border-secondary/20 animate-spin-slower" />
      <div className="absolute inset-16 rounded-full border border-dashed border-accent/20" />

      {/* Glow blobs */}
      <div className="absolute h-64 w-64 rounded-full bg-primary/25 blur-[80px]" />
      <div className="absolute h-52 w-52 translate-x-16 translate-y-10 rounded-full bg-secondary/25 blur-[70px]" />

      {/* Core scanning face */}
      <motion.div
        initial={{ opacity: 0, scale: 0.85 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.9, ease: "easeOut" }}
        className="glass glow-border relative flex h-56 w-56 items-center justify-center rounded-full sm:h-64 sm:w-64"
      >
        <ScanFace className="h-24 w-24 text-blue-300 sm:h-28 sm:w-28" strokeWidth={1.2} />
        <motion.div
          className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-cyan-300 to-transparent shadow-[0_0_12px_2px_rgba(34,211,238,0.9)]"
          animate={{ top: ["12%", "88%", "12%"] }}
          transition={{ duration: 3.2, repeat: Infinity, ease: "easeInOut" }}
        />
        <span className="absolute -bottom-3 flex items-center gap-1 rounded-full border border-accent/40 bg-[#050816]/90 px-3 py-1 text-[10px] font-semibold uppercase tracking-widest text-cyan-300">
          <Sparkles className="h-3 w-3" /> Analyzing
        </span>
      </motion.div>

      {/* Orbiting analysis nodes */}
      {ORBIT_ITEMS.map((item, i) => {
        const Icon = item.icon;
        return (
          <motion.div
            key={item.label}
            className="absolute"
            animate={{ rotate: 360 }}
            transition={{ duration: 18 + i * 4, repeat: Infinity, ease: "linear" }}
            style={{ width: item.radius * 2, height: item.radius * 2 }}
          >
            <div
              className="absolute flex flex-col items-center gap-1.5"
              style={{
                top: `calc(50% + ${(Math.sin((item.angle * Math.PI) / 180) * item.radius - 28).toFixed(3)}px)`,
                left: `calc(50% + ${(Math.cos((item.angle * Math.PI) / 180) * item.radius - 28).toFixed(3)}px)`,
              }}
            >
              <motion.div
                animate={{ rotate: -360 }}
                transition={{ duration: 18 + i * 4, repeat: Infinity, ease: "linear" }}
                className="glass flex h-14 w-14 items-center justify-center rounded-2xl"
                style={{ boxShadow: `0 0 24px -4px ${item.color}88` }}
              >
                <Icon className="h-6 w-6" style={{ color: item.color }} />
              </motion.div>
            </div>
          </motion.div>
        );
      })}

      {/* Floating data chips */}
      <motion.div
        className="glass absolute -left-2 top-6 rounded-xl px-3 py-2 text-[11px] font-mono text-emerald-300 sm:-left-8"
        animate={{ y: [0, -10, 0] }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
      >
        conf: 0.9642
      </motion.div>
      <motion.div
        className="glass absolute -right-2 bottom-10 rounded-xl px-3 py-2 text-[11px] font-mono text-cyan-300 sm:-right-8"
        animate={{ y: [0, 10, 0] }}
        transition={{ duration: 4.5, repeat: Infinity, ease: "easeInOut", delay: 0.5 }}
      >
        attn_map: OK
      </motion.div>
      <motion.div
        className="glass absolute bottom-0 left-1/2 -translate-x-1/2 rounded-xl px-3 py-2 text-[11px] font-mono text-purple-300"
        animate={{ y: [0, -8, 0] }}
        transition={{ duration: 3.6, repeat: Infinity, ease: "easeInOut", delay: 1 }}
      >
        patches: 196
      </motion.div>
    </div>
  );
}
