"use client";

import { motion } from "framer-motion";
import { BrainCircuit, Eye, MessagesSquare, ShieldCheck, Layers3, Radar } from "lucide-react";
import { SectionHeading } from "@/components/ui/glass-card";

const HIGHLIGHTS = [
  {
    icon: MessagesSquare,
    title: "Semantic Explainability",
    description:
      "Goes beyond a bare probability score — every verdict is accompanied by natural-language forensic reasoning.",
    color: "from-blue-500/20 to-blue-500/0",
  },
  {
    icon: BrainCircuit,
    title: "Vision Transformer Backbone",
    description:
      "Leverages self-attention over image patches to model long-range facial dependencies more effectively than CNNs.",
    color: "from-purple-500/20 to-purple-500/0",
  },
  {
    icon: Eye,
    title: "Attention Maps",
    description:
      "Attention rollout visualizes precisely where the model is looking, exposing manipulated regions.",
    color: "from-cyan-500/20 to-cyan-500/0",
  },
  {
    icon: Layers3,
    title: "Human-Readable Explanations",
    description:
      "Zone-level anomaly scores are translated into forensic-analyst-style statements, not just heatmaps.",
    color: "from-emerald-500/20 to-emerald-500/0",
  },
  {
    icon: ShieldCheck,
    title: "Forensic Transparency",
    description:
      "Every case produces a structured, auditable report suitable for chain-of-custody documentation.",
    color: "from-amber-500/20 to-amber-500/0",
  },
  {
    icon: Radar,
    title: "Cross-Dataset Evaluation",
    description:
      "Evaluated on both FaceForensics++ and Celeb-DF v2 — two distinct manipulation methods — with real results reported for each, not averaged away.",
    color: "from-rose-500/20 to-rose-500/0",
  },
];

export function HighlightsSection() {
  return (
    <section className="section-pad relative">
      <div className="container-app flex flex-col gap-12">
        <SectionHeading
          eyebrow="Research Highlights"
          title="What makes this research different"
          description="A forensic-first approach to deepfake detection — designed for transparency, auditability, and trust."
        />
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {HIGHLIGHTS.map((item, i) => {
            const Icon = item.icon;
            return (
              <motion.div
                key={item.title}
                initial={{ opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-60px" }}
                transition={{ duration: 0.5, delay: i * 0.07 }}
                className="glass glow-hover group relative flex flex-col gap-4 p-7"
              >
                <div className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${item.color} opacity-60`} />
                <div className="relative flex h-12 w-12 items-center justify-center rounded-xl bg-white/[0.06] text-blue-300 ring-1 ring-white/10">
                  <Icon className="h-6 w-6" />
                </div>
                <h3 className="relative font-display text-lg font-semibold text-white">
                  {item.title}
                </h3>
                <p className="relative text-sm leading-relaxed text-slate-400">
                  {item.description}
                </p>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
