"use client";

import { motion } from "framer-motion";
import {
  UploadCloud,
  ScanFace,
  SlidersHorizontal,
  Grid3x3,
  BrainCircuit,
  Eye,
  Network,
  Fingerprint,
  FileText,
  ArrowDown,
} from "lucide-react";
import { cn } from "@/lib/utils";

const STEPS = [
  {
    icon: UploadCloud,
    title: "Upload",
    description: "User submits an image or video through the secure ingestion pipeline.",
  },
  {
    icon: ScanFace,
    title: "Face Detection",
    description: "MTCNN localizes the face region; dlib's 68-point predictor extracts landmarks from the crop.",
  },
  {
    icon: SlidersHorizontal,
    title: "Preprocessing",
    description: "Alignment, normalization and frame sampling prepare data for the encoder.",
  },
  {
    icon: Grid3x3,
    title: "Patch Generation",
    description: "The face crop is divided into 16×16 patches with positional embeddings.",
  },
  {
    icon: BrainCircuit,
    title: "Vision Transformer",
    description: "A multi-head self-attention encoder extracts deep visual representations.",
  },
  {
    icon: Eye,
    title: "Attention Analysis",
    description: "Attention rollout produces spatial heatmaps highlighting salient regions.",
  },
  {
    icon: Network,
    title: "Semantic Mapping",
    description: "Attention weights are mapped onto interpretable facial-forensic zones.",
  },
  {
    icon: Fingerprint,
    title: "Classification",
    description: "The pretrained ViT head outputs a softmax fake-probability; the zone-explanation layer built on top is what is separately Bayesian-calibrated.",
  },
  {
    icon: FileText,
    title: "Forensic Report",
    description: "A structured, human-readable forensic report is generated for the case file.",
  },
];

export function WorkflowDiagram() {
  return (
    <div className="mx-auto flex max-w-3xl flex-col items-center">
      {STEPS.map((step, i) => {
        const Icon = step.icon;
        const isLast = i === STEPS.length - 1;
        return (
          <div key={step.title} className="flex w-full flex-col items-center">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.5, delay: i * 0.05 }}
              className={cn(
                "glass glow-hover flex w-full items-center gap-5 px-6 py-5",
                i % 2 === 0 ? "sm:flex-row" : "sm:flex-row-reverse sm:text-right"
              )}
            >
              <div className="relative flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/25 to-secondary/25 text-blue-300">
                <span className="absolute -top-2 -left-2 flex h-5 w-5 items-center justify-center rounded-full bg-[#050816] text-[10px] font-bold text-slate-300 ring-1 ring-white/15">
                  {i + 1}
                </span>
                <Icon className="h-6 w-6" />
              </div>
              <div className="flex-1">
                <h3 className="font-display text-lg font-semibold text-white">{step.title}</h3>
                <p className="mt-1 text-sm leading-relaxed text-slate-400">{step.description}</p>
              </div>
            </motion.div>
            {!isLast && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                whileInView={{ opacity: 1, height: 36 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: i * 0.05 + 0.15 }}
                className="flex flex-col items-center py-1"
              >
                <div className="h-full w-px bg-gradient-to-b from-primary/60 to-secondary/60" />
                <ArrowDown className="h-4 w-4 -mt-1 text-secondary/70" />
              </motion.div>
            )}
          </div>
        );
      })}
    </div>
  );
}
