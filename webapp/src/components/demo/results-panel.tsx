"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, ScanSearch } from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import { PredictionCard } from "./prediction-card";
import { ClassificationTheoryCard } from "./classification-theory-card";
import { SemanticExplanationsCard } from "./semantic-explanations-card";
import { HeatmapCard } from "./heatmap-card";
import { MaskGroundTruthCard } from "./mask-ground-truth-card";
import { FacialRegionsCard } from "./facial-regions-card";
import { TimelineCard } from "./timeline-card";
import type { DetectionResponse } from "@/lib/types";

export function ResultsPanel({
  result,
  isProcessing,
}: {
  result: DetectionResponse | null;
  isProcessing: boolean;
}) {
  return (
    <div className="flex h-full flex-col gap-5">
      <AnimatePresence mode="wait">
        {!result && !isProcessing && (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <GlassCard className="flex h-full min-h-[420px] flex-col items-center justify-center gap-4 p-10 text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-white/[0.04] text-slate-500 ring-1 ring-white/10">
                <ScanSearch className="h-8 w-8" />
              </div>
              <h3 className="font-display text-lg font-semibold text-slate-300">
                Awaiting Analysis
              </h3>
              <p className="max-w-xs text-sm text-slate-500">
                Upload media on the left and run detection to view the prediction, semantic
                explanations, and attention heatmap here.
              </p>
            </GlassCard>
          </motion.div>
        )}

        {isProcessing && !result && (
          <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <GlassCard className="flex h-full min-h-[420px] flex-col items-center justify-center gap-5 p-10 text-center">
              <div className="relative flex h-20 w-20 items-center justify-center">
                <span className="absolute inset-0 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
                <Sparkles className="h-8 w-8 text-secondary" />
              </div>
              <h3 className="font-display text-lg font-semibold text-slate-200">
                Analyzing Media...
              </h3>
              <p className="max-w-xs text-sm text-slate-500">
                Watch the processing console for live pipeline status. Results will appear here
                automatically.
              </p>
            </GlassCard>
          </motion.div>
        )}

        {result && (
          <motion.div
            key="results"
            initial="hidden"
            animate="visible"
            className="flex flex-col gap-5"
          >
            <FadeItem index={0}>
              <PredictionCard
                prediction={result.prediction}
                confidence={result.confidence}
                riskLevel={result.riskLevel}
                processingTime={result.processing_time}
                breakdown={result.confidence_breakdown}
              />
            </FadeItem>
            {result.classification_reasoning && (
              <FadeItem index={1}>
                <ClassificationTheoryCard
                  reasoning={result.classification_reasoning}
                  prediction={result.prediction}
                />
              </FadeItem>
            )}
            <FadeItem index={2}>
              <FacialRegionsCard regions={result.facial_regions} />
            </FadeItem>
            <FadeItem index={3}>
              <SemanticExplanationsCard explanations={result.semantic_explanations} />
            </FadeItem>
            <FadeItem index={4}>
              <HeatmapCard beforeImage={result.faceCrop} afterImage={result.heatmap} />
            </FadeItem>
            {result.mask_ground_truth?.available && result.mask_ground_truth.overlay && (
              <FadeItem index={5}>
                <MaskGroundTruthCard
                  beforeImage={result.faceCrop}
                  maskGt={result.mask_ground_truth}
                />
              </FadeItem>
            )}
            {result.mode === "video" && result.timeline.length > 0 && (
              <FadeItem index={result.mask_ground_truth?.available ? 6 : 5}>
                <TimelineCard markers={result.timeline} />
              </FadeItem>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function FadeItem({ children, index }: { children: React.ReactNode; index: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: index * 0.12 }}
    >
      {children}
    </motion.div>
  );
}
