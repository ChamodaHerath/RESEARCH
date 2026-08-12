"use client";

import type { ReactNode } from "react";
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
    <div className="flex flex-col gap-4">
      <AnimatePresence mode="wait">
        {!result && !isProcessing && (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <GlassCard className="flex min-h-[160px] flex-col items-center justify-center gap-3 p-6 text-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-white/[0.04] text-slate-500 ring-1 ring-white/10">
                <ScanSearch className="h-6 w-6" />
              </div>
              <h3 className="font-display text-base font-semibold text-slate-300">
                Awaiting Analysis
              </h3>
              <p className="max-w-md text-sm text-slate-500">
                Upload media above and run detection to view prediction, zones, and heatmap here.
              </p>
            </GlassCard>
          </motion.div>
        )}

        {isProcessing && !result && (
          <motion.div
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <GlassCard className="flex min-h-[160px] flex-col items-center justify-center gap-3 p-6 text-center">
              <div className="relative flex h-14 w-14 items-center justify-center">
                <span className="absolute inset-0 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
                <Sparkles className="h-6 w-6 text-secondary" />
              </div>
              <h3 className="font-display text-base font-semibold text-slate-200">
                Analyzing Media…
              </h3>
              <p className="max-w-md text-sm text-slate-500">
                Watch the processing console. Results appear here when the pipeline finishes.
              </p>
            </GlassCard>
          </motion.div>
        )}

        {result && (
          <motion.div
            key="results"
            initial="hidden"
            animate="visible"
            className="grid grid-cols-1 gap-4 lg:grid-cols-2"
          >
            <FadeItem index={0} className="lg:col-span-2">
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
              <FadeItem
                index={6}
                className="lg:col-span-2"
              >
                <TimelineCard markers={result.timeline} />
              </FadeItem>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function FadeItem({
  children,
  index,
  className = "",
}: {
  children: ReactNode;
  index: number;
  className?: string;
}) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.06 }}
    >
      {children}
    </motion.div>
  );
}
