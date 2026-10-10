"use client";

import { Database, FlaskConical, ShieldCheck, Sigma, Timer } from "lucide-react";
import { SectionHeading } from "@/components/ui/glass-card";
import { AnimatedCounterCard } from "@/components/ui/animated-counter";
import {
  FFPP_EVAL,
  CELEBDF_EVAL,
  FUSION_CONFIG,
  combinedTotalFrames,
  combinedAvgSecondsPerFrame,
} from "@/lib/real-eval";

// Every value below is read directly from outputs/evaluation_results_ffpp.json,
// outputs/evaluation_results_celebdf.json and configs/fusion_config.json —
// the actual files produced by evaluate_pipeline.py and recalibrate_v2.py.
// None of these numbers are placeholders. Full classification accuracy is
// reported separately on the Results page (it differs materially between
// in-domain FF++ and cross-dataset Celeb-DF, so a single headline
// percentage here would be misleading).
const avgMsPerFrame = combinedAvgSecondsPerFrame() * 1000;
const faithfulnessPct =
  Math.min(FFPP_EVAL.explanation_metrics.faithfulness_pass_rate, CELEBDF_EVAL.explanation_metrics.faithfulness_pass_rate) *
  100;

const STATS = [
  {
    icon: <Database className="h-6 w-6" />,
    value: combinedTotalFrames(),
    label: "Frames Evaluated (FF++ + Celeb-DF v2)",
    suffix: "",
  },
  {
    icon: <FlaskConical className="h-6 w-6" />,
    value: FUSION_CONFIG.calibration_n_frames,
    label: "Calibration Frames (β fitting)",
    suffix: "",
  },
  {
    icon: <ShieldCheck className="h-6 w-6" />,
    value: faithfulnessPct,
    label: "Explanation Faithfulness Pass Rate",
    suffix: "%",
    decimals: 0,
  },
  {
    icon: <Sigma className="h-6 w-6" />,
    value: FUSION_CONFIG.beta * 100,
    label: "Calibrated Fusion Weight β (MAP)",
    suffix: "%",
    decimals: 1,
  },
  {
    icon: <Timer className="h-6 w-6" />,
    value: avgMsPerFrame,
    label: "Avg. Inference Time / Frame",
    suffix: "ms",
    decimals: 0,
  },
];

export function StatsSection() {
  return (
    <section className="section-pad relative">
      <div className="container-app flex flex-col gap-6">
        <SectionHeading
          eyebrow="Real Evaluation Results"
          title="Measured directly from evaluate_pipeline.py runs"
          description="Every figure below is read from the actual JSON output files of the evaluation pipeline — no simulated or placeholder values. Full classification accuracy (including the honest cross-dataset numbers) is reported on the Results page."
        />
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
          {STATS.map((stat, i) => (
            <AnimatedCounterCard key={stat.label} index={i} {...stat} />
          ))}
        </div>
      </div>
    </section>
  );
}
