import { SectionBadge } from "@/components/ui/glass-card";
import { ResultsCharts } from "@/components/results/results-charts";
import { ComparisonTable } from "@/components/results/comparison-table";
import { ConfusionMatrix } from "@/components/results/confusion-matrix";

export default function ResultsPage() {
  return (
    <div className="section-pad container-app flex flex-col gap-16">
      <div className="flex flex-col items-center gap-5 text-center">
        <SectionBadge>Experimental Results</SectionBadge>
        <h1 className="max-w-3xl font-display text-3xl font-bold text-white sm:text-4xl md:text-5xl">
          Real evaluation results
        </h1>
        <p className="max-w-2xl text-slate-400">
          Every number on this page is read directly from the JSON files produced by
          <code className="mx-1 rounded bg-white/10 px-1.5 py-0.5 text-xs text-cyan-300">evaluate_pipeline.py</code>
          on the FaceForensics++ and Celeb-DF v2 test partitions. Classification accuracy is
          intentionally reported as-is, including its limitations — this project&apos;s
          contribution is the calibrated Bayesian explanation layer on top of the classifier, not
          a state-of-the-art detection accuracy claim.
        </p>
      </div>

      <ResultsCharts />

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-2 lg:items-start">
        <ConfusionMatrix />
        <ComparisonTable />
      </div>
    </div>
  );
}
