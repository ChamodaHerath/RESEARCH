import { GlassCard } from "@/components/ui/glass-card";
import { Check, Minus } from "lucide-react";
import { cn } from "@/lib/utils";
import { FFPP_EVAL, CELEBDF_EVAL } from "@/lib/real-eval";

// Accuracy figures for MesoNet / Xception / EfficientNet-B4 / CNN+Grad-CAM were
// NOT reproduced or benchmarked in this project, so no numeric accuracy is shown
// for them — showing a number without re-running the baseline would be
// fabricated data. Capability flags below reflect well-established, citable
// architectural facts about each family of methods from the deepfake-detection
// literature. Only "ExplainDeep (ours)" reports a number, and it is the real,
// measured accuracy from this project's own evaluation runs.
const ROWS = [
  { method: "MesoNet (2018)", accuracy: "not reproduced here", explainable: false, attention: false, uncertainty: false },
  { method: "Xception (2019)", accuracy: "not reproduced here", explainable: false, attention: false, uncertainty: false },
  { method: "EfficientNet-B4", accuracy: "not reproduced here", explainable: false, attention: false, uncertainty: false },
  { method: "CNN + Grad-CAM", accuracy: "not reproduced here", explainable: "partial" as const, attention: true, uncertainty: false },
  {
    method: "ExplainDeep (ours)",
    accuracy: `${(FFPP_EVAL.classification_metrics.accuracy * 100).toFixed(1)}% FF++ / ${(CELEBDF_EVAL.classification_metrics.accuracy * 100).toFixed(1)}% Celeb-DF`,
    explainable: true,
    attention: true,
    uncertainty: true,
    highlight: true,
  },
];

export function ComparisonTable() {
  return (
    <GlassCard className="flex flex-col gap-5 p-6">
      <h3 className="font-display text-base font-semibold text-white">
        Qualitative Comparison with Literature Approaches
      </h3>
      <p className="text-xs text-slate-500">
        Capability flags reflect documented architectural properties of each method family. No
        baseline accuracy is fabricated here — only this project&apos;s own measured accuracy is
        shown numerically.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[520px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-white/10 text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="py-2.5 pr-3">Method</th>
              <th className="py-2.5 pr-3">Accuracy (measured in this project)</th>
              <th className="py-2.5 pr-3">Semantic Zone Explanations</th>
              <th className="py-2.5 pr-3">Attention / Saliency Maps</th>
              <th className="py-2.5">Bayesian Uncertainty</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row) => (
              <tr
                key={row.method}
                className={cn(
                  "border-b border-white/5",
                  row.highlight && "bg-primary/5"
                )}
              >
                <td className={cn("py-3 pr-3 font-medium", row.highlight ? "text-white" : "text-slate-300")}>
                  {row.method}
                </td>
                <td className={cn("py-3 pr-3 font-semibold", row.highlight ? "text-accent" : "text-slate-500 italic")}>
                  {row.accuracy}
                </td>
                <td className="py-3 pr-3">
                  {row.explainable === true ? (
                    <Check className="h-4 w-4 text-emerald-400" />
                  ) : row.explainable === "partial" ? (
                    <span className="text-xs text-amber-400">Partial</span>
                  ) : (
                    <Minus className="h-4 w-4 text-slate-600" />
                  )}
                </td>
                <td className="py-3 pr-3">
                  {row.attention ? (
                    <Check className="h-4 w-4 text-emerald-400" />
                  ) : (
                    <Minus className="h-4 w-4 text-slate-600" />
                  )}
                </td>
                <td className="py-3">
                  {row.uncertainty ? (
                    <Check className="h-4 w-4 text-emerald-400" />
                  ) : (
                    <Minus className="h-4 w-4 text-slate-600" />
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </GlassCard>
  );
}
