"use client";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
} from "recharts";
import { GlassCard, SectionHeading } from "@/components/ui/glass-card";
import { FFPP_EVAL, CELEBDF_EVAL } from "@/lib/real-eval";

// Every data point below comes directly from the two real evaluation runs
// (outputs/evaluation_results_ffpp.json and outputs/evaluation_results_celebdf.json).
// There is only one real model, evaluated on two real test partitions —
// no synthetic training curves or baseline comparisons are shown here.

const ACCURACY_BY_DATASET = [
  { name: "FaceForensics++", accuracy: +(FFPP_EVAL.classification_metrics.accuracy * 100).toFixed(2) },
  { name: "Celeb-DF v2", accuracy: +(CELEBDF_EVAL.classification_metrics.accuracy * 100).toFixed(2) },
];

const ZONE_SEPARATION = Object.entries(CELEBDF_EVAL.explanation_metrics.zone_R_separation).map(
  ([zone, s]) => ({
    zone: zone.replace("_", " "),
    fakeMeanR: +(s.fake_mean_R * 100).toFixed(2),
    realMeanR: +(s.real_mean_R * 100).toFixed(2),
  })
);

function metricRow(name: string, key: "ffpp" | "celebdf") {
  const m = key === "ffpp" ? FFPP_EVAL.classification_metrics : CELEBDF_EVAL.classification_metrics;
  return name === "AUC-ROC" ? m.auc_roc * 100 : name === "Accuracy" ? m.accuracy * 100 : name === "Precision" ? m.precision * 100 : name === "Recall" ? m.recall * 100 : m.f1 * 100;
}

const METRIC_RADAR = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"].map((metric) => ({
  metric,
  ffpp: +metricRow(metric, "ffpp").toFixed(1),
  celebdf: +metricRow(metric, "celebdf").toFixed(1),
}));

const TOOLTIP_STYLE = {
  background: "#0B1120",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 12,
  color: "#e5e9f5",
  fontSize: 12,
};

export function ResultsCharts() {
  return (
    <div className="flex flex-col gap-10">
      <SectionHeading
        eyebrow="Real Evaluation Charts"
        title="Classification accuracy & zone-level separation"
        description="Sourced directly from evaluate_pipeline.py output files. The detector is currently set to hamzenium/ViT-Deepfake-Classifier at the user's request; its in-domain and cross-dataset scores are reported exactly as measured, without smoothing or replacement."
      />
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <GlassCard className="flex flex-col gap-4 p-6">
          <h3 className="font-display text-base font-semibold text-white">
            Classification Accuracy by Dataset (real)
          </h3>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ACCURACY_BY_DATASET} margin={{ left: -20, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <YAxis domain={[0, 100]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "rgba(59,130,246,0.08)" }} />
                <defs>
                  <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#3B82F6" />
                    <stop offset="100%" stopColor="#8B5CF6" />
                  </linearGradient>
                </defs>
                <Bar dataKey="accuracy" name="Accuracy %" fill="url(#barGrad)" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-slate-500">
            n = {FFPP_EVAL.classification_metrics.n_total} frames (FF++), {CELEBDF_EVAL.classification_metrics.n_total} frames (Celeb-DF v2).
          </p>
        </GlassCard>

        <GlassCard className="flex flex-col gap-4 p-6">
          <h3 className="font-display text-base font-semibold text-white">
            Fused Zone Anomaly R(z): Fake vs Real (Celeb-DF, real)
          </h3>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ZONE_SEPARATION} margin={{ left: -20, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
                <XAxis
                  dataKey="zone"
                  tick={{ fill: "#94a3b8", fontSize: 9 }}
                  interval={0}
                  angle={-25}
                  textAnchor="end"
                  height={55}
                />
                <YAxis domain={[0, 100]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Legend wrapperStyle={{ fontSize: 10, color: "#94a3b8" }} />
                <Bar dataKey="fakeMeanR" name="Fake mean R(z) %" fill="#ef4444" radius={[6, 6, 0, 0]} />
                <Bar dataKey="realMeanR" name="Real mean R(z) %" fill="#3B82F6" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-slate-500">
            Most zones separate clearly; the mouth zone is a known exception — its fake-mean R(z)
            is actually <em>lower</em> than its real-mean R(z) for this detector, an honest
            limitation discussed in README.md rather than hidden here.
          </p>
        </GlassCard>

        <GlassCard className="flex flex-col gap-4 p-6 lg:col-span-2">
          <h3 className="font-display text-base font-semibold text-white">
            Classification Metrics — FaceForensics++ vs Celeb-DF v2 (real)
          </h3>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={METRIC_RADAR} outerRadius="75%">
                <PolarGrid stroke="rgba(255,255,255,0.1)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <PolarRadiusAxis domain={[0, 100]} tick={{ fill: "#64748b", fontSize: 9 }} />
                <Radar name="FaceForensics++" dataKey="ffpp" stroke="#22D3EE" fill="#22D3EE" fillOpacity={0.3} />
                <Radar name="Celeb-DF v2" dataKey="celebdf" stroke="#8B5CF6" fill="#8B5CF6" fillOpacity={0.25} />
                <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-slate-500">
            No external baseline is plotted here because no baseline model was re-implemented or
            evaluated in this project — see the qualitative comparison table below instead.
          </p>
        </GlassCard>
      </div>
    </div>
  );
}
