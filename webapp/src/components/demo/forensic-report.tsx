"use client";

import * as React from "react";
import { motion } from "framer-motion";
import {
  FileText,
  Download,
  ShieldCheck,
  ShieldAlert,
  Fingerprint,
} from "lucide-react";
import { GlassCard, SectionBadge } from "@/components/ui/glass-card";
import { Button } from "@/components/ui/button";
import { generateReportPdf } from "@/lib/generate-report-pdf";
import type { DetectionResponse } from "@/lib/types";

function faithfulnessPct(result: DetectionResponse): string {
  const v = result.explanation_validation?.live;
  if (v) return `${(v.faithfulness_pass_rate * 100).toFixed(0)}%`;
  const fromBreakdown = result.confidence_breakdown.find((c) =>
    c.metric.toLowerCase().includes("faithfulness")
  );
  if (fromBreakdown) return `${fromBreakdown.value}%`;
  return result.semantic_explanations.length > 0 ? "100%" : "—";
}

function whyZeroZones(result: DetectionResponse): string {
  const isFake = result.prediction === "Fake";
  const confPct = result.confidence;
  const pFakePct = isFake ? confPct : Math.max(0, 100 - confPct);

  const elevated = [...result.facial_regions]
    .filter(
      (r) =>
        r.clears_threshold === true ||
        r.flagged === true ||
        (r.threshold != null && r.anomalyScore > r.threshold) ||
        r.anomalyScore > 0.65
    )
    .sort((a, b) => b.anomalyScore - a.anomalyScore)
    .slice(0, 3);

  const zoneList =
    elevated.length > 0
      ? elevated
          .map(
            (r) =>
              `${r.label} (${(r.anomalyScore * 100).toFixed(1)}%` +
              (r.threshold != null
                ? `; thr ${(r.threshold * 100).toFixed(1)}%`
                : "") +
              `)`
          )
          .join("; ")
      : null;

  if (!isFake) {
    return (
      `No semantic sentences were emitted (verdict is Real).` +
      (zoneList
        ? ` Elevated region scores for inspection: ${zoneList}.`
        : ` Region scores are shown for inspection only.`) +
      ` Fake confidence is ${pFakePct.toFixed(1)}%. We cannot claim facial zones are corrupted.`
    );
  }

  if (pFakePct <= 70) {
    return (
      `No semantic sentences were emitted.` +
      (zoneList
        ? ` Candidate / elevated zones: ${zoneList}.`
        : ` No zone clearly exceeded its calibrated threshold.`) +
      ` Fake confidence is ${pFakePct.toFixed(1)}% (needs > 70% for NL claims).`
    );
  }

  return (
    `No semantic sentences were emitted.` +
    (zoneList
      ? ` Elevated region scores: ${zoneList}.`
      : ` No region crossed the calibrated Q(z) threshold.`) +
    ` Fake confidence is ${pFakePct.toFixed(1)}%.`
  );
}

export function ForensicReport({ result }: { result: DetectionResponse }) {
  const isFake = result.prediction === "Fake";
  const color = isFake ? "#ef4444" : "#22c55e";
  const rankedRegions = [...result.facial_regions].sort(
    (a, b) => b.anomalyScore - a.anomalyScore
  );
  const flaggedCount = rankedRegions.filter((r) => r.flagged).length;
  const heatmapSrc = result.heatmap || result.attention_map;

  return (
    <motion.section
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{ duration: 0.6 }}
      className="w-full"
    >
      <div className="mb-8 flex flex-col items-center gap-4 text-center">
        <SectionBadge>Forensic Documentation</SectionBadge>
        <h2 className="font-display text-3xl font-bold text-white">Digital Forensic Report</h2>
        <p className="max-w-xl text-sm text-slate-400">
          Case file with classification verdict, attention heatmap, region scores, and
          semantic explanations.
        </p>
      </div>

      <GlassCard glow className="mx-auto max-w-4xl overflow-hidden !p-0">
        <div className="bg-[#0b1120] p-1">
          <div className="rounded-2xl bg-gradient-to-b from-slate-100 to-slate-200 p-8 text-slate-900 sm:p-10">
            <div className="flex flex-col items-start justify-between gap-4 border-b border-slate-300 pb-6 sm:flex-row sm:items-center">
              <div>
                <div className="flex items-center gap-2 text-lg font-bold text-slate-900">
                  <Fingerprint className="h-5 w-5 text-blue-600" />
                  ExplainDeep Forensic Report
                </div>
                <p className="text-xs text-slate-500">
                  Extended Explainable Vision Transformer Framework For Deepfake Detection
                </p>
              </div>
              <div
                className="flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-bold"
                style={{ background: `${color}1a`, color }}
              >
                {isFake ? <ShieldAlert className="h-4 w-4" /> : <ShieldCheck className="h-4 w-4" />}
                {result.prediction.toUpperCase()} · {result.confidence.toFixed(2)}%
              </div>
            </div>

            {/* 1. Case information */}
            <ReportSection title="1. Case Information">
              <ReportGrid
                rows={[
                  ["Case ID", result.report.caseId],
                  ["Date", result.report.date],
                  ["Time", result.report.time],
                  ["Model Version", result.report.modelVersion],
                  ["Processing Time", result.processing_time],
                  ["Input File", result.report.inputFile],
                  ["Hash Value (SHA-256)", `${result.report.hashValue.slice(0, 24)}…`],
                ]}
              />
            </ReportSection>

            {/* 2. Classification head */}
            <ReportSection title="2. Classification Head Result">
              <ReportGrid
                rows={[
                  ["Verdict", result.prediction.toUpperCase()],
                  ["Confidence", `${result.confidence.toFixed(2)}%`],
                  ["Risk Level", result.riskLevel],
                ]}
              />
            </ReportSection>

            {/* 3. Heatmap + facial regions */}
            <ReportSection title="3. Heatmap & Detected Facial Regions">
              <div className="mb-4 overflow-hidden rounded-xl border border-slate-200 bg-slate-900">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={heatmapSrc}
                  alt="Attention heatmap overlay"
                  className="mx-auto max-h-72 w-full object-contain"
                />
              </div>
              {result.mask_ground_truth?.available && result.mask_ground_truth.overlay && (
                <div className="mb-4">
                  <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-emerald-700">
                    Manipulation mask ground truth (true edited region)
                  </p>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="overflow-hidden rounded-xl border border-slate-200 bg-slate-900">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={result.faceCrop}
                        alt="Aligned face crop"
                        className="mx-auto max-h-56 w-full object-contain"
                      />
                      <p className="bg-slate-100 px-2 py-1 text-center text-[10px] font-medium text-slate-600">
                        Face crop
                      </p>
                    </div>
                    <div className="overflow-hidden rounded-xl border border-emerald-200 bg-slate-900">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={result.mask_ground_truth.overlay}
                        alt="FF++ manipulation mask ground truth"
                        className="mx-auto max-h-56 w-full object-contain"
                      />
                      <p className="bg-emerald-50 px-2 py-1 text-center text-[10px] font-medium text-emerald-800">
                        Green = true edited pixels (FF++ mask)
                      </p>
                    </div>
                  </div>
                  <p className="mt-2 text-[11px] text-slate-600">
                    {result.mask_ground_truth.note}
                  </p>
                </div>
              )}
              <p className="mb-3 text-[11px] text-slate-600">
                Detected facial regions with corruption score and calibrated flagging
                threshold (higher score = more likely manipulated).
              </p>
              <table className="w-full border-collapse text-xs">
                <thead>
                  <tr className="border-b border-slate-300 text-left text-slate-500">
                    <th className="py-1.5 font-semibold">Region</th>
                    <th className="py-1.5 font-semibold">Score</th>
                    <th className="py-1.5 font-semibold">Threshold</th>
                    <th className="py-1.5 font-semibold">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {rankedRegions.map((r) => (
                    <tr key={r.region} className="border-b border-slate-200">
                      <td className="py-1.5 font-medium text-slate-800">{r.label}</td>
                      <td className="py-1.5 font-semibold text-slate-800">
                        {(r.anomalyScore * 100).toFixed(1)}%
                      </td>
                      <td className="py-1.5 text-slate-600">
                        {r.threshold != null
                          ? `${(r.threshold * 100).toFixed(1)}%`
                          : "—"}
                      </td>
                      <td className="py-1.5 text-slate-600">
                        {r.flagged
                          ? "Claimed"
                          : r.anomalyScore > 0.65
                            ? "High"
                            : r.anomalyScore > 0.35
                              ? "Moderate"
                              : "Low"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </ReportSection>

            {/* 4. Semantic explanation */}
            <ReportSection title="4. Semantic Explanation" last>
              <div className="mb-4 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                <ReportGrid
                  rows={[
                    ["Faithfulness score", faithfulnessPct(result)],
                    ["Zones claimed", String(flaggedCount)],
                    [
                      "NL claims",
                      String(result.semantic_explanations.length),
                    ],
                  ]}
                />
                <p className="mt-2 text-[10px] text-slate-500">
                  Faithfulness: share of NL sentences that match their source zone/artifact/confidence numbers.
                </p>
              </div>

              <h4 className="mb-2 text-[11px] font-bold uppercase tracking-wider text-slate-500">
                Sentences
              </h4>
              {result.semantic_explanations.length === 0 ? (
                <p className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-relaxed text-amber-900">
                  {whyZeroZones(result)}
                </p>
              ) : (
                <ul className="mb-4 flex flex-col gap-2">
                  {result.semantic_explanations.map((exp) => (
                    <li
                      key={exp.id}
                      className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-700"
                    >
                      <p className="font-semibold text-slate-800">{exp.title}</p>
                      <p className="mt-1 leading-relaxed">{exp.description}</p>
                      <div className="mt-2 flex flex-wrap gap-2">
                        <span className="rounded-md bg-slate-100 px-2 py-0.5 text-[10px] font-semibold text-slate-600">
                          {(exp.score * 100).toFixed(0)}% match
                        </span>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </ReportSection>
          </div>
        </div>

        <div className="flex items-center justify-between gap-4 px-6 py-5 sm:px-8">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <FileText className="h-4 w-4" />
            Report ready · {result.report.caseId}
          </div>
          <Button onClick={() => generateReportPdf(result)}>
            <Download className="h-4 w-4" /> Download PDF Report
          </Button>
        </div>
      </GlassCard>
    </motion.section>
  );
}

function ReportSection({
  title,
  children,
  last,
}: {
  title: string;
  children: React.ReactNode;
  last?: boolean;
}) {
  return (
    <div className={last ? "pt-5" : "border-b border-slate-200 py-5"}>
      <h3 className="mb-3 text-xs font-bold uppercase tracking-wider text-blue-700">{title}</h3>
      {children}
    </div>
  );
}

function ReportGrid({ rows }: { rows: [string, string][] }) {
  return (
    <div className="grid grid-cols-1 gap-x-8 gap-y-1.5 text-xs sm:grid-cols-2">
      {rows.map(([label, value]) => (
        <div
          key={label}
          className="flex justify-between gap-4 border-b border-dashed border-slate-200 py-1"
        >
          <span className="text-slate-500">{label}</span>
          <span className="font-medium text-slate-800">{value}</span>
        </div>
      ))}
    </div>
  );
}
