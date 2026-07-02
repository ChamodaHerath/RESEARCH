"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { FileText, Download, ShieldCheck, ShieldAlert, Fingerprint } from "lucide-react";
import { GlassCard, SectionBadge } from "@/components/ui/glass-card";
import { Button } from "@/components/ui/button";
import { generateReportPdf } from "@/lib/generate-report-pdf";
import type { DetectionResponse } from "@/lib/types";

export function ForensicReport({ result }: { result: DetectionResponse }) {
  const isFake = result.prediction === "Fake";
  const color = isFake ? "#ef4444" : "#22c55e";

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
          A structured, auditable case report suitable for chain-of-custody documentation and
          expert review.
        </p>
      </div>

      <GlassCard glow className="mx-auto max-w-4xl overflow-hidden !p-0">
        {/* Report "paper" preview */}
        <div className="bg-[#0b1120] p-1">
          <div className="rounded-2xl bg-gradient-to-b from-slate-100 to-slate-200 p-8 text-slate-900 sm:p-10">
            <div className="flex flex-col items-start justify-between gap-4 border-b border-slate-300 pb-6 sm:flex-row sm:items-center">
              <div>
                <div className="flex items-center gap-2 text-lg font-bold text-slate-900">
                  <Fingerprint className="h-5 w-5 text-blue-600" />
                  ExplainDeep Forensic Report
                </div>
                <p className="text-xs text-slate-500">
                  Explainable Deepfake Detection using Vision Transformers
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

            <ReportSection title="2. Prediction Summary">
              <ReportGrid
                rows={[
                  ["Overall Verdict", result.prediction.toUpperCase()],
                  ["Confidence", `${result.confidence.toFixed(2)}%`],
                  ["Risk Level", result.riskLevel],
                ]}
              />
            </ReportSection>

            {result.classification_reasoning && (
              <ReportSection title="2b. Theoretical Basis — Why Real/Fake?">
                <p className="mb-3 text-xs leading-relaxed text-slate-700">
                  {result.classification_reasoning.summary}
                </p>
                <ol className="flex flex-col gap-2">
                  {result.classification_reasoning.steps.map((step) => (
                    <li
                      key={step.stage}
                      className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-700"
                    >
                      <p className="font-semibold text-slate-800">
                        {step.stage}
                        <span className="ml-2 font-mono text-[10px] text-blue-700">
                          {step.value}
                        </span>
                      </p>
                      {step.formula && (
                        <p className="mt-0.5 font-mono text-[10px] text-slate-500">{step.formula}</p>
                      )}
                      <p className="mt-1 leading-relaxed">{step.interpretation}</p>
                    </li>
                  ))}
                </ol>
              </ReportSection>
            )}

            <ReportSection title="3. Evidence — Semantic Explanations">
              <ul className="grid grid-cols-1 gap-3 text-xs text-slate-700">
                {result.semantic_explanations.map((exp) => (
                  <li key={exp.id} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                    <div className="flex items-start gap-1.5">
                      <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-500" />
                      <div>
                        <p>
                          <strong>{exp.title}</strong> — {(exp.score * 100).toFixed(0)}% match
                        </p>
                        <p className="mt-1 text-slate-600">{exp.description}</p>
                        {exp.theory && (
                          <div className="mt-2 border-t border-slate-200 pt-2">
                            <p className="font-semibold text-slate-800">Why this explanation?</p>
                            <p className="mt-1 leading-relaxed">{exp.theory.summary}</p>
                            <ul className="mt-2 flex flex-col gap-1.5">
                              {exp.theory.steps.map((step) => (
                                <li key={step.label} className="leading-relaxed text-slate-600">
                                  <span className="font-medium text-slate-700">{step.label}:</span>{" "}
                                  {step.detail}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            </ReportSection>

            <ReportSection title="4. Attention Analysis / Confidence Table">
              <table className="w-full border-collapse text-xs">
                <thead>
                  <tr className="border-b border-slate-300 text-left text-slate-500">
                    <th className="py-1.5 font-semibold">Metric</th>
                    <th className="py-1.5 font-semibold">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {result.confidence_breakdown.map((c) => (
                    <tr key={c.metric} className="border-b border-slate-200">
                      <td className="py-1.5">{c.metric}</td>
                      <td className="py-1.5 font-semibold text-slate-800">{c.value}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </ReportSection>

            <ReportSection title="5. Recommendations">
              <p className="text-xs leading-relaxed text-slate-700">
                {isFake
                  ? "This media exhibits strong indicators of synthetic manipulation. Treat as unverified, flag for manual forensic review, and avoid redistribution pending independent verification."
                  : "No significant indicators of manipulation were detected. The media is assessed as authentic at the stated confidence level. Standard verification is still recommended for high-stakes use cases."}
              </p>
            </ReportSection>

            <ReportSection title="6. Disclaimer" last>
              <p className="text-[11px] italic leading-relaxed text-slate-500">
                This report is generated by an academic research prototype for demonstration
                purposes as part of a final year research project. It is not a certified legal
                or forensic instrument.
              </p>
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
        <div key={label} className="flex justify-between gap-4 border-b border-dashed border-slate-200 py-1">
          <span className="text-slate-500">{label}</span>
          <span className="font-medium text-slate-800">{value}</span>
        </div>
      ))}
    </div>
  );
}
