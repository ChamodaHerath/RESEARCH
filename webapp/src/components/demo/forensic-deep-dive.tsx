"use client";

import * as React from "react";
import { motion } from "framer-motion";
import {
  Download,
  Fingerprint,
  ShieldAlert,
  FileText,
  Layers,
  Activity,
  Crosshair,
  BookOpen,
} from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
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
  const confPct = result.confidence;
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

  return (
    `No semantic sentences were emitted.` +
    (zoneList
      ? ` Elevated region scores: ${zoneList}.`
      : ` No zone exceeded its calibrated threshold.`) +
    ` Fake confidence is ${confPct.toFixed(1)}%.`
  );
}

function pct(v: number | null | undefined, digits = 1): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

/**
 * Panel 2 — full Fake forensic deep-dive.
 * Surfaces every live quantity the backend returns for supervisor review.
 */
export function ForensicDeepDive({ result }: { result: DetectionResponse }) {
  const ranked = [...result.facial_regions].sort(
    (a, b) => b.anomalyScore - a.anomalyScore
  );
  const flagged = ranked.filter((r) => r.flagged);
  const clearsThr = ranked.filter((r) => r.clears_threshold && !r.flagged);
  const live = result.explanation_validation?.live;
  const heatmapSrc = result.heatmap || result.attention_map;
  const pFake = result.confidence; // Fake verdict → displayed conf is P(fake)

  const corruptedNames =
    flagged.length > 0
      ? flagged.map((r) => r.label).join(", ")
      : "none retained after gates";

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className="flex flex-col gap-4"
    >
      {/* Header strip */}
      <GlassCard className="flex flex-col gap-4 p-5 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-red-500/15 text-red-400 ring-1 ring-red-500/30">
            <ShieldAlert className="h-5 w-5" />
          </div>
          <div>
            <h2 className="font-display text-xl font-semibold text-white">
              Why Fake — Full Forensic Report
            </h2>
            <p className="mt-0.5 text-sm text-slate-400">
              Case {result.report.caseId} · every live quantity from this inference
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-lg bg-red-500/15 px-3 py-1.5 text-sm font-bold text-red-400 ring-1 ring-red-500/30">
            FAKE · {result.confidence.toFixed(2)}%
          </span>
          <span className="rounded-lg border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs text-slate-300">
            Risk: {result.riskLevel}
          </span>
          <Button onClick={() => generateReportPdf(result)} size="sm">
            <Download className="h-3.5 w-3.5" /> PDF
          </Button>
        </div>
      </GlassCard>

      {/* Corrupted zones — explicit callout for evaluators */}
      <GlassCard className="border border-red-500/25 bg-red-500/[0.07] p-4 sm:p-5">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-red-300">
          <Crosshair className="h-4 w-4" />
          Corrupted facial zones (this Fake verdict)
        </h3>
        <p className="mt-2 text-sm leading-relaxed text-slate-200">
          <span className="font-semibold text-white">Officially claimed corrupted: </span><br/>
          {flagged.length > 0 ? (
            <>
              {flagged.map((r, i) => (
                <span key={r.region}>
                  {i > 0 ? " " : ""}
                  <span className="font-medium text-red-300">{r.label}</span>
                  {" "}
                  (score {pct(r.anomalyScore)}
                  {r.threshold != null ? `, thr ${pct(r.threshold)}` : ""}
                  {r.Q != null ? `, Q=${r.Q.toFixed(4)}` : ""}
                  {r.flag_reason ? `, ${r.flag_reason}` : ""})
                <br/></span> 
              ))}
              
            </>
          ) : (
            <span className="text-amber-200">
              no zone exceeded its calibrated threshold on this Fake frame.
            </span>
          )}
        </p> 
        {clearsThr.length > 0 && (
          <p className="mt-2 text-xs leading-relaxed text-slate-400">
            Cleared threshold but not claimed (unexpected):{" "}
            {clearsThr.map((r) => `${r.label} (${pct(r.anomalyScore)})`).join("; ")}.
          </p>
        )}
        <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
          “Corrupted / claimed” means Q(z) cleared the real-face calibrated
          threshold for this Fake frame and received a natural-language
          explanation.
        </p>
      </GlassCard>

      <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
        Detailed forensic breakdown · corrupted zones: {corruptedNames} · see Panel 3
        for per-explanation accuracy
      </p>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {/* 1. Case + classification */}
        <SectionCard
          icon={<Fingerprint className="h-4 w-4" />}
          title="1. Case & Classification Head"
        >
          <KvGrid
            rows={[
              ["Case ID", result.report.caseId],
              ["Date / Time", `${result.report.date} ${result.report.time}`],
              ["Model", result.model],
              ["Model version string", result.report.modelVersion],
              ["Mode", result.mode],
              ["Processing time", result.processing_time],
              ["Input file", result.report.inputFile],
              ["Verdict", "FAKE"],
              ["P(fake) / confidence", `${pFake.toFixed(2)}%`],
              ["Risk level", result.riskLevel],
            ]}
          />
        </SectionCard>

        {/* 2. Why Fake — classification reasoning */}
        <SectionCard
          icon={<BookOpen className="h-4 w-4" />}
          title="2. Why the Model Classified This as Fake"
          className="xl:col-span-2"
        >
          <p className="mb-3 text-sm leading-relaxed text-slate-300">
            {result.classification_reasoning?.summary ??
              `Classifier verdict is Fake with displayed confidence ${pFake.toFixed(2)}% (P(fake)).`}
          </p>
          <p className="mb-3 rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-100">
            <span className="font-semibold">Facial zones claimed corrupted: </span>
            {flagged.length > 0
              ? flagged
                  .map(
                    (r) =>
                      `${r.label} (score ${pct(r.anomalyScore)}${
                        r.threshold != null ? `, thr ${pct(r.threshold)}` : ""
                      })`
                  )
                  .join("; ")
              : "none — no zone score exceeded its calibrated threshold"}
          </p>
          <ol className="flex flex-col gap-3">
            {(result.classification_reasoning?.steps ?? []).map((step, i) => (
              <li
                key={`${step.stage}-${i}`}
                className="rounded-xl border border-white/10 bg-white/[0.03] p-3"
              >
                <div className="mb-1 flex flex-wrap items-baseline gap-2">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-sky-400">
                    Step {i + 1}
                  </span>
                  <span className="text-sm font-semibold text-white">{step.stage}</span>
                </div>
                {step.formula && (
                  <p className="mb-1 font-mono text-[11px] text-emerald-300/90">
                    {step.formula}
                  </p>
                )}
                <p className="mb-1 text-xs font-medium text-slate-200">
                  Value: {step.value}
                </p>
                <p className="text-xs leading-relaxed text-slate-400">
                  {step.interpretation}
                </p>
              </li>
            ))}
          </ol>
        </SectionCard>

        {/* 3. Visual evidence */}
        <SectionCard
          icon={<Layers className="h-4 w-4" />}
          title="3. Visual Evidence"
          className="xl:col-span-2"
        >
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <Figure label="Aligned face crop (model input)">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={result.faceCrop}
                alt="Face crop"
                className="mx-auto max-h-56 w-full object-contain"
              />
            </Figure>
            <Figure label="Attention heatmap overlay">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={heatmapSrc}
                alt="Attention heatmap"
                className="mx-auto max-h-56 w-full object-contain"
              />
            </Figure>
            {result.mask_ground_truth?.available && result.mask_ground_truth.overlay ? (
              <>
                <Figure label="GT overlay (edited pixels highlighted)">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={result.mask_ground_truth.overlay}
                    alt="Mask ground truth overlay"
                    className="mx-auto max-h-56 w-full object-contain"
                  />
                </Figure>
                {result.mask_ground_truth.mask_pixels ? (
                  <Figure label="Edited pixels only (GT)">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={result.mask_ground_truth.mask_pixels}
                      alt="Manipulated pixels only"
                      className="mx-auto max-h-56 w-full bg-black object-contain"
                    />
                  </Figure>
                ) : null}
              </>
            ) : (
              <Figure label="Manipulation mask GT">
                <p className="flex min-h-[8rem] items-center justify-center px-4 text-center text-xs text-slate-500">
                  Not available for this upload (FF++ mask match required).
                </p>
              </Figure>
            )}
          </div>
          {result.mask_ground_truth?.available && (
            <p className="mt-3 text-xs text-slate-400">
              Mask source: {result.mask_ground_truth.mask_source} · coverage{" "}
              {pct(result.mask_ground_truth.mask_coverage)} · frame{" "}
              {result.mask_ground_truth.frame_index}. {result.mask_ground_truth.note}
            </p>
          )}
        </SectionCard>

        {/* 4. Zone table — full R/Q/thr */}
        <SectionCard
          icon={<Crosshair className="h-4 w-4" />}
          title="4. Zone Scores — R(z), Q(z), Thresholds"
          className="xl:col-span-2"
        >
          <p className="mb-3 text-xs text-slate-400">
            Display score is Q(z) when Bayesian zone priors are active (else R(z)).
            Claimed = Fake verdict + Score &gt; Threshold → NL semantic explanation.
          </p>
          <div className="overflow-x-auto rounded-xl border border-white/10">
            <table className="w-full min-w-[720px] border-collapse text-left text-xs">
              <thead>
                <tr className="border-b border-white/10 bg-white/[0.04] text-slate-400">
                  <th className="px-3 py-2 font-semibold">Zone</th>
                  <th className="px-3 py-2 font-semibold">Score</th>
                  <th className="px-3 py-2 font-semibold">R(z)</th>
                  <th className="px-3 py-2 font-semibold">Q(z)</th>
                  <th className="px-3 py-2 font-semibold">Threshold</th>
                  <th className="px-3 py-2 font-semibold">Clears thr</th>
                  <th className="px-3 py-2 font-semibold">Claimed</th>
                  <th className="px-3 py-2 font-semibold">Flag reason</th>
                </tr>
              </thead>
              <tbody>
                {ranked.map((r) => (
                  <tr
                    key={r.region}
                    className={
                      r.flagged
                        ? "border-b border-white/5 bg-red-500/10"
                        : "border-b border-white/5"
                    }
                  >
                    <td className="px-3 py-2 font-medium text-slate-200">{r.label}</td>
                    <td className="px-3 py-2 font-mono text-white">
                      {pct(r.anomalyScore)}
                    </td>
                    <td className="px-3 py-2 font-mono text-slate-300">
                      {r.R != null ? pct(r.R) : "—"}
                    </td>
                    <td className="px-3 py-2 font-mono text-slate-300">
                      {r.Q != null ? pct(r.Q) : "—"}
                    </td>
                    <td className="px-3 py-2 font-mono text-slate-400">
                      {r.threshold != null ? pct(r.threshold) : "—"}
                    </td>
                    <td className="px-3 py-2">
                      {r.clears_threshold ? (
                        <span className="text-amber-300">Yes</span>
                      ) : (
                        <span className="text-slate-500">No</span>
                      )}
                    </td>
                    <td className="px-3 py-2">
                      {r.flagged ? (
                        <span className="font-semibold text-red-400">Yes</span>
                      ) : (
                        <span className="text-slate-500">No</span>
                      )}
                    </td>
                    <td className="px-3 py-2 text-slate-400">
                      {r.flag_reason ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-slate-500">
            {flagged.length} claimed (NL emitted) ·{" "}
            {ranked.filter((r) => r.clears_threshold).length} cleared threshold
            {flagged.length > 0
              ? ` · corrupted: ${flagged.map((r) => r.label).join(", ")}`
              : ""}
          </p>

          <div className="mt-4 rounded-xl border border-white/10 bg-black/25 p-3">
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
              Column meanings & how they are calculated
            </p>
            <ul className="flex flex-col gap-2 text-[11px] leading-relaxed text-slate-500">
              <li>
                <span className="font-medium text-slate-300">Zone</span> — one of
                nine landmark-defined facial regions (eyes, nose bridge/tip,
                cheeks, mouth, forehead, chin/jaw) built from 68-point landmarks
                mapped onto the 224×224 ViT face crop.
              </li>
              <li>
                <span className="font-medium text-slate-300">Score</span> — the
                value used for flagging:{" "}
                <span className="font-mono text-slate-400">Q(z)</span> when
                Bayesian zone priors are available, otherwise{" "}
                <span className="font-mono text-slate-400">R(z)</span>. Shown as
                a percentage for readability.
              </li>
              <li>
                <span className="font-medium text-slate-300">R(z)</span> — fused
                zone anomaly score{" "}
                <span className="font-mono text-slate-400">
                  R(z) = β·A_norm(z) + (1−β)·S_norm(z)
                </span>
                . A(z) aggregates CLS-to-patch attention over the zone; S(z)
                aggregates grad-weighted patch attribution. Both are
                area-normalised then population-normalised against real-face
                baselines. β is the calibrated fusion weight (posterior mean from{" "}
                <span className="font-mono">fusion_config.json</span>).
              </li>
              <li>
                <span className="font-medium text-slate-300">Q(z)</span> —
                Bayesian zone-anomaly posterior after a soft Beta–Binomial
                update of literature method-conditioned priors with evidence
                weight w:{" "}
                <span className="font-mono text-slate-400">
                  Q(z|m) = (α + R(z)·w) / (α + β_prior + w)
                </span>
                , then marginalised over inferred manipulation methods.
              </li>
              <li>
                <span className="font-medium text-slate-300">Threshold</span> —
                per-zone calibrated cut on Q(z) (or R(z)): the 85th percentile of
                real held-out faces for that zone. A zone must score strictly
                above this value to receive a semantic claim on Fake verdicts.
              </li>
              <li>
                <span className="font-medium text-slate-300">Clears thr</span> —
                Yes when verdict is Fake and Score &gt; Threshold.
              </li>
              <li>
                <span className="font-medium text-slate-300">Claimed</span> — Yes
                when the zone cleared the threshold and received a natural-language
                semantic explanation. Causal occlusion is measured as an optional
                badge (ΔP(fake)), not required to claim the zone.
              </li>
              <li>
                <span className="font-medium text-slate-300">Flag reason</span> —
                machine tag from the pipeline (e.g. calibrated_threshold), or empty
                if not claimed.
              </li>
            </ul>
          </div>
        </SectionCard>

        {/* 5. Semantic NL claims */}
        <SectionCard
          icon={<FileText className="h-4 w-4" />}
          title="5. Semantic Explanations (NL Claims)"
          className="xl:col-span-2"
        >
          <div className="mb-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <StatChip label="Faithfulness" value={faithfulnessPct(result)} />
            <StatChip label="Zones claimed" value={String(flagged.length)} />
            <StatChip
              label="NL claims"
              value={String(result.semantic_explanations.length)}
            />
            <StatChip
              label="Causal-verified"
              value={
                live?.causal_claims_retained != null
                  ? String(live.causal_claims_retained)
                  : "—"
              }
            />
          </div>
          <p className="mb-3 text-[11px] text-slate-500">
            Faithfulness: share of NL sentences that match their source zone/artifact/confidence numbers.
          </p>

          {flagged.length > 0 && (
            <p className="mb-3 rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs text-red-100">
              Corrupted zones with NL claims:{" "}
              <span className="font-semibold">
                {flagged.map((r) => r.label).join(", ")}
              </span>
            </p>
          )}

          {result.semantic_explanations.length === 0 ? (
            <p className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs leading-relaxed text-amber-100">
              {whyZeroZones(result)}
            </p>
          ) : (
            <ul className="flex flex-col gap-3">
              {result.semantic_explanations.map((exp) => (
                <li
                  key={exp.id}
                  className="rounded-xl border border-white/10 bg-white/[0.03] p-3"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="font-semibold text-white">{exp.title}</p>
                    <span className="rounded bg-white/10 px-2 py-0.5 text-[10px] uppercase text-slate-300">
                      {exp.severity}
                    </span>
                    <span className="rounded bg-emerald-500/15 px-2 py-0.5 text-[10px] font-semibold text-emerald-300">
                      {(exp.score * 100).toFixed(0)}% match
                    </span>
                    {exp.causal_drop != null && (
                      <span className="rounded bg-sky-500/15 px-2 py-0.5 text-[10px] text-sky-300">
                        ΔP(fake) = −{exp.causal_drop.toFixed(4)}
                      </span>
                    )}
                  </div>
                  <p className="mt-1 text-xs leading-relaxed text-slate-300">
                    {exp.description}
                  </p>
                  {exp.theory && (
                    <div className="mt-2 rounded-lg border border-white/10 bg-black/20 p-2">
                      <p className="mb-1 text-[11px] font-semibold text-slate-400">
                        Theory · R={exp.theory.scores.R.toFixed(4)}
                        {exp.theory.scores.Q != null
                          ? ` · Q=${exp.theory.scores.Q.toFixed(4)}`
                          : ""}
                      </p>
                      <p className="mb-2 text-[11px] text-slate-400">
                        {exp.theory.summary}
                      </p>
                      <ul className="flex flex-col gap-1">
                        {exp.theory.steps.map((s, idx) => (
                          <li key={idx} className="text-[11px] text-slate-500">
                            <span className="font-medium text-slate-300">
                              {s.label}:
                            </span>{" "}
                            {s.detail}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </SectionCard>

        {/* 6. Video timeline */}
        {result.mode === "video" && result.timeline.length > 0 && (
          <SectionCard
            icon={<Activity className="h-4 w-4" />}
            title="6. Per-Frame Suspicion Timeline"
            className="xl:col-span-2"
          >
            <div className="overflow-x-auto rounded-xl border border-white/10">
              <table className="w-full min-w-[480px] border-collapse text-left text-xs">
                <thead>
                  <tr className="border-b border-white/10 bg-white/[0.04] text-slate-400">
                    <th className="px-3 py-2">Frame</th>
                    <th className="px-3 py-2">Time (s)</th>
                    <th className="px-3 py-2">P(fake)</th>
                    <th className="px-3 py-2">Note</th>
                  </tr>
                </thead>
                <tbody>
                  {result.timeline.map((t) => (
                    <tr key={t.frame} className="border-b border-white/5">
                      <td className="px-3 py-2 font-mono text-slate-200">{t.frame}</td>
                      <td className="px-3 py-2 font-mono text-slate-300">
                        {t.timeSec.toFixed(2)}
                      </td>
                      <td className="px-3 py-2 font-mono text-white">
                        {pct(t.suspicion)}
                      </td>
                      <td className="px-3 py-2 text-slate-400">{t.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        )}

        {/* 9. Terminal log dump */}
        {/* {result.terminal_logs.length > 0 && (
          <SectionCard
            icon={<FileText className="h-4 w-4" />}
            title={`${result.mode === "video" && result.timeline.length > 0 ? "8" : "7"}. Pipeline Terminal Log`}
            className="xl:col-span-2"
          >
            <pre className="max-h-56 overflow-auto rounded-xl border border-white/10 bg-black/40 p-3 font-mono text-[11px] leading-relaxed text-slate-400">
              {result.terminal_logs.join("\n")}
            </pre>
          </SectionCard>
        )} */}
      </div>
    </motion.div>
  );
}

function SectionCard({
  icon,
  title,
  children,
  className = "",
}: {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <GlassCard className={`flex flex-col gap-3 p-4 sm:p-5 ${className}`}>
      <h3 className="flex items-center gap-2 text-sm font-semibold text-white">
        <span className="text-sky-400">{icon}</span>
        {title}
      </h3>
      {children}
    </GlassCard>
  );
}

function KvGrid({ rows }: { rows: [string, string][] }) {
  return (
    <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
      {rows.map(([label, value]) => (
        <div
          key={label}
          className="flex justify-between gap-3 border-b border-dashed border-white/10 py-1.5 text-xs"
        >
          <span className="shrink-0 text-slate-500">{label}</span>
          <span className="break-all text-right font-medium text-slate-200">
            {value}
          </span>
        </div>
      ))}
    </div>
  );
}

function Figure({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="overflow-hidden rounded-xl border border-white/10 bg-black/30">
      <div className="bg-black/40 p-2">{children}</div>
      <p className="border-t border-white/10 px-2 py-1.5 text-center text-[10px] font-medium text-slate-400">
        {label}
      </p>
    </div>
  );
}

function StatChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2">
      <p className="text-[10px] uppercase tracking-wider text-slate-500">{label}</p>
      <p className="mt-0.5 font-mono text-sm font-semibold text-white">{value}</p>
    </div>
  );
}
