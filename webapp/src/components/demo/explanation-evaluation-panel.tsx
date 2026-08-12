"use client";

import { motion } from "framer-motion";
import {
  BadgeCheck,
  CheckCircle2,
  ClipboardCheck,
  Crosshair,
  Scale,
  ShieldCheck,
  Sparkles,
  Target,
  XCircle,
} from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import type {
  DetectionResponse,
  ExplanationValidationClaim,
  SemanticExplanation,
} from "@/lib/types";

function pct(v: number | null | undefined, digits = 1): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

type ClaimEval = {
  explanation: SemanticExplanation;
  validation: ExplanationValidationClaim | null;
  region: DetectionResponse["facial_regions"][number] | null;
};

function buildClaimEvals(result: DetectionResponse): ClaimEval[] {
  const live = result.explanation_validation?.live;
  const byZone = new Map(
    (live?.per_claim ?? []).map((c) => [c.zone, c] as const)
  );
  const regionById = new Map(result.facial_regions.map((r) => [r.region, r]));

  return result.semantic_explanations.map((explanation) => ({
    explanation,
    validation: byZone.get(explanation.id) ?? null,
    region: regionById.get(explanation.id) ?? null,
  }));
}

function overallVerdict(evals: ClaimEval[]) {
  if (evals.length === 0) {
    return {
      tone: "amber" as const,
      title: "No explanations to evaluate",
      detail:
        "The Fake verdict did not produce any natural-language zone claims on this file.",
    };
  }
  const faithfulCount = evals.filter(
    (e) => e.validation?.faithfulness_passed !== false
  ).length;
  const thresholdCount = evals.filter((e) => e.region?.flagged).length;
  const allFaithful = faithfulCount === evals.length;
  const allThreshold = thresholdCount === evals.length;

  if (allFaithful && allThreshold) {
    return {
      tone: "emerald" as const,
      title: "All explanations passed automated accuracy checks",
      detail: `${evals.length} claim(s): text matches pipeline numbers and each zone cleared its calibrated threshold.`,
    };
  }
  return {
    tone: "amber" as const,
    title: "Partial verification",
    detail: `${faithfulCount}/${evals.length} text-accurate · ${thresholdCount}/${evals.length} threshold-backed.`,
  };
}

export function ExplanationEvaluationPanel({
  result,
}: {
  result: DetectionResponse;
}) {
  const live = result.explanation_validation?.live;
  const maskVal = result.explanation_validation?.mask_ground_truth_validation;
  const hasUploadMask = Boolean(
    result.mask_ground_truth?.available && result.mask_ground_truth.overlay
  );
  const claimedVisual = result.claimed_zones_visual;
  const hasClaimedZones = Boolean(
    claimedVisual?.available && claimedVisual.zone_pixels
  );
  const showPixelCompare = hasUploadMask || hasClaimedZones;
  const evals = buildClaimEvals(result);
  const verdict = overallVerdict(evals);
  const faithfulPct = live
    ? Math.round(live.faithfulness_pass_rate * 100)
    : evals.length > 0
      ? 100
      : null;

  const toneBorder = {
    emerald: "border-emerald-500/30 bg-emerald-500/10",
    amber: "border-amber-500/30 bg-amber-500/10",
  }[verdict.tone];

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className="flex flex-col gap-4"
    >
      <GlassCard className="flex flex-col gap-4 p-5 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-500/15 text-violet-400 ring-1 ring-violet-500/30">
            <ClipboardCheck className="h-5 w-5" />
          </div>
          <div>
            <h2 className="font-display text-xl font-semibold text-white">
              Explanation Evaluation
            </h2>
            <p className="mt-0.5 text-sm text-slate-400">
              Case {result.report.caseId} · automated accuracy audit per claim
            </p>
          </div>
        </div>
        <span className="rounded-lg border border-violet-500/30 bg-violet-500/10 px-3 py-1.5 text-xs font-medium text-violet-300">
          Panel 3 · Verification
        </span>
      </GlassCard>

      <GlassCard className={`border p-4 sm:p-5 ${toneBorder}`}>
        <p className="flex items-center gap-2 text-sm font-semibold text-white">
          {verdict.tone === "emerald" ? (
            <ShieldCheck className="h-4 w-4 text-emerald-400" />
          ) : (
            <Sparkles className="h-4 w-4 text-amber-400" />
          )}
          {verdict.title}
        </p>
        <p className="mt-1 text-sm text-slate-300">{verdict.detail}</p>
      </GlassCard>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <SummaryStat label="Claims evaluated" value={String(evals.length)} />
        <SummaryStat
          label="Text accuracy"
          value={faithfulPct != null ? `${faithfulPct}%` : "—"}
        />
        <SummaryStat
          label="Threshold-backed"
          value={`${evals.filter((e) => e.region?.flagged).length}/${evals.length || "—"}`}
        />
        <SummaryStat
          label="Causal influence"
          value={
            live?.causal_zones_tested
              ? `${live.causal_claims_retained}/${live.causal_zones_tested}`
              : "—"
          }
        />
      </div>

      <GlassCard className="flex flex-col gap-4 p-5">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-white">
          <BadgeCheck className="h-4 w-4 text-emerald-400" />
          Per-explanation accuracy (this upload)
        </h3>
        <p className="text-xs leading-relaxed text-slate-400">
          Each card shows whether the written sentence is faithful to the pipeline
          numbers and whether the zone had enough evidence to be claimed.
        </p>

        {evals.length === 0 ? (
          <p className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-100">
            No semantic claims were emitted — switch to Panel 2 to inspect zone
            scores and thresholds.
          </p>
        ) : (
          <ul className="flex flex-col gap-3">
            {evals.map(({ explanation, validation, region }, i) => (
              <ClaimAccuracyCard
                key={explanation.id}
                index={i + 1}
                explanation={explanation}
                validation={validation}
                region={region}
                causalMinDrop={live?.causal_min_drop}
              />
            ))}
          </ul>
        )}
      </GlassCard>

      {showPixelCompare && (
        <GlassCard className="flex flex-col gap-4 p-5">
          <h3 className="flex items-center gap-2 text-sm font-semibold text-white">
            <Target className="h-4 w-4 text-emerald-400" />
            Pixel-level comparison
          </h3>
          <p className="text-xs leading-relaxed text-slate-400">
            Side-by-side view on a black background: green = FaceForensics++ edited
            pixels (ground truth) · red = pixels inside the semantic zones the
            system explained. Overlap between the two panels indicates the
            explanation points where editing actually occurred.
          </p>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {result.mask_ground_truth?.mask_pixels ? (
              <GtFigure
                label="Ground truth — edited pixels"
                src={result.mask_ground_truth.mask_pixels}
                dark
                accent="emerald"
              />
            ) : (
              <ComparePlaceholder label="Ground truth unavailable for this upload" />
            )}
            {claimedVisual?.zone_pixels ? (
              <GtFigure
                label="System — explained zone pixels"
                src={claimedVisual.zone_pixels}
                dark
                accent="red"
              />
            ) : (
              <ComparePlaceholder label="No semantic zone claims to rasterise" />
            )}
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500">
            {hasUploadMask && result.mask_ground_truth?.mask_coverage != null ? (
              <span>
                GT edited area: {pct(result.mask_ground_truth.mask_coverage)} of
                face crop
                {result.mask_ground_truth.mask_source
                  ? ` · ${result.mask_ground_truth.mask_source}`
                  : ""}
              </span>
            ) : null}
            {hasClaimedZones && claimedVisual?.zone_coverage != null ? (
              <span>
                Claimed zones: {claimedVisual.zone_labels.join(", ")} ·{" "}
                {pct(claimedVisual.zone_coverage)} of face crop
              </span>
            ) : null}
          </div>
          {hasUploadMask && hasClaimedZones ? (
            <ol className="list-decimal space-y-2 pl-5 text-sm text-slate-300">
              <li>Compare the green (GT) and red (system) silhouettes.</li>
              <li>Strong overlap means the explanation localises the true edit.</li>
              <li>
                Mismatch suggests the model flagged a different facial region than
                was manipulated.
              </li>
            </ol>
          ) : null}
        </GlassCard>
      )}

      {!showPixelCompare && (
        <GlassCard className="border border-white/10 p-4">
          <p className="flex items-center gap-2 text-sm text-slate-400">
            <Crosshair className="h-4 w-4 shrink-0" />
            No pixel-level ground truth or zone rasterisation for this upload.
            Per-claim text and threshold checks above still apply.
          </p>
        </GlassCard>
      )}

      {maskVal && (
        <GlassCard className="flex flex-col gap-4 p-5">
          <h3 className="flex items-center gap-2 text-sm font-semibold text-white">
            <Scale className="h-4 w-4 text-amber-400" />
            Research benchmark (FaceForensics++ test set)
          </h3>
          <p className="text-xs text-slate-400">
            {maskVal.scope_note ??
              "Population metrics from offline evaluation — not a score for this single file alone."}
          </p>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <BenchmarkStat
              label="Flag precision"
              value={maskVal.precision != null ? pct(maskVal.precision, 1) : "—"}
              hint="When the system flags a zone, how often it overlaps true edited pixels."
            />
            <BenchmarkStat
              label="Pointing accuracy"
              value={
                maskVal.pointing_game_accuracy != null
                  ? pct(maskVal.pointing_game_accuracy, 1)
                  : "—"
              }
              hint="How often the top-scored zone hits a genuinely edited region."
            />
            <BenchmarkStat
              label="Test frames"
              value={
                maskVal.n_frames_evaluated != null
                  ? String(maskVal.n_frames_evaluated)
                  : "—"
              }
              hint="Held-out FF++ frames used for these statistics."
            />
          </div>
        </GlassCard>
      )}
    </motion.div>
  );
}

function ClaimAccuracyCard({
  index,
  explanation,
  validation,
  region,
  causalMinDrop,
}: {
  index: number;
  explanation: SemanticExplanation;
  validation: ExplanationValidationClaim | null;
  region: DetectionResponse["facial_regions"][number] | null;
  causalMinDrop: number | null | undefined;
}) {
  const faithful = validation?.faithfulness_passed ?? true;
  const thresholdOk = region?.flagged === true;
  const scorePct = (explanation.score * 100).toFixed(0);
  const thrPct =
    region?.threshold != null ? (region.threshold * 100).toFixed(1) : null;
  const displayScore =
    region != null ? (region.anomalyScore * 100).toFixed(1) : null;

  const causalDrop = validation?.causal_drop ?? explanation.causal_drop;
  const causalTested = validation?.causal_passed != null;
  const causalOk = validation?.causal_passed === true;

  const checks = [
    {
      label: "Text matches pipeline",
      detail: faithful
        ? "Zone, artifact, and confidence in the sentence match the ConceptVector."
        : "Automated check found a mismatch between text and source numbers.",
      ok: faithful,
      required: true,
    },
    {
      label: "Zone cleared threshold",
      detail:
        thresholdOk && displayScore && thrPct
          ? `Score ${displayScore}% > threshold ${thrPct}% — sufficient evidence to claim this zone.`
          : "Zone did not meet the calibrated threshold for a claim.",
      ok: thresholdOk,
      required: true,
    },
    {
      label: "Causal influence",
      detail:
        causalDrop != null && causalMinDrop != null
          ? causalOk
            ? `Hiding this zone dropped P(fake) by ${causalDrop.toFixed(4)} (≥ ${causalMinDrop}).`
            : `Hiding this zone changed P(fake) by ${causalDrop.toFixed(4)} (needs ≥ ${causalMinDrop} to pass).`
          : "Influence test not available for this claim.",
      ok: causalOk,
      required: false,
      skipped: !causalTested,
    },
  ];

  const requiredPass = checks.filter((c) => c.required && c.ok).length;
  const requiredTotal = checks.filter((c) => c.required).length;
  const cardOk = requiredPass === requiredTotal && faithful;

  return (
    <li
      className={
        cardOk
          ? "rounded-xl border border-emerald-500/25 bg-emerald-500/[0.06] p-4"
          : "rounded-xl border border-white/10 bg-white/[0.03] p-4"
      }
    >
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Claim {index}
          </p>
          <h4 className="text-sm font-semibold text-white">{explanation.title}</h4>
        </div>
        <span
          className={
            cardOk
              ? "rounded-full bg-emerald-500/20 px-2.5 py-0.5 text-[10px] font-bold uppercase text-emerald-300"
              : "rounded-full bg-amber-500/20 px-2.5 py-0.5 text-[10px] font-bold uppercase text-amber-300"
          }
        >
          {cardOk ? "Verified" : "Review"}
        </span>
      </div>

      <p className="mb-3 text-xs leading-relaxed text-slate-300">
        {explanation.description}
      </p>

      <ul className="flex flex-col gap-2">
        {checks.map((c) => (
          <CheckRow key={c.label} {...c} />
        ))}
      </ul>

      <p className="mt-3 text-[10px] text-slate-500">
        Match score {scorePct}%
        {explanation.theory?.scores.Q != null
          ? ` · Q=${explanation.theory.scores.Q.toFixed(4)}`
          : ""}
        {explanation.theory?.scores.R != null
          ? ` · R=${explanation.theory.scores.R.toFixed(4)}`
          : ""}
      </p>
    </li>
  );
}

function CheckRow({
  label,
  detail,
  ok,
  required,
  skipped,
}: {
  label: string;
  detail: string;
  ok: boolean;
  required: boolean;
  skipped?: boolean;
}) {
  return (
    <li className="flex gap-2 rounded-lg border border-white/[0.06] bg-black/20 px-3 py-2">
      {skipped ? (
        <span className="mt-0.5 text-slate-500">—</span>
      ) : ok ? (
        <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
      ) : required ? (
        <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-400" />
      ) : (
        <span className="mt-0.5 h-4 w-4 shrink-0 rounded-full bg-amber-500/30" />
      )}
      <div>
        <p className="text-xs font-medium text-slate-200">
          {label}
          {!required && (
            <span className="ml-1 text-[10px] font-normal text-slate-500">optional</span>
          )}
        </p>
        <p className="text-[11px] leading-relaxed text-slate-400">{detail}</p>
      </div>
    </li>
  );
}

function SummaryStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] px-3 py-3">
      <p className="text-[10px] uppercase tracking-wider text-slate-500">{label}</p>
      <p className="mt-1 font-mono text-lg font-bold text-white">{value}</p>
    </div>
  );
}

function BenchmarkStat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div className="rounded-lg border border-white/10 bg-black/20 px-3 py-2">
      <p className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
        {label}
      </p>
      <p className="mt-1 font-mono text-xl font-bold text-white">{value}</p>
      <p className="mt-1 text-[11px] leading-relaxed text-slate-400">{hint}</p>
    </div>
  );
}

function GtFigure({
  label,
  src,
  dark,
  accent,
}: {
  label: string;
  src: string;
  dark?: boolean;
  accent?: "emerald" | "red";
}) {
  const accentBorder =
    accent === "emerald"
      ? "border-emerald-500/30"
      : accent === "red"
        ? "border-red-500/30"
        : "border-white/10";
  return (
    <div
      className={`overflow-hidden rounded-xl border ${accentBorder} ${dark ? "bg-black" : "bg-black/30"}`}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={label} className="mx-auto aspect-square w-full object-contain" />
      <p className="border-t border-white/10 px-2 py-1.5 text-center text-[10px] text-slate-400">
        {label}
      </p>
    </div>
  );
}

function ComparePlaceholder({ label }: { label: string }) {
  return (
    <div className="flex aspect-square flex-col items-center justify-center rounded-xl border border-dashed border-white/15 bg-black/40 px-4 text-center">
      <Crosshair className="mb-2 h-5 w-5 text-slate-600" />
      <p className="text-[11px] leading-relaxed text-slate-500">{label}</p>
    </div>
  );
}
