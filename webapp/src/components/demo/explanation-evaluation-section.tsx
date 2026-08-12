"use client";

import type { ReactNode } from "react";
import { CheckCircle2, HelpCircle, Scale, Sparkles, Target } from "lucide-react";
import type { DetectionResponse } from "@/lib/types";

function pct(v: number | null | undefined, digits = 1): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function ExplanationEvaluationSection({
  result,
}: {
  result: DetectionResponse;
}) {
  const live = result.explanation_validation?.live;
  const maskVal = result.explanation_validation?.mask_ground_truth_validation;
  const hasUploadMask = Boolean(
    result.mask_ground_truth?.available && result.mask_ground_truth.overlay
  );
  const claimed = result.facial_regions.filter((r) => r.flagged);
  const claimedLabels =
    claimed.length > 0
      ? claimed.map((r) => r.label).join(", ")
      : "None — no facial zone was flagged on this image.";

  const faithfulPct = live
    ? Math.round(live.faithfulness_pass_rate * 100)
    : result.semantic_explanations.length > 0
      ? 100
      : null;

  const causalTested = live?.causal_zones_tested ?? 0;
  const causalPassed = live?.causal_claims_retained ?? 0;

  return (
    <div className="flex flex-col gap-4">
      <p className="text-sm leading-relaxed text-slate-300">
        This section checks whether the system&apos;s written explanations are
        trustworthy and, when a reference mask exists, how they compare to known
        edited pixels.
      </p>

      {/* This upload: system vs reference */}
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <EvalCard
          icon={<Sparkles className="h-4 w-4 text-sky-400" />}
          title="What the system reported (this file)"
          tone="sky"
        >
          <ul className="flex flex-col gap-2 text-sm text-slate-300">
            <li>
              <span className="font-medium text-white">Verdict:</span> Fake (
              {result.confidence.toFixed(1)}% confidence)
            </li>
            <li>
              <span className="font-medium text-white">Zones flagged as edited:</span>{" "}
              {claimedLabels}
            </li>
            <li>
              <span className="font-medium text-white">Written explanations:</span>{" "}
              {result.semantic_explanations.length}
            </li>
          </ul>
        </EvalCard>

        <EvalCard
          icon={<Target className="h-4 w-4 text-emerald-400" />}
          title="Reference: where editing really happened"
          tone="emerald"
        >
          {hasUploadMask ? (
            <ul className="flex flex-col gap-2 text-sm text-slate-300">
              <li>
                A <span className="font-medium text-emerald-300">green overlay</span>{" "}
                is available for this upload (see Visual Evidence). Green pixels =
                known edited area from the FaceForensics++ dataset.
              </li>
              <li>
                <span className="font-medium text-white">Edited area size:</span>{" "}
                {pct(result.mask_ground_truth?.mask_coverage)} of the face crop
              </li>
              <li className="text-xs text-slate-400">
                Compare: do the zones the system named overlap the green regions?
                Strong overlap suggests the explanation points to the right part of
                the face.
              </li>
            </ul>
          ) : (
            <p className="text-sm leading-relaxed text-slate-400">
              No reference mask for this upload. Ground-truth comparison is only
              possible for FaceForensics++ clips that match a stored manipulation
              mask. You can still read the system&apos;s flagged zones and
              explanations above.
            </p>
          )}
        </EvalCard>
      </div>

      {hasUploadMask && claimed.length > 0 && (
        <div className="rounded-xl border border-emerald-500/25 bg-emerald-500/5 p-4">
          <p className="mb-2 flex items-center gap-2 text-sm font-semibold text-emerald-200">
            <Scale className="h-4 w-4" />
            Quick visual evaluation for this image
          </p>
          <ol className="flex list-decimal flex-col gap-2 pl-5 text-sm leading-relaxed text-slate-300">
            <li>
              Open <span className="font-medium text-white">Visual Evidence</span>{" "}
              and look at the green mask panel.
            </li>
            <li>
              Check each zone the system flagged:{" "}
              <span className="font-medium text-red-300">{claimedLabels}</span>
            </li>
            <li>
              If green editing marks sit inside those zones, the system&apos;s story
              matches the dataset record. If green appears elsewhere but was not
              flagged, the system may have missed part of the edit.
            </li>
          </ol>
        </div>
      )}

      {/* Automatic checks on this run */}
      <EvalCard
        icon={<CheckCircle2 className="h-4 w-4 text-violet-400" />}
        title="Automatic quality checks (this run only)"
        tone="violet"
      >
        {!live && result.semantic_explanations.length === 0 ? (
          <p className="text-sm text-slate-400">
            No explanations were emitted, so there is nothing to validate on this
            run.
          </p>
        ) : (
          <ul className="flex flex-col gap-3 text-sm text-slate-300">
            <li className="flex gap-2">
              <StatusDot ok={faithfulPct === 100} partial={faithfulPct != null && faithfulPct > 0 && faithfulPct < 100} />
              <span>
                <span className="font-medium text-white">Explanation text accuracy:</span>{" "}
                {faithfulPct != null
                  ? `${faithfulPct}% of sentences match the numbers the pipeline computed (zone, artifact, score).`
                  : "Not measured."}
              </span>
            </li>
            {live?.causal_gate_enabled && causalTested > 0 ? (
              <li className="flex gap-2">
                <StatusDot ok={causalPassed > 0} partial={causalPassed === 0} />
                <span>
                  <span className="font-medium text-white">Influence test (optional):</span>{" "}
                  When we temporarily hide a flagged zone and re-score the face,{" "}
                  {causalPassed} of {causalTested} zones reduced the fake score enough
                  to count as &quot;driving&quot; the decision. This is extra evidence,
                  not required to show an explanation.
                </span>
              </li>
            ) : (
              <li className="flex gap-2">
                <HelpCircle className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" />
                <span className="text-slate-400">
                  Influence test not run or no flagged zones to test on this file.
                </span>
              </li>
            )}
          </ul>
        )}
      </EvalCard>

      {/* Benchmark population stats */}
      {maskVal && (
        <EvalCard
          icon={<Scale className="h-4 w-4 text-amber-400" />}
          title="Research benchmark (many FF++ test videos — not just this upload)"
          tone="amber"
        >
          <p className="mb-3 text-xs text-slate-400">
            {maskVal.scope_note ??
              "These numbers come from offline evaluation on a held-out benchmark set."}
          </p>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <BenchmarkStat
              label="When the system flags a zone"
              value={
                maskVal.precision != null ? pct(maskVal.precision, 1) : "—"
              }
              hint="How often a flagged zone truly overlaps edited pixels (precision)."
            />
            <BenchmarkStat
              label="Top flagged zone hits edit"
              value={
                maskVal.pointing_game_accuracy != null
                  ? pct(maskVal.pointing_game_accuracy, 1)
                  : "—"
              }
              hint="How often the highest-scoring zone lands on a genuinely edited region."
            />
            <BenchmarkStat
              label="Frames in benchmark"
              value={
                maskVal.n_frames_evaluated != null
                  ? String(maskVal.n_frames_evaluated)
                  : "—"
              }
              hint="Size of the offline test set used for these statistics."
            />
          </div>
        </EvalCard>
      )}
    </div>
  );
}

function EvalCard({
  icon,
  title,
  tone,
  children,
}: {
  icon: ReactNode;
  title: string;
  tone: "sky" | "emerald" | "violet" | "amber";
  children: ReactNode;
}) {
  const border: Record<string, string> = {
    sky: "border-sky-500/20 bg-sky-500/5",
    emerald: "border-emerald-500/20 bg-emerald-500/5",
    violet: "border-violet-500/20 bg-violet-500/5",
    amber: "border-amber-500/20 bg-amber-500/5",
  };
  return (
    <div className={`rounded-xl border p-4 ${border[tone]}`}>
      <p className="mb-3 flex items-center gap-2 text-sm font-semibold text-white">
        {icon}
        {title}
      </p>
      {children}
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

function StatusDot({
  ok,
  partial,
}: {
  ok: boolean;
  partial?: boolean;
}) {
  const color = ok
    ? "bg-emerald-400"
    : partial
      ? "bg-amber-400"
      : "bg-slate-500";
  return (
    <span
      className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${color}`}
      aria-hidden
    />
  );
}
