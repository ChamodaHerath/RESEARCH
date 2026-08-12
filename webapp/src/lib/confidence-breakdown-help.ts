/**
 * Plain-language descriptions for pipeline confidence_breakdown metrics
 * returned by inference_server.build_confidence_breakdown().
 */
export function getConfidenceMetricHelp(metric: string): string | undefined {
  const normalized = metric.trim();

  if (normalized.startsWith("Mean Fused Zone Anomaly")) {
    return (
      "Average fused zone score R̄(z) across all 9 facial regions. " +
      "Each R(z) combines normalized attention A(z) and gradient attribution S(z): " +
      "R(z) = β·A_norm(z) + (1−β)·S_norm(z). Higher values mean stronger " +
      "overall spatial anomaly on this frame."
    );
  }

  if (normalized.startsWith("Calibrated Fusion Weight β")) {
    return (
      "Calibrated β from fusion_config.json (posterior mean): how much weight " +
      "the pipeline gives to CLS-to-patch attention versus grad-weighted " +
      "patch attribution when fusing zone scores. Shown as attention's share " +
      "in percent."
    );
  }

  if (normalized === "Claimed Zone Coverage") {
    return (
      "Fraction of the 9 SAEDF zones that cleared the per-zone calibrated " +
      "threshold and received a natural-language forensic claim on this " +
      "Fake verdict (claimed zones ÷ 9)."
    );
  }

  if (normalized.startsWith("Inferred Manipulation Method Confidence")) {
    return (
      "Confidence in the most likely deepfake manipulation family inferred from " +
      "the R(z) zone pattern (cosine similarity to calibrated method signatures, " +
      "then Bayesian zone-prior update). The method name in parentheses is the " +
      "top hypothesis for this frame."
    );
  }

  if (normalized === "Explanation Faithfulness Pass Rate") {
    return (
      "Share of emitted NL sentences that pass automated faithfulness checks: " +
      "the text must match its source ConceptVector (zone, artifact type, and " +
      "reported confidence numbers)."
    );
  }

  if (normalized.startsWith("Causal-verified Claims")) {
    return (
      "Share of claimed zones where blanking the region reduced P(fake) by at " +
      "least the causal minimum drop (default 3%). This is an optional sanity " +
      "badge in the live demo — threshold-clearing claims are still shown even " +
      "when causal verification fails."
    );
  }

  return undefined;
}
