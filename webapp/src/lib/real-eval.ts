/**
 * Real evaluation data — imported verbatim (via sync_webapp_data.py, a pure
 * copy/trim step that computes nothing) from the actual pipeline output
 * files produced by evaluate_pipeline.py and recalibrate_v2.py:
 *
 *   outputs/evaluation_results_ffpp.json     -> FaceForensics++ test partition
 *   outputs/evaluation_results_celebdf.json  -> Celeb-DF v2 test partition
 *   configs/fusion_config.json               -> Bayesian beta calibration
 *
 * Detector: hamzenium/ViT-Deepfake-Classifier (see fusion-config.json
 * `model_name`, and README.md "Detector: configurable, not assumed" for the
 * current configured detector and the surrounding calibration/evaluation notes).
 *
 * Nothing in this file is hand-typed or fabricated. Every number is a direct
 * field from those JSON files. The confusion matrix is the pipeline's own
 * measured TP/FP/TN/FN counts (`classification_metrics.confusion_matrix`),
 * not a derived approximation — `deriveConfusionMatrix` below is kept only
 * as a fallback for older eval JSON that predates that field.
 */
import ffppRaw from "@/data/eval-ffpp.json";
import celebdfRaw from "@/data/eval-celebdf.json";
import fusionConfigRaw from "@/data/fusion-config.json";

export interface ConfusionMatrixCounts {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface ClassificationMetrics {
  accuracy: number;
  auc_roc: number;
  average_precision?: number;
  brier_score?: number;
  ece?: number;
  precision: number;
  recall: number;
  f1: number;
  confusion_matrix?: ConfusionMatrixCounts;
  n_total: number;
  n_fake: number;
  n_real: number;
  n_skipped: number;
}

export interface EvaluationMetadata {
  total_frames: number;
  processed_frames: number;
  skipped_frames: number;
  elapsed_seconds: number;
  fps: number;
}

export interface ExplanationMetrics {
  faithfulness_pass_rate: number;
  mean_flagged_zones_per_fake_frame: number;
  zone_flag_frequency: Record<string, number>;
  zone_R_separation: Record<
    string,
    { fake_mean_R: number; real_mean_R: number; separation: number; flag_count: number }
  >;
}

export interface EvalFile {
  evaluation_metadata: EvaluationMetadata;
  classification_metrics: ClassificationMetrics;
  explanation_metrics: ExplanationMetrics;
  bayesian_metrics: {
    beta_posterior: { map: number; mean: number; std: number; ci_lower: number; ci_upper: number };
    mean_CI_width_per_zone: Record<string, number>;
  };
}

export const FFPP_EVAL = ffppRaw as EvalFile;
export const CELEBDF_EVAL = celebdfRaw as EvalFile;
export const FUSION_CONFIG = fusionConfigRaw as {
  model_name: string;
  beta: number;
  beta_posterior: { map: number; mean: number; std: number; ci_lower: number; ci_upper: number };
  severity_thresholds: { high: number; medium: number; low: number };
  zone_thresholds: Record<string, number>;
  zone_thresholds_scale: string;
  max_flagged_zones: number;
  calibration_n_frames: number;
  calibration_n_fake: number;
  calibration_n_real: number;
};

export const DATASETS: { key: string; name: string; eval: EvalFile }[] = [
  { key: "ffpp", name: "FaceForensics++", eval: FFPP_EVAL },
  { key: "celebdf", name: "Celeb-DF v2", eval: CELEBDF_EVAL },
];

/**
 * Return the confusion matrix for a classification-metrics block. Prefers
 * the pipeline's own measured `confusion_matrix` field (present since
 * evaluate_pipeline.py started storing TP/FP/TN/FN directly); falls back to
 * recovering it from precision/recall/n_fake/n_real for older eval JSON that
 * predates that field. The system is over-determined (accuracy is a
 * redundant check), so the fallback recovery is exact, not approximate:
 *
 *   recall    = TP / n_fake        =>  TP = round(recall * n_fake)
 *   precision = TP / (TP + FP)     =>  FP = round(TP / precision - TP)
 *   FN = n_fake - TP
 *   TN = n_real - FP
 */
export function deriveConfusionMatrix(m: ClassificationMetrics): ConfusionMatrixCounts {
  if (m.confusion_matrix) return m.confusion_matrix;
  const tp = Math.round(m.recall * m.n_fake);
  const fn = m.n_fake - tp;
  const fp = m.precision > 0 ? Math.round(tp / m.precision - tp) : 0;
  const tn = m.n_real - fp;
  return { tp, fp, tn, fn };
}

export function combinedTotalFrames(): number {
  return FFPP_EVAL.evaluation_metadata.total_frames + CELEBDF_EVAL.evaluation_metadata.total_frames;
}

export function combinedAvgSecondsPerFrame(): number {
  const totalSeconds =
    FFPP_EVAL.evaluation_metadata.elapsed_seconds + CELEBDF_EVAL.evaluation_metadata.elapsed_seconds;
  const totalFrames = combinedTotalFrames();
  return totalSeconds / totalFrames;
}
