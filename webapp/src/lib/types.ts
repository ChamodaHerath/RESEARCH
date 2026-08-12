// Only one model is actually wired to a trained checkpoint and calibrated
// fusion weights (see configs/fusion_config.json). There is no second or
// third real model behind this demo. The backend returns the real
// Hugging Face model id (attention_extractor.DEFAULT_HF_DEEPFAKE_MODEL,
// e.g. "hamzenium/ViT-Deepfake-Classifier") rather than a fixed literal, so
// this is a plain string rather than a hardcoded union -- keep it in sync
// manually if you change SAEDF_MODEL_NAME/DEFAULT_HF_DEEPFAKE_MODEL.
export type ModelOption = string;

export type DetectionMode = "image" | "video";

export interface ReasoningStep {
  stage: string;
  formula?: string;
  value: string;
  interpretation: string;
}

export interface ClassificationReasoning {
  summary: string;
  steps: ReasoningStep[];
}

export interface ExplanationTheoryStep {
  label: string;
  detail: string;
}

export interface ExplanationTheory {
  summary: string;
  steps: ExplanationTheoryStep[];
  scores: {
    R: number;
    Q?: number;
  };
}

export interface SemanticExplanation {
  id: string;
  title: string;
  description: string;
  score: number;
  severity: "low" | "medium" | "high";
  /** Theoretical basis for why this zone/artifact was flagged — from live pipeline. */
  theory?: ExplanationTheory;
  /** P(fake) drop when this zone is occluded (causal gate). */
  causal_drop?: number | null;
}

export interface FacialRegionScore {
  /** Stable zone id (9 SAEDF zones, or legacy 5-region aggregate). */
  region: string;
  anomalyScore: number;
  label: string;
  /** True when this zone crossed the calibrated flagging threshold. */
  flagged?: boolean;
  /** How the zone was selected: calibrated_threshold when score > thr. */
  flag_reason?: string | null;
  /** Optional fused R(z) alongside Q(z)/display score. */
  R?: number;
  Q?: number;
  /** Calibrated flagging threshold for this zone. */
  threshold?: number;
  /** True when score > threshold under Fake+confidence gate (may still fail causal). */
  clears_threshold?: boolean;
}

export interface TimelineMarker {
  frame: number;
  timeSec: number;
  suspicion: number;
  note: string;
}

export interface ConfidenceBreakdown {
  metric: string;
  value: number;
}

export interface ExplanationValidationClaim {
  zone: string;
  artifact_type: string;
  faithfulness_passed: boolean;
  causal_drop: number | null;
  causal_passed: boolean | null;
}

export interface ExplanationValidationLive {
  n_claims: number;
  faithfulness_passed_count: number;
  faithfulness_pass_rate: number;
  causal_gate_enabled: boolean;
  causal_min_drop: number | null;
  causal_zones_tested: number;
  causal_claims_retained: number;
  causal_retention_rate: number | null;
  per_claim: ExplanationValidationClaim[];
  note: string;
}

export interface MaskGroundTruthValidation {
  dataset: string;
  n_frames_evaluated?: number | null;
  precision?: number | null;
  pointing_game_accuracy?: number | null;
  zone_score_auc_micro?: number | null;
  scope_note: string;
}

export interface ExplanationValidation {
  live: ExplanationValidationLive;
  mask_ground_truth_validation: MaskGroundTruthValidation | null;
  novel_contribution_note: string;
}

/** Live FF++ manipulation-mask overlay when the upload matches a local mask clip. */
export interface MaskGroundTruthVisual {
  available: boolean;
  /** Face crop with green GT manipulation mask overlay (base64 data URL). */
  overlay: string;
  /** Black background — only edited GT pixels visible (base64 data URL). */
  mask_pixels?: string;
  mask_coverage: number;
  frame_index: number;
  mask_source: string;
  note: string;
}

/** Rasterised semantic zone claims for pixel-level comparison with GT. */
export interface ClaimedZonesVisual {
  available: boolean;
  /** Face crop with red tint on explained zone pixels (base64 data URL). */
  overlay?: string;
  /** Black background — only explained zone pixels visible (base64 data URL). */
  zone_pixels?: string;
  zones: string[];
  zone_labels: string[];
  zone_coverage: number;
  note: string;
}

export interface DetectionResponse {
  prediction: "Real" | "Fake";
  confidence: number;
  riskLevel: "Low" | "Moderate" | "High" | "Critical";
  processing_time: string;
  model: ModelOption;
  mode: DetectionMode;
  terminal_logs: string[];
  semantic_explanations: SemanticExplanation[];
  facial_regions: FacialRegionScore[];
  timeline: TimelineMarker[];
  heatmap: string;
  attention_map: string;
  /** Real 224x224 aligned face crop actually fed to the ViT (before overlay). */
  faceCrop: string;
  /**
   * Present when the uploaded file matches a local FaceForensics++ mask video.
   * Green overlay = true edited pixels (dataset GT), not model attention.
   */
  mask_ground_truth?: MaskGroundTruthVisual | null;
  /** Red-highlight rasterisation of semantic zone claims (convex hulls). */
  claimed_zones_visual?: ClaimedZonesVisual | null;
  confidence_breakdown: ConfidenceBreakdown[];
  /** Step-by-step theoretical basis for the Real/Fake verdict. */
  classification_reasoning: ClassificationReasoning;
  /** Live faithfulness/causal checks + offline FF++ mask-GT research metrics. */
  explanation_validation?: ExplanationValidation;
  report: {
    caseId: string;
    date: string;
    time: string;
    modelVersion: string;
    inputFile: string;
    hashValue: string;
    fileType: string;
  };
}

export interface DetectionError {
  error: string;
  code?: string;
}
