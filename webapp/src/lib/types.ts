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
}

export interface FacialRegionScore {
  region: "forehead" | "eyes" | "nose" | "mouth" | "jaw";
  anomalyScore: number;
  label: string;
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
  confidence_breakdown: ConfidenceBreakdown[];
  /** Step-by-step theoretical basis for the Real/Fake verdict. */
  classification_reasoning: ClassificationReasoning;
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
