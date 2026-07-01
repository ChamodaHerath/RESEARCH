# SAEDF — Semantic Attribution for Explainable Deepfake Forensics

A research pipeline that combines a Vision Transformer deepfake detector with
a four-tier Bayesian explanation system to produce **dual-head forensic reports**:
a classification head (fake probability + confidence tier) and a semantic
explanation head (zone-level attribution with verified natural-language sentences).

---

## Research contributions

| # | Contribution | Module |
|---|---|---|
| 1 | Dual-score fusion of CLS attention A(z) and grad-weighted attribution S(z) | `semantic_patch_mapper.py` |
| 2 | Bayesian β calibration — Beta(1,1) posterior over fusion weight, 95 % credible interval | `bayesian_calibration.py` |
| 3 | Dirichlet-Categorical artifact-type selection replacing deterministic rank-0 | `bayesian_calibration.py` |
| 4 | Per-zone 95 % credible intervals on R(z) propagated analytically from β uncertainty | `bayesian_calibration.py` |
| 5 | Manipulation-type-conditioned Bayesian zone priors from the FF++ literature | `zone_prior.py` |
| 6 | Method inference — zone activation pattern classifies manipulation type at runtime | `zone_prior.py` |
| 7 | Faithfulness verifier — every NL claim audited against its source ConceptVector | `semantic_patch_mapper.py` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Input: 224×224 face image  +  68-point dlib landmarks          │
└─────────────────┬───────────────────────────────────────────────┘
                  │  preprocess_landmarks.py
                  │  MTCNN face detection → dlib 68 landmarks
                  │  zone-patch matrix M ∈ ℝ^(196×9)
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2 — Dual-score extraction (single forward pass)          │
│                                                                 │
│  Model: prithivMLmods/Deep-Fake-Detector-v2-Model (HF ViT)     │
│                                                                 │
│  ┌─────────────────────┐   ┌─────────────────────────────────┐ │
│  │ Attention head A(z) │   │  Gradient head S(z)             │ │
│  │                     │   │                                 │ │
│  │ CLS→patch weights   │   │  Grad-weighted attention rollout│ │
│  │ last ViT layer      │   │  A_l = 0.5I + 0.5(W_l⊙ReLU(∂))│ │
│  │ normalised to [0,1] │   │  averaged, blurred, patch-avg   │ │
│  └──────────┬──────────┘   └───────────────┬─────────────────┘ │
│             │                              │                    │
│             │  zone aggregation via M      │                    │
│             │  A(z) = Σ M·α / area        │  S(z) = Σ M·φ/area│
│             └──────────────┬───────────────┘                    │
│                            │  also: fake_probability ← softmax │
└────────────────────────────┼───────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3 — Bayesian fusion calibration  (fusion_config.json)    │
│                                                                 │
│  Population normalisation: A_norm = sigmoid((A−μ_real)/σ_real) │
│                                                                 │
│  Option 1 — β posterior:  Beta(1,1) prior + BCE likelihood      │
│    MAP β = 0.243  [95% CI: 0.019 – 0.767]                      │
│    R(z) = β·A_norm(z) + (1−β)·S_norm(z)                       │
│                                                                 │
│  Option 3 — Dirichlet artifact selection:                       │
│    α_k^prior = K − k  (rank-0 gets highest concentration)      │
│    updated with R(z) quantile pseudo-counts from calibration   │
│                                                                 │
│  Option 4 — Credible intervals on R(z):                         │
│    σ_R(z) = σ_β · |A_norm(z) − S_norm(z)|                     │
│    95% CI = R̄(z) ± 1.96·σ_R(z)                               │
│                                                                 │
│  Option 2 — Zone priors  (literature-based Beta priors):        │
│    P(zone z anomalous | method m) ~ Beta(α_z^m, β_z^m)         │
│    Method inference: cosine similarity R(z) ↔ method signatures│
│    Q(z) = (α + R(z)·w) / (α + β + w)   [posterior mean]       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4 — ForensicReport  (dual-head output)                   │
│                                                                 │
│  ┌─────────────────────┐   ┌─────────────────────────────────┐ │
│  │ HEAD 1              │   │  HEAD 2                         │ │
│  │ ClassificationHead  │   │  ExplanationOutput              │ │
│  │                     │   │                                 │ │
│  │ label               │   │  flagged zones (sorted by Q(z)) │ │
│  │ fake_probability    │   │  artifact_type (Dirichlet MAP)  │ │
│  │ confidence_tier     │   │  confidence with 95% CI         │ │
│  │ model_name          │   │  zone_anomaly_posterior Q(z)    │ │
│  └─────────────────────┘   │  inferred_method + probs        │ │
│                            │  NL sentence (faithfulness ✓)   │ │
│                            └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Zone definitions

Nine semantic zones derived from dlib 68-point landmarks:

| Zone | Landmark indices | Primary deepfake artefact |
|---|---|---|
| left_eye | 36–41 | Iris texture, specular mismatch |
| right_eye | 42–47 | Iris texture, specular mismatch |
| mouth | 48–67 | Blending boundary, lip-sync gap |
| nose_bridge | 27–30 | Geometric distortion |
| nose_tip | 31–35 | Texture blending discontinuity |
| left_cheek | 1–5, 48–49 | Skin smoothing, pore erasure |
| right_cheek | 11–15, 54–55 | Skin smoothing, pore erasure |
| forehead | 17–26 | Hair boundary, skin inconsistency |
| chin_jaw | 0–2, 7–9, 14–16 | Jawline blending seam |

---

## Manipulation-type zone priors  (zone_prior.py)

Prior mean P(zone anomalous | method):

| Zone | Deepfakes | Face2Face | FaceSwap | NeuralTextures |
|---|---|---|---|---|
| mouth | 0.80 | **0.90** | 0.70 | 0.67 |
| left_eye | 0.70 | 0.67 | 0.67 | 0.38 |
| right_eye | 0.70 | 0.67 | 0.67 | 0.38 |
| chin_jaw | 0.63 | 0.38 | **0.80** | 0.25 |
| left_cheek | 0.50 | 0.25 | 0.70 | **0.80** |
| right_cheek | 0.50 | 0.25 | 0.70 | **0.80** |
| forehead | 0.38 | 0.17 | 0.50 | 0.70 |
| nose_bridge | 0.25 | 0.17 | 0.38 | 0.38 |
| nose_tip | 0.25 | 0.29 | 0.38 | 0.38 |

Source: Rössler et al. (2019) FaceForensics++; Zhao et al. (2021)
Multi-attentional Deepfake Detection; Li et al. (2020) Face X-Ray.

---

## Example output

```
══════════════════════════════════════════════════════════════
  FORENSIC REPORT
  Image : Deepfakes/000_003.mp4::f000000
══════════════════════════════════════════════════════════════

[HEAD 1 — CLASSIFICATION]
  Label            : Fake
  Fake probability : 0.8721  (high)
  Model            : prithivMLmods/Deep-Fake-Detector-v2-Model

[HEAD 2 — SEMANTIC EXPLANATION]
  Inferred method  : Deepfakes  (p=0.612)
    Deepfakes            0.612  ▪▪▪▪▪▪▪▪▪▪▪▪
    FaceSwap             0.241  ▪▪▪▪▪
    NeuralTextures       0.096  ▪▪
    Face2Face            0.051  ▪

  • High-confidence blending boundary artefact detected in the mouth region
    (confidence: 79.1% [72.4%–85.7%]).
  • Moderate iris texture inconsistency detected in the left eye region
    (confidence: 63.2% [56.1%–70.3%]).

  Zone scores  [R(z) = fusion,  Q(z) = Bayesian posterior]:
    mouth          R=0.712  Q=0.791  ██████████████
    left_eye       R=0.541  Q=0.632  ██████████▊
    chin_jaw       R=0.483  Q=0.572  █████████▋
    right_eye      R=0.401  Q=0.492  ████████
    ...
══════════════════════════════════════════════════════════════
```

---

## Installation

```bash
# Python 3.10+
pip install torch torchvision timm transformers
pip install opencv-python-headless numpy matplotlib scipy
pip install dlib mtcnn scikit-learn   # preprocessing + evaluation
```

Download the dlib 68-point shape predictor:
```bash
# Place in repo root:
# shape_predictor_68_face_landmarks.dat
```

---

## Usage

### 1. Preprocess a FaceForensics++ dataset

```bash
python preprocess_landmarks.py \
    --ffpp-root /data/FaceForensics++ \
    --output-json outputs/ffpp_landmarks.json \
    --num-frames 10
```

### 2. Calibrate the fusion pipeline

```bash
python recalibrate_v2.py   # reads outputs/ffpp_landmarks_cal.json
# writes configs/fusion_config.json
```

### 3. Run the dual-head pipeline on one frame

```bash
python run_pipeline_single_frame.py \
    --json outputs/ffpp_landmarks_test.json \
    --key "Deepfakes/000_003.mp4::f000000" \
    --output outputs/report.json
```

### 4. Evaluate over a full test set

```bash
python evaluate_pipeline.py \
    --json outputs/ffpp_landmarks_test.json \
    --output outputs/evaluation_results.json
```

### 5. Use as a library

```python
from pipeline import SAEDFPipeline

with SAEDFPipeline() as pipe:
    report = pipe.predict_from_landmark_entry(entry)
    report.print_summary()
    print(report.to_json())
```

---

## File index

| File | Role |
|---|---|
| `preprocess_landmarks.py` | MTCNN + dlib preprocessing → landmark JSON |
| `attention_extractor.py` | ViT model loading + CLS attention extraction |
| `baseline_heatmap_generator.py` | Grad-weighted rollout, phi extraction, visual heatmaps |
| `semantic_patch_mapper.py` | Zone fusion, ConceptVector, NL templates, ForensicReport |
| `bayesian_calibration.py` | BayesianBetaCalibrator, DirichletArtifactPrior, CI propagation |
| `zone_prior.py` | ZonePriorModel: literature priors, method inference, Q(z) posteriors |
| `pipeline.py` | SAEDFPipeline orchestrator (single forward pass, dual-head output) |
| `recalibrate_v2.py` | Calibration script → configs/fusion_config.json |
| `run_pipeline_single_frame.py` | CLI demo: single frame → ForensicReport |
| `evaluate_pipeline.py` | Batch evaluation: classification AUC + explanation metrics |
| `configs/fusion_config.json` | Calibrated β, population stats, Dirichlet posteriors |

---

## References

Rössler, A. et al. (2019). FaceForensics++: Learning to Detect Manipulated
Facial Images. *ICCV 2019*. arXiv:1901.08971.

Caron, M. et al. (2021). Emerging Properties in Self-Supervised Vision
Transformers (DINO). *ICCV 2021*.

Chefer, H. et al. (2021). Transformer Interpretability Beyond Attention
Visualization. *CVPR 2021*.

Zhao, H. et al. (2021). Multi-attentional Deepfake Detection. *CVPR 2021*.

Li, L. et al. (2020). Face X-Ray for More General Face Forgery Detection.
*CVPR 2020*.
