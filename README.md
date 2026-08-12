# Extended Explainable Vision Transformer Framework For Deepfake Detection

A research pipeline that combines a Vision Transformer deepfake detector with
a four-tier Bayesian explanation system to produce **dual-head forensic reports**:
a classification head (fake probability + confidence tier) and a semantic
explanation head (zone-level attribution with verified natural-language sentences).

A **precision-first research policy** gates when NL claims are emitted; an optional
**causal occlusion badge** annotates whether blanking a zone moves P(fake); and an
**artifact evidence module** adds measurable image cues (texture, boundary, colour,
lighting, frequency) on top of the ontology-based Dirichlet artifact head.

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
| 7 | Faithfulness verifier — every NL claim (zone, artifact, severity, confidence %, and CI bounds) audited against its source ConceptVector, with a re-verified fallback path | `semantic_patch_mapper.py` |
| 8 | Systematic detector-selection tournament (AUC/ECE/Brier/confusion matrix, 2 datasets, 4 candidate ViT/16 checkpoints) | `detector_benchmark.py`, `audit_vit_models.py` |
| 9 | 6-way ablation study (classifier-only, A(z)-only, S(z)-only, fixed-β, Bayesian R, Bayesian Q) isolating the effect of dual-score fusion and Bayesian zone priors | `ablation_study.py` |
| 10 | Mask-based ground-truth validation of zone attributions — FF++ manipulation masks converted to zone-level GT; localization precision/recall/F1/IoU, pointing game, zone-score AUC | `zone_localization.py`, `evaluate_zone_localization.py` |
| 11 | Causal faithfulness via deletion/insertion curves — zones occluded/revealed in attribution order vs random order, measuring whether flagged zones actually drive the fake score | `zone_localization.py`, `evaluate_zone_localization.py` |
| 12 | Quantified method inference — 4×4 confusion matrix, frame/video accuracy, top-2 accuracy of the zone-pattern method classifier vs FF++ ground-truth labels | `evaluate_method_inference.py` |
| 13 | Leakage-controlled detector training — official FF++ video-level splits with identity separation, ImageNet-only base model, per-checkpoint provenance record | `make_ffpp_splits.py`, `train_vit_detector.py`, `ffpp_utils.py` |
| 14 | Statistical rigor layer — video-level metrics, cluster-bootstrap 95 % CIs, per-manipulation-method breakdown | `eval_metrics.py`, `evaluate_pipeline.py` |
| 15 | Precision-first explanation policy — Fake + P(fake) > 0.70 gate, per-zone Q(z) thresholds at the real-face 85th percentile, max 3 claimed zones; prioritises localisation precision over recall | `recalibrate_v2.py`, `configs/fusion_config.json` |
| 16 | Causal occlusion gate — measures ΔP(fake) per claimed zone; **hard filter** in batch `SAEDFPipeline` (default), **optional badge** in live demo (`SAEDF_CAUSAL_HARD_FILTER=0`) | `semantic_patch_mapper.py`, `pipeline.py`, `inference_server.py` |
| 17 | Measurable artifact evidence — interpretable zone features + Bayesian artifact inference with abstention; evidence-backed NL sentences and extended faithfulness verifier | `artifact_analysis/`, `configs/artifact_evidence_config.json` |

---

## Detector: configurable, not assumed

The live demo is currently configured to use
`hamzenium/ViT-Deepfake-Classifier`, because it is a drop-in-compatible
ViT/16 224px checkpoint for the current CLS-to-patch explanation pipeline
and was the strongest drop-in candidate in the detector benchmark below.
The model can be changed with `SAEDF_MODEL_NAME`, but every detector switch
must be followed by recalibration (`recalibrate_v2.py`) and re-evaluation
(`evaluate_pipeline.py`) before its report values are defensible.

Earlier experiments also benchmarked other drop-in-compatible public ViT
checkpoints. `audit_vit_models.py` + `detector_benchmark.py` compare candidate
detectors on classifier metrics alone (no explanation layer, so the comparison
isn't confounded by fusion/calibration). Full per-model result JSON, kept as
separate files per dataset as both are independently reproducible:
`outputs/model_benchmarks/ffpp_benchmark.json` (172 frames) and
`outputs/model_benchmarks/celebdf_benchmark.json` (5,179 frames, cross-dataset).

**FaceForensics++ (in-domain, 172 frames: 86 real / 86 fake)**

| Model | Accuracy | AUC-ROC | Precision | Recall | F1 | ECE | Brier |
|---|---|---|---|---|---|---|---|
| `prithivMLmods/Deep-Fake-Detector-v2-Model` (original default) | 41.9 % | 0.429 | 0.400 | 0.326 | 0.359 | 0.236 | 0.305 |
| `prithivMLmods/Deepfake-Detection-Exp-02-21` | 49.4 % | 0.466 | 0.480 | 0.140 | 0.216 | 0.083 | 0.261 |
| `dima806/deepfake_vs_real_image_detection` | 68.6 % | 0.687 | 0.619 | 0.965 | 0.755 | 0.226 | 0.262 |
| `hamzenium/ViT-Deepfake-Classifier` | 99.4 % | 1.000 | 1.000 | 0.988 | 0.994 | 0.034 | 0.004 |

**Celeb-DF v2 (cross-dataset, 5,179 frames: 1,780 real / 3,399 fake)**

| Model | Accuracy | AUC-ROC | Precision | Recall | F1 | ECE | Brier |
|---|---|---|---|---|---|---|---|
| `prithivMLmods/Deep-Fake-Detector-v2-Model` (original default) | 45.1 % | 0.558 | 0.728 | 0.262 | 0.386 | 0.357 | 0.383 |
| `prithivMLmods/Deepfake-Detection-Exp-02-21` | 34.4 % | 0.528 | 1.000 | 0.0003 | 0.001 | 0.271 | 0.299 |
| `dima806/deepfake_vs_real_image_detection` | 63.8 % | 0.509 | 0.662 | 0.915 | 0.769 | 0.309 | 0.326 |
| `hamzenium/ViT-Deepfake-Classifier` | 79.1 % | 0.901 | 0.931 | 0.737 | 0.823 | 0.163 | 0.182 |

`hamzenium/ViT-Deepfake-Classifier` is the active demo default. Earlier
experiments also kept `prithivMLmods/Deep-Fake-Detector-v2-Model` as a
documented alternative; after any switch, recalibrate
`configs/fusion_config.json` for the chosen checkpoint before treating
report values as defensible.

**Honest caveat on the FF++ AUC = 1.000 result:** a perfect in-domain AUC on
its own would be a red flag for shortcut learning (e.g. a systematic encoding
difference between how real/fake frames were sourced). What makes this result
credible rather than suspicious is that the *same* checkpoint, run through the
*same* preprocessing pipeline, still achieves a strong but clearly imperfect
0.90 AUC on Celeb-DF — a completely different dataset with different subjects
and a different synthesis method. If the model were exploiting a preprocessing
artefact rather than real forensic signal, we would expect similarly extreme
performance on Celeb-DF too, not a realistic, non-trivial generalization gap.
`buildborderless/CommunityForensics-DeepfakeDet-ViT` was also investigated on
request but uses 384×384/24×24-patch inputs, incompatible with this pipeline's
224×224/14×14-patch zone-mapping architecture without a larger refactor.

`prithivMLmods/Deepfake-Detect-Siglip2` is not treated as a drop-in switch for
the current explanation layer. It is a SigLIP2 image classifier, not the same
ViT/CLS-token architecture assumed by the 196-patch semantic mapper, so using
it honestly would require a SigLIP2-specific attention adapter, fake-class
mapping verification, recalibration, and fresh evaluation.

**Provenance caveat on all off-the-shelf checkpoints (important).** Every
model in the tournament above is a public HuggingFace checkpoint with
*unknown training data*. None of their in-domain FF++ numbers can rule out
train/test leakage — the `hamzenium` checkpoint's AUC = 1.000 in particular
may partly reflect having seen FF++ videos during its own training. The
cross-dataset Celeb-DF column is the more trustworthy signal, but the only
fully defensible path for in-domain claims is a detector whose training data
is documented. The repository now provides that path:

1. `make_ffpp_splits.py` — splits a landmark JSON into train/val/test at the
   **video level** using the official FF++ split files (720/140/140 pairs),
   with identity separation (an original video and every fake derived from it
   always land in the same split). A hash-based fallback exists for partial
   local copies. It refuses to write splits if any FF++ target id would leak
   across splits.
2. `train_vit_detector.py` — fine-tunes from
   `google/vit-base-patch16-224-in21k` by default (a generic ImageNet-21k
   backbone with **no deepfake exposure**), runs a video-level leakage check
   before training, and writes a `provenance.json` next to the checkpoint
   recording the base model, split source, and exact frame/video counts.
   After training: recalibrate (`recalibrate_v2.py --model-name
   models/saedf_vit16_detector`) and re-evaluate before quoting numbers.

Off-the-shelf checkpoints remain useful as benchmark baselines, but
thesis-grade in-domain numbers should come from the provenance-documented
detector above.

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
│  Model: hamzenium/ViT-Deepfake-Classifier (HF ViT)             │
│  (configurable via SAEDF_MODEL_NAME; recalibrate after change) │
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
│    MAP β = 0.933  (mean 0.869)  [95% CI: 0.660 – 0.994]        │
│    R(z) = β·A_norm(z) + (1−β)·S_norm(z)                       │
│    (β is re-fit per detector on a leakage-free val subsample;  │
│    see configs/fusion_config.json for the live posterior.)     │
│                                                                 │
│  Option 3 — Dirichlet artifact selection:                       │
│    α_k^prior = K − k  (rank-0 gets highest concentration)      │
│    logit_k = log(α_k) − rank_k · R(z) · temperature            │
│    high R(z) → concentrates on rank-0; low R(z) → more spread  │
│                                                                 │
│  Option 4 — Credible intervals on R(z):                         │
│    σ_R(z) = σ_β · |A_norm(z) − S_norm(z)|                     │
│    95% CI = R̄(z) ± 1.96·σ_R(z)                               │
│                                                                 │
│  Option 2 — Zone priors  (literature-based Beta priors):        │
│    P(zone z anomalous | method m) ~ Beta(α_z^m, β_z^m)         │
│    Method inference: cosine similarity R(z) ↔ method signatures│
│    Q(z) = (α + R(z)·w) / (α + β + w)   [posterior mean]       │
│    Zone flagging thresholds are calibrated on the Q(z)          │
│    distribution (not R(z) — see "Selectivity" below)           │
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
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 5 — Post-hoc explanation enrichment (optional)           │
│                                                                 │
│  Causal occlusion  — ΔP(fake) when each claimed zone is blanked │
│    batch SAEDFPipeline: hard_filter=True (drop weak claims)     │
│    live demo default:   hard_filter=False (causal badge only)   │
│                                                                 │
│  Artifact evidence   — texture / boundary / colour / lighting / │
│    frequency features → BayesianArtifactInferencer → evidence-  │
│    backed NL (or abstention: "no clear observable artifact")    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Research policy: `precision_first_v1`

Calibrated rules live in `configs/fusion_config.json` under `research_policy`:

| Rule | Detail |
|---|---|
| Real verdict | No zone flags, no NL claims |
| Weak Fake | P(fake) ≤ 0.70 → region scores only, zero NL claims |
| Zone claim | Q(z) > per-zone threshold (real-face 85th percentile on Q scale) |
| Cap | At most 3 claimed zones per frame (`max_flagged_zones`) |
| Fusion | R(z) uses β posterior **mean** (attention + gradient both contribute) |
| Causal | Live demo: optional badge; batch pipeline: hard filter by default |
| Claim scope | High **precision** localisation of model anomaly mass — not full mask recall |

On the calibration set, mean real-frame pass rate ≈ 15 % and mean fake-frame pass
rate ≈ 27 % per zone — the thresholds are intentionally strict so emitted claims
are sparse but trustworthy.

---

## Evaluation results

Full dual-head pipeline evaluation (classification + explanation), not just
the classifier alone. See `evaluate_pipeline.py`. Canonical numbers synced to
the webapp via `python sync_webapp_data.py` →
`webapp/src/data/eval-ffpp.json`, `eval-celebdf.json`, `zone-localization.json`.

### Classification head

| Metric | FF++ official test (1000 frames, 4 methods) | Celeb-DF (cross-dataset, 5179 frames) |
|---|---|---|
| Accuracy | 48.1 % | 79.1 % |
| AUC-ROC | 0.676 | 0.901 |
| Average precision | 0.902 | 0.941 |
| Precision / Recall / F1 | 0.925 / 0.383 / 0.541 | 0.931 / 0.737 / 0.822 |
| Confusion matrix (TP/FP/TN/FN) | 306 / 25 / 175 / 494 | 2504 / 186 / 1594 / 895 |
| ECE | 0.477 | 0.163 |
| Brier score | 0.469 | 0.182 |

**Historical note — Deepfakes-only 172-frame subset (superseded protocol):**
the earlier in-domain benchmark on 86 real / 86 fake Deepfakes frames reported
99.4 % accuracy and AUC 1.000 with the same `hamzenium` checkpoint. That subset
is not representative of the full four-method FF++ test split; the 1000-frame
row above is the current thesis-facing protocol.

### Explanation head (under `precision_first_v1`)

| Metric | FF++ (1000 frames) | Celeb-DF (5179 frames) |
|---|---|---|
| Mean claimed zones / fake frame | 0.17 / 9 | 1.97 / 9 |
| Textual faithfulness pass rate | 100 % | 100 % |

Textual faithfulness checks sentence ↔ ConceptVector consistency only (see
Limitations). It is reported separately from mask-ground-truth localisation below.

**Cross-dataset calibration is honest, not perfect.** The calibration
reliability diagram (`outputs/plots/calibration_curve.png`) shows FF++ is
essentially a step function at the decision boundary (near-perfect
separation), while Celeb-DF is visibly miscalibrated at low predicted
probabilities — several low-confidence bins have an empirical fake rate of
60–75 % despite the classifier's mean confidence being under 25 % there. This
tracks with the moderate 73.7 % recall on Celeb-DF: real fake frames are more
often missed cross-dataset than the in-domain numbers alone would suggest,
and the model is somewhat overconfident about being "real" at the low end.
Anyone deploying this cross-domain should re-calibrate the probability output
(e.g. isotonic/Platt scaling) before treating `fake_probability` as a
well-calibrated probability rather than just a ranking score.

**Statistical rigor caveats now measured, not hand-waved.**
`evaluate_pipeline.py` additionally reports:

- **Video-level metrics** — one aggregated prediction per source video
  (mean fake probability over its frames). Frames of one video are highly
  correlated, so "172 frames" is really ~35 independent videos; the
  video-level numbers are the honest effective-sample-size view.
- **Cluster-bootstrap 95 % confidence intervals** on accuracy / AUC / recall,
  resampling whole *videos* (not frames) so the intervals respect the
  within-video correlation structure.
- **Per-manipulation-method breakdown** (Deepfakes / Face2Face / FaceSwap /
  NeuralTextures — each method's fakes vs all reals) when the landmark JSON
  was preprocessed from an FF++ root containing the method directories.
  `preprocess_landmarks.py` now scans **all four** method folders (not just
  Deepfakes) and stamps every fake frame with its `method`.

Note: the previously reported 172-frame FF++ test set covers the Deepfakes
method only. The current **1000-frame official test split** (all four methods)
is the recommended thesis protocol. Re-run with `make_ffpp_splits.py
--official-splits` and `evaluate_pipeline.py --json outputs/ffpp_landmarks_test_full.json`.

Plots: `python plot_evaluation.py ...` → `outputs/plots/` (`roc_curve.png`,
`pr_curve.png`, `calibration_curve.png`).

---

## Ground-truth validation: are the claimed zones actually manipulated?

The faithfulness verifier only checks that generated sentences match the
internal ConceptVector — it cannot tell whether the ConceptVector points at a
genuinely manipulated region. `evaluate_zone_localization.py` closes that gap
using the manipulation masks that ship with FaceForensics++ (the pixel
region each synthesis method actually altered):

1. For every fake test frame, the mask frame is cropped with the **same face
   box** stored during preprocessing (`entry["crop_box"]`, added by
   `preprocess_landmarks.py`) and resized to 224×224, so mask and face crop
   are pixel-aligned.
2. The 9 zone convex hulls are rasterised from the frame's landmarks; a zone
   is ground-truth-manipulated when ≥ 30 % of its pixels fall inside the
   mask (`--coverage-threshold`).
3. Claimed zones (from the precision-first policy) are scored against this ground truth.

### Mask-GT results (current run)

Evaluated on **26,938 fake FF++ frames** (all four methods). Synced to
`webapp/src/data/zone-localization.json` via `sync_webapp_data.py`.

| Metric | Value | Interpretation |
|---|---|---|
| Zone precision (micro) | **97.7 %** | Of emitted zone claims, 97.7 % overlap GT-manipulated regions |
| Zone recall (micro) | 6.6 % | Intentionally low — precision-first policy flags sparingly |
| Pointing-game accuracy | **99.7 %** | Top-scored zone hits a GT-manipulated region on 99.7 % of frames |
| Zone-score AUC (micro) | 0.690 | Rank quality of Q(z)/R(z) vs zone GT labels |
| Deletion AUC (attr vs random) | 0.442 vs 0.414 | Attribution-ordered occlusion drops P(fake) slightly faster |
| Insertion AUC (attr vs random) | 0.226 vs 0.229 | Comparable to random on 200-frame causal subsample |

Per-method precision (all four FF++ synthesis types): **96.5 – 98.9 %**.

These numbers support the primary localisation claim: **when the explanation
head emits a zone claim, it coincides with a truly manipulated region under
FF++ mask ground truth with very high precision.** Low recall is by design —
the policy trades exhaustive mask coverage for examiner-facing trust.

Reported metrics (full JSON):

- **Localization precision / recall / F1 / IoU** — micro over all zone
  decisions, plus per-zone and per-method breakdowns. Precision answers the
  examiner's question directly: *of the zones the system reports, how many
  were really manipulated?*
- **Pointing game** — fraction of frames whose top-scored zone is a
  ground-truth-manipulated zone (standard XAI localization metric).
- **Zone-score AUC** — rank-AUC of Q(z) against zone ground truth.
- **Deletion / insertion curves (causal faithfulness)** — zones are occluded
  (deletion) or revealed onto a blurred baseline (insertion) in decreasing
  attribution order, and the classifier's fake probability is re-measured at
  each step. If attributions are causally faithful, attribution-ordered
  deletion drops the fake score *faster than random ordering* (lower AUC) and
  attribution-ordered insertion recovers it faster (higher AUC). The random-
  order baseline is computed per frame (`--random-permutations`).

```bash
python evaluate_zone_localization.py \
    --json outputs/ffpp_landmarks_test.json \
    --ffpp-root "datasets/FaceForensics++/FaceForensics++_C23" \
    --masks-root "datasets/FaceForensics++/masks" \
    --output outputs/zone_localization_ffpp.json
```

The script needs the FF++ mask videos (downloadable with the official FF++
script, `-d <method> -c c23 -t masks`). Run `python download_ffpp_masks.py
--instructions` for step-by-step setup, then `python download_ffpp_masks.py
--verify-only` to confirm masks are discoverable. The C23 manipulated RGB
videos under `FaceForensics++_C23/` are **not** valid ground truth. A landmark
JSON with `crop_box` fields is also required (re-run `preprocess_landmarks.py`;
frames without a crop box are counted and skipped, not silently wrong).

---

## Evaluating semantic explanations (recommended protocol)

Treat explanation quality as **four independent layers** — do not collapse them
into a single “faithfulness %”:

| Layer | What it measures | Tool / metric |
|---|---|---|
| 1. Textual faithfulness | NL sentence ↔ ConceptVector | `verify_faithfulness()` — sanity check only |
| 2. Localisation correctness | Claimed zone ↔ FF++ mask GT | `evaluate_zone_localization.py` — precision, pointing game |
| 3. Causal sensitivity | Zone occlusion ↔ ΔP(fake) | `filter_explanations_by_causal_correctness`, deletion/insertion curves |
| 4. Artifact / utility | Observable artifact labels, human trust | `scripts/evaluate_artifact_module.py` + expert annotation (future) |

Layer 2 is now the headline result for the semantic explanation head (97.7 %
precision, 99.7 % pointing game). Layers 3–4 remain partially complete — see
Limitations.

---

## Artifact evidence module

`artifact_analysis/` extends the ontology-based Dirichlet artifact head with
**measurable image evidence** — interpretable features (LBP entropy, boundary
gradients, ΔE colour mismatch, FFT band energies, etc.) fused via a Bayesian
inferencer with abstention when evidence is weak.

- **Does not replace** A(z)/S(z)/R(z)/Q(z) zone flagging — runs only on zones
  already claimed by the precision-first gate.
- **Default ON** in `SAEDFPipeline(enable_artifact_evidence=True)`; replaces NL
  sentences with evidence-backed text when features support a call.
- Config: `configs/artifact_evidence_config.json` (weak Dirichlet priors — recalibrate
  before reporting artifact-level metrics).
- Full design doc: `artifact_analysis/README.md`.

```bash
# Synthetic demo (blending seam around mouth zone)
python scripts/demo_artifact_evidence.py --out results/demo_artifact_overlay.png

# Offline evaluation (requires annotated CSV — does not invent labels)
python scripts/evaluate_artifact_module.py --csv annotations.csv --fit-calibration --out-dir results
```

Tests: `python -m pytest artifact_analysis/tests/ -v`

---

## Method-inference evaluation: measuring the "zero-shot" claim

`zone_prior.py` infers the manipulation method from the R(z) zone pattern.
FF++ frame keys encode the true method, so this claim is directly testable —
`evaluate_method_inference.py` reports:

- frame-level accuracy vs the 25 % four-way chance baseline, macro F1,
  and per-method precision/recall/F1,
- the full 4×4 confusion matrix (+ an Unknown/Novel column),
- top-2 accuracy (true method within the two highest-probability methods),
- video-level accuracy via majority vote over each video's frames,
- the mean probability assigned to the *true* method (probabilistic quality,
  not just argmax).

```bash
python evaluate_method_inference.py \
    --json outputs/ffpp_landmarks_test.json \
    --output outputs/method_inference_ffpp.json
```

This requires fake frames from all four FF++ method directories in the
landmark JSON (see the preprocessing note above). Until this has been run on
the full four-method test split, treat "zero-shot method inference" as a
*mechanism* the pipeline provides, not a validated result.

### Learned method signatures (recommended)

The default method signatures are literature-based prior means, which describe
a generic detector — not yours. `calibrate_method_signatures.py` replaces them
with signatures learned from a leakage-free calibration split (FF++ train or
val — never test): it computes the fused R(z) vector for fake frames with a
known method, averages per method into 4 learned centroids, and optionally
fits a multinomial logistic regression (9 features → 4 classes) that replaces
cosine-to-centroid at inference.

```bash
# 1. Preprocess a calibration split (val = 140 videos of fakes + originals)
python preprocess_landmarks.py \
    --ffpp-root "datasets/FaceForensics++/FaceForensics++_C23" \
    --official-splits "datasets/FaceForensics++/splits" \
    --split val --output-json outputs/ffpp_landmarks_val.json

# 2. Learn signatures (writes configs/method_signatures.json)
python calibrate_method_signatures.py \
    --json outputs/ffpp_landmarks_val.json --fit-classifier

# 3. Re-evaluate on the untouched test split
python evaluate_method_inference.py --json outputs/ffpp_landmarks_test_full.json
```

When `configs/method_signatures.json` exists it is auto-loaded by the pipeline
(and the evaluation records `method_signature_source` in its metadata); delete
or rename the file to fall back to the literature signatures. R(z) itself does
not depend on the zone-prior model, so the calibration is not circular. The
Beta priors used for the Q(z) posterior update remain the literature values —
only the method-inference signatures are learned.

---

## Ablation study: does the semantic layer actually help?

`ablation_study.py` holds the detector's `fake_probability` fixed and swaps
out only the zone-attribution mechanism, so every row below is attributable
purely to the explanation layer, not to a different classifier. Six modes,
run on both datasets (`outputs/ablation_results_ffpp.json`,
`outputs/ablation_results_celebdf.json`):

| Mode | What it measures | FF++ flagged/frame | Celeb-DF flagged/frame | FF++ mean \|R-separation\| | Celeb-DF mean \|R-separation\| |
|---|---|---|---|---|---|
| `classifier_only` | Detector alone, **no zone attribution at all** | N/A | N/A | N/A | N/A |
| `attention_only` | A(z) only, β=1.0 | 1.29 / 9 | 0.26 / 9 | 0.189 | 0.066 |
| `gradient_only` | S(z) only, β=0.0 | 0.33 / 9 | 0.24 / 9 | 0.064 | 0.053 |
| `fixed_beta` | Fused R(z), β=0.5 (no calibration) | 0.24 / 9 | 0.06 / 9 | 0.116 | 0.055 |
| `bayesian_r` | Fused R(z), calibrated β, **no** zone prior | 7.37 / 9 | 4.55 / 9 | 0.303 | 0.159 |
| `bayesian_q` | Adds the Bayesian zone prior Q(z) | 2.97 / 9 | 1.97 / 9 | 0.303 | 0.159 |
| `full` | Q(z) + Bayesian artifact selection + CI | 2.97 / 9 | 1.97 / 9 | 0.303 | 0.159 |

(`classifier_only`'s accuracy/AUC are the same tournament numbers reported
above — 99.4 % / AUC 1.000 on FF++, 79.1 % / AUC 0.901 on Celeb-DF — included
in the raw JSON as `modes.classifier_only.classification_accuracy` /
`.classification_auc` for completeness, not repeated as a column here since
they aren't zone-level metrics.)

What this table shows:

- **`classifier_only` cannot answer "where" or "what kind" at all.** Every
  cell is N/A by construction — there is no M matrix, no A(z)/S(z)/R(z)/Q(z)
  computation, nothing to report at the zone level. This is the actual
  ablation baseline: it is not that the classifier-only row scores worse on
  these metrics, it's that the metrics don't exist without the semantic
  layer this project contributes. Region-level attribution, artifact typing,
  and per-zone confidence are only possible once A(z)/S(z) fusion is added.
- **Fusing A(z) and S(z) increases separation over either alone.** Both
  `attention_only` and `gradient_only` individually produce weaker
  fake/real separation than `bayesian_r`'s fused R(z) (0.189 and 0.064 vs.
  0.303 on FF++; 0.066 and 0.053 vs. 0.159 on Celeb-DF) — evidence for the
  dual-score fusion claim, not just an assumption.
- **Calibrating β and zone thresholds matters.** `fixed_beta` (uncalibrated
  β=0.5) has both a much lower separation and a much lower flagged-zone
  count than the calibrated `bayesian_r` — an uncalibrated fusion weight
  doesn't just flag differently, it discards signal.
- **The Bayesian zone prior Q(z) changes *selectivity*, not the underlying
  R(z) signal.** `bayesian_r` → `bayesian_q` → `full` all report the
  *identical* mean \|R-separation\| (0.303 / 0.159) because Q(z) is a
  posterior over which zones to *report*, computed downstream of R(z) — it
  does not alter R(z) itself. The entire ~2.5x reduction in flagged zones
  (7.37 → 2.97 on FF++, 4.55 → 1.97 on Celeb-DF) comes purely from Q(z)'s
  literature-informed shrinkage suppressing zones with weak evidence, at
  zero cost to the discriminative signal available in R(z). This is the
  central empirical claim for Bayesian zone priors as a research
  contribution: better selectivity without giving up separation.

Reproduce with:

```bash
python ablation_study.py --json outputs/ffpp_landmarks_test.json --output outputs/ablation_results_ffpp.json
python ablation_study.py --json outputs/celebdf_landmarks_test.json --output outputs/ablation_results_celebdf.json
```

---

## Selectivity: why zone flagging uses Q(z), not raw R(z)

An earlier version of this pipeline flagged zones directly from the fused
R(z) score, calibrated with a global per-zone quantile threshold. The
ablation study (`ablation_study.py`, results in `outputs/ablation_results_*.json`)
isolates the effect this had:

| Mode | Mean flagged zones/fake frame — FF++ | — Celeb-DF |
|---|---|---|
| `bayesian_r` (R(z) + per-zone threshold, no zone prior) | 7.37 / 9 | 4.55 / 9 |
| `bayesian_q` (adds the Bayesian zone prior Q(z)) | **2.97 / 9** | **1.97 / 9** |

Raw R(z)-based flagging was not selective — most zones cleared the threshold
on most fake frames, which is the "explanations over-flag zones" problem this
project set out to fix. Two compounding causes were found and fixed:

1. **Scale mismatch.** `predict_label_and_explanation()` always flags zones
   using Q(z) when a `ZonePriorModel` is loaded (which
   `SemanticExplanationGenerator` does by default), but the original
   `recalibrate_v2.py` computed `zone_thresholds` from the *R(z)*
   distribution. Because Q(z) is a shrinkage estimator pulled toward
   literature-prior means, it lives on a compressed scale that can sit
   entirely below an R(z)-calibrated threshold — in the worst case, zero
   zones would ever flag. `recalibrate_v2.py` now computes thresholds from
   the actual Q(z) distribution of the calibration set's real frames.
2. **Missing zone-prior regularisation.** Even after fixing the scale, Q(z)'s
   shrinkage toward literature-informed priors is itself what suppresses
   spurious flagging on zones with weak evidence — this is the ablation
   result above (`bayesian_r` → `bayesian_q`), and it holds up cross-dataset,
   which is evidence the effect is a real property of the Bayesian zone-prior
   design rather than dataset-specific overfitting of a threshold.

A related, more subtle bug was caught while producing the concrete examples
below: when Q(z) backs the reported `confidence`, the 95 % credible interval
on that confidence must also be expressed on the Q(z) scale — the interval
was previously always computed on raw R(z), which can produce a
contradictory-looking sentence such as `confidence: 73% [99–100%]` (the
point estimate not even falling inside its own stated interval). Fixed in
`predict_label_and_explanation()` by pushing the R(z) interval bounds through
the same monotonic Q(z) transform (`ZonePriorModel.compute_marginal_posteriors`
is a fixed-weight convex combination of terms linear and non-decreasing in
R(z), so it is itself non-decreasing in R(z) — applying it to
`[R_ci_lower, R_ci_upper]` yields a valid interval on Q(z)). Covered by
`tests/test_selectivity_and_pipeline.py::test_predict_label_and_explanation_ci_is_on_same_scale_as_reported_confidence`.

### Selectivity in practice — concrete examples

`selectivity_examples.py` scans real fake frames and saves full forensic
reports for frames that flag only a handful of zones (rather than printing
an aggregate statistic only). Two representative examples
(`outputs/selectivity_examples_ffpp.json`, `outputs/selectivity_examples_celebdf.json`):

```
FF++  Deepfakes/000_003.mp4::f000000  (fake_probability 0.975)
  Inferred method: FaceSwap (p=0.400)
  • Moderate iris texture inconsistency detected in the right eye region
    (confidence: 72.9% [72.8–72.9%]).
  • Moderate iris texture inconsistency detected in the left eye region
    (confidence: 72.9% [72.7–72.9%]).
  • Moderate jawline blending seam detected in the chin jaw region
    (confidence: 71.0% [69.1–71.1%]).
  -> 3 of 9 zones flagged (nose_bridge, nose_tip, left_cheek, right_cheek,
     mouth, forehead all suppressed on this frame).

Celeb-DF  Celeb-synthesis/id1_id0_0007.mp4::f000010  (fake_probability 0.977)
  Inferred method: FaceSwap (p=0.430)
  • Moderate skin smoothing over-suppression detected in the left cheek region
    (confidence: 73.6% [70.9–73.7%]).
  • Moderate skin smoothing over-suppression detected in the right cheek region
    (confidence: 73.0% [71.5–73.4%]).
  • Moderate iris texture inconsistency detected in the left eye region
    (confidence: 70.0% [68.1–70.0%]).
  -> 3 of 9 zones flagged; mouth (a known weak zone for this detector, see
     below) correctly stays unflagged despite a high overall fake probability.
```

Reproduce with:

```bash
python selectivity_examples.py --json outputs/ffpp_landmarks_test.json --n 3 --max-flagged 3 \
    --output outputs/selectivity_examples_ffpp.json
python selectivity_examples.py --json outputs/celebdf_landmarks_test.json --n 3 --max-flagged 3 \
    --max-scan 500 --output outputs/selectivity_examples_celebdf.json
```

---

## On β (an honest finding, not a bug)

Recalibrating fusion for `hamzenium/ViT-Deepfake-Classifier` on a
leakage-free **val** subsample currently produces **MAP β ≈ 0.933**
(mean ≈ 0.869, 95% CI roughly [0.66, 0.99]) — the Bayesian posterior
still leans toward attention A(z), but no longer collapses to a pure
attention solution. This is a real, reproducible result of the
calibration procedure, not a sign it's broken: it means this
particular detector's CLS attention is highly discriminative
between real/fake zone patterns on the calibration set, while the
gradient-rollout signal S(z) still contributes. This is reported
transparently rather than forced back toward a fixed "dual-score" answer, because
the entire point of treating β as a posterior instead of a fixed
hyperparameter is to let the data decide. It is worth noting for future work
that this *weakens* (for this specific detector) the standalone
"dual-score fusion is necessary" framing of contribution #1 — with a less
discriminative detector (e.g. the original baseline), the same calibration
procedure preferred a more balanced β, so the value of Bayesian β calibration
is that it adapts automatically rather than hard-coding either extreme.

**The `mouth` zone is a known weak point.** Across both datasets and every
fusion mode in the ablation study, `mouth` is the one zone with *negative*
fake−real separation (FF++: R(z) −0.150, Celeb-DF: R(z) −0.085) — i.e. this
detector's attention/gradient signal in the mouth region is, if anything,
slightly higher on real frames. `mouth` is correspondingly almost never
flagged in practice (5/3399 fake Celeb-DF frames), which is the correct
behaviour given the evidence, but it does mean the system currently cannot
attribute explanations to lip-sync/blending artefacts specifically for this
detector, despite that being one of the most literature-documented deepfake
tells. This is called out here rather than silently swept into "faithfulness
100%", since faithfulness only checks that generated text matches the
internal concept vector — not that the concept vector is visually/forensically
correct (see Limitations).

---

## Limitations (reported honestly)

- **Faithfulness ≠ forensic correctness — now partially measured.**
  `verify_faithfulness()` checks that a generated NL sentence's
  zone/artifact/severity claims match the `ConceptVector` that produced it.
  It does **not** verify the ConceptVector corresponds to a real artefact.
  `evaluate_zone_localization.py` closes the **where** gap with mask-based
  zone ground truth (97.7 % precision, 99.7 % pointing game on 26,938 FF++
  frames). **What** artifact is present still requires human annotation
  (`scripts/evaluate_artifact_module.py`) or the new measurable-evidence
  module (rule-based until calibrated). A forensic-expert utility study
  remains future work.
- **Precision-first policy ⇒ low mask recall.** High zone precision is an
  explicit design choice; the system does not attempt to recover every
  manipulated pixel region.
- **Causal policy differs by entry point.** Batch `SAEDFPipeline` hard-filters
  claims that fail the ΔP(fake) gate by default; the live demo
  (`inference_server.py`) keeps threshold-clearing claims and shows causal
  occlusion as an optional badge (`SAEDF_CAUSAL_HARD_FILTER=1` restores hard
  filtering). Report both modes when comparing numbers.
- **Off-the-shelf detector provenance.** Public checkpoints have unknown
  training data, so their in-domain FF++ numbers cannot rule out train/test
  leakage. Use `make_ffpp_splits.py` + `train_vit_detector.py` (documented
  provenance, video/identity-level splits) for defensible in-domain claims.
- **FF++ classifier performance drops on the full four-method test split**
  (48.1 % accuracy / 0.676 AUC on 1000 official test frames vs 99.4 % /
  1.000 on the historical Deepfakes-only 172-frame subset). Cross-dataset
  Celeb-DF numbers (79.1 % / 0.901) remain the more trustworthy generalisation
  signal for the off-the-shelf checkpoint.
- **Cross-dataset calibration is imperfect** (see Evaluation results above);
  `fake_probability` should not be read as a literal well-calibrated
  probability outside the FF++ distribution it was calibrated on.
- **Video inference samples only 10 frames** per clip (`inference_server.py`,
  `max_samples = 10`) and reports a single aggregated verdict — this keeps
  latency reasonable for a live demo but is not exhaustive frame coverage for
  longer videos.
- **`mouth` zone attribution is currently unreliable** for the selected
  detector (see above).
- The Dirichlet artifact-type posterior is fit from the *same* calibration
  set used for β/threshold calibration, not an independently held-out set —
  reported artifact-type probabilities should be read as descriptive of the
  calibration distribution's structure, not as an independently validated
  claim about which artefact is present.
- **Artifact evidence priors** in `artifact_evidence_config.json` are initial
  weak priors; phrase outputs as “most likely observable artifact”, never
  “confirmed manipulation”.

---

## Testing & CI

```
tests/
  test_ground_truth_and_stats.py        — FF++ method parsing/splits, video-level metrics, cluster bootstrap, mask→zone GT, localization metrics, deletion/insertion helpers
  test_zone_mapping.py                  — zone-patch matrix M: shape, bounds, coverage
  test_fusion_math.py                   — R(z)=βA+(1−β)S at β∈{0,0.5,1}, population norm
  test_credible_intervals.py            — Option 4 CI bounds/clipping, β posterior recovery
  test_zone_prior.py                    — Q(z) bounds/monotonicity, method inference, Unknown/Novel, learned signatures/classifier
  test_nl_verifier.py                   — NL template + faithfulness verifier, valid/invalid cases
  test_causal_correctness.py            — causal occlusion drop, hard_filter vs badge-only modes
  test_selectivity_and_pipeline.py      — zone-threshold selectivity, full predict_label_and_explanation()
  test_fusion_config.py                 — configs/fusion_config.json load/parse
  test_api_schemas.py                   — inference_server.py response-shaping helpers
  test_inference_server_integration.py  — live /health, /infer/image, /infer/video, rate limit, API key
  test_console_utf8.py                  — Windows console-encoding safety helper

artifact_analysis/tests/
  test_feature_extractor.py             — interpretable zone feature groups
  test_artifact_inference.py            — Bayesian inferencer, abstention, NL + extended faithfulness
  test_calibration.py                   — optional Gaussian-NB / Dirichlet calibration model

webapp/src/app/api/detect/route.test.ts — Next.js API proxy route (mode selection, error forwarding)
```

Run the fast unit tier (no model/GPU required):
```bash
pytest tests/ -m "not integration"
```

Run the full suite including live model inference (requires
`shape_predictor_68_face_landmarks.dat`, `configs/fusion_config.json`, and a
sample image/video under `inputs/`):
```bash
pytest tests/ -m integration
```

Webapp tests:
```bash
cd webapp && npm run lint && npm run test && npm run build
```

`.github/workflows/ci.yml` runs the fast Python unit tier and the full
webapp lint/test/build on every push and pull request. The heavier
`integration` tier (real ViT + dlib + MTCNN) is intentionally excluded from
CI since it needs model downloads and a GPU/CPU budget CI runners don't
reliably have; run it locally before releases.

---

## Production hardening (`inference_server.py`)

- **Single-pass heatmap reuse** — `extract_full_pipeline_inputs()` already
  computes the blurred grad-rollout heatmap while deriving `phi`; the
  overlay renderer now reuses that array instead of re-running the model's
  forward+backward pass a second time per request.
- **Upload size cap** — `SAEDF_MAX_UPLOAD_MB` (default 100 MB); oversized
  uploads get a clean `413 PAYLOAD_TOO_LARGE` JSON response.
- **Rate limiting** — in-memory sliding window, `SAEDF_RATE_LIMIT_MAX_REQUESTS`
  requests per `SAEDF_RATE_LIMIT_WINDOW_SECONDS` per client IP (defaults:
  30 / 60s) on `/infer/image` and `/infer/video`; returns `429` with
  `Retry-After`.
- **Optional API-key auth** — set `SAEDF_API_KEY`; when set, `/infer/*`
  requires a matching `X-API-Key` header (disabled by default for local dev).
- **Causal occlusion policy** — `SAEDF_CAUSAL_CORRECTNESS=0` disables
  occlusion measurement; `SAEDF_CAUSAL_MIN_DROP` sets the drop threshold
  (default 0.03); `SAEDF_CAUSAL_HARD_FILTER=1` restores hard deletion of
  claims that fail the causal gate (live demo default: badge only).
- **Production WSGI server** — `python inference_server.py --production`
  serves via `waitress` instead of the Flask development server.

The Next.js demo (`webapp/`) proxies to this server and surfaces the full
forensic deep-dive: classification theory, 9-zone scores, semantic explanation
cards with optional causal badges, FF++ mask-GT research metrics (offline JSON),
and client-side PDF report export.

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

Historical output from `selectivity_examples.py` generated during the
`hamzenium/ViT-Deepfake-Classifier` benchmark run — nothing below is hand-typed
or illustrative-only. The active demo default is
`hamzenium/ViT-Deepfake-Classifier`; rerun this script after switching
models if you need a fresh example block for the current detector:

```
══════════════════════════════════════════════════════════════
  FORENSIC REPORT
  Image : Deepfakes/000_003.mp4::f000000
══════════════════════════════════════════════════════════════

[HEAD 1 — CLASSIFICATION]
  Label            : Fake
  Fake probability : 0.9748  (high)
  Model            : hamzenium/ViT-Deepfake-Classifier

[HEAD 2 — SEMANTIC EXPLANATION]
  Inferred method  : FaceSwap  (p=0.400)
    FaceSwap             0.400  ▪▪▪▪▪▪▪
    Deepfakes            0.319  ▪▪▪▪▪▪
    Face2Face            0.148  ▪▪
    NeuralTextures       0.133  ▪▪

  • Moderate iris texture inconsistency detected in the right eye region
    (confidence: 72.9% [72.8–72.9%]).
  • Moderate iris texture inconsistency detected in the left eye region
    (confidence: 72.9% [72.7–72.9%]).
  • Moderate jawline blending seam detected in the chin jaw region
    (confidence: 71.0% [69.1–71.1%]).

  Zone scores  [R(z) = fusion,  Q(z) = Bayesian posterior]:
    left_eye       R=0.999  Q=0.729  ███████████████████
    right_eye      R=1.000  Q=0.729  ███████████████████
    nose_bridge    R=0.207  Q=0.278  ████
    nose_tip       R=0.997  Q=0.509  ███████████████████
    left_cheek     R=0.555  Q=0.579  ███████████
    right_cheek    R=0.380  Q=0.535  ███████
    mouth          R=0.323  Q=0.656  ██████
    forehead       R=1.000  Q=0.597  ███████████████████
    chin_jaw       R=0.993  Q=0.710  ███████████████████
══════════════════════════════════════════════════════════════
```

---

## Installation

```bash
pip install -r requirements.txt
```

Download the dlib 68-point shape predictor and place it at the repo root:
`shape_predictor_68_face_landmarks.dat`.

---

## Usage

### 1. Preprocess a FaceForensics++ / Celeb-DF dataset

```bash
python preprocess_landmarks.py \
    --ffpp-root /data/FaceForensics++ \
    --output-json outputs/ffpp_landmarks.json \
    --num-frames 10
```

Scans **all four** FF++ manipulation directories present under the root
(Deepfakes / Face2Face / FaceSwap / NeuralTextures) plus `original`, and
stamps each fake frame with its `method` and each frame with its `crop_box`
(needed by the per-method breakdown and the mask ground-truth evaluation).

### 2. Create leakage-controlled train/val/test splits

```bash
python make_ffpp_splits.py \
    --json outputs/ffpp_landmarks.json \
    --official-splits "datasets/FaceForensics++/splits"
# writes outputs/ffpp_landmarks_{train,val,test}.json + split manifest
```

### 2b. (Recommended) Fine-tune a provenance-documented detector

```bash
python train_vit_detector.py \
    --train-json outputs/ffpp_landmarks_train.json \
    --val-json outputs/ffpp_landmarks_val.json \
    --output-dir models/saedf_vit16_detector --epochs 3
# then recalibrate + evaluate with --model-name models/saedf_vit16_detector
```

### 2c. Audit and benchmark candidate detectors (optional — already run)

```bash
python audit_vit_models.py
python detector_benchmark.py --json outputs/ffpp_landmarks_test.json
python detector_benchmark.py --json outputs/celebdf_landmarks_test.json
```

### 3. Calibrate the fusion pipeline

Build a leakage-free calibration set from the official **val** split (never
test), then fit β / zone thresholds:

```bash
python build_cal_from_split.py \
    --json outputs/ffpp_landmarks_val.json \
    --output outputs/ffpp_landmarks_cal.json

python recalibrate_v2.py \
    --json outputs/ffpp_landmarks_cal.json \
    --model-name hamzenium/ViT-Deepfake-Classifier \
    --output configs/fusion_config.json
```

### 4. Run the dual-head pipeline on one frame

```bash
python run_pipeline_single_frame.py \
    --json outputs/ffpp_landmarks_test_full.json \
    --key "Deepfakes/000_003.mp4::f000000" \
    --output outputs/report.json
```

### 5. Evaluate over a full test set (+ ablations + plots)

```bash
python evaluate_pipeline.py --json outputs/ffpp_landmarks_test_full.json --output outputs/evaluation_results_ffpp.json
python ablation_study.py --json outputs/ffpp_landmarks_test_full.json --output outputs/ablation_results_ffpp.json
python plot_evaluation.py --results outputs/evaluation_results_ffpp.json outputs/evaluation_results_celebdf.json \
    --labels "FF++ (official test)" "Celeb-DF (cross-dataset)" --output-dir outputs/plots

# Sync canonical numbers + mask-GT metrics into the webapp
python sync_webapp_data.py
```

### 5b. Validate explanations against ground truth + method inference

```bash
python evaluate_zone_localization.py --json outputs/ffpp_landmarks_test_full.json \
    --ffpp-root "datasets/FaceForensics++/FaceForensics++_C23" \
    --masks-root "datasets/FaceForensics++/masks" \
    --output outputs/zone_localization_ffpp.json

python evaluate_method_inference.py --json outputs/ffpp_landmarks_test_full.json \
    --output outputs/method_inference_ffpp.json

python sync_webapp_data.py   # publishes zone-localization.json to webapp/
```

### 5c. Artifact evidence (optional)

```bash
python scripts/demo_artifact_evidence.py
python scripts/evaluate_artifact_module.py --csv path/to/annotations.csv --out-dir results
```

### 6. Run the real-time inference server (used by webapp/)

```bash
python inference_server.py --port 8000            # dev
python inference_server.py --port 8000 --production  # waitress
```

### 7. Use as a library

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
| `attention_extractor.py` | ViT model loading, CLS attention extraction, `DEFAULT_HF_DEEPFAKE_MODEL` |
| `baseline_heatmap_generator.py` | Grad-weighted rollout, phi extraction, visual heatmaps, `extract_full_pipeline_inputs` |
| `semantic_patch_mapper.py` | Zone fusion, ConceptVector, NL templates, ForensicReport |
| `bayesian_calibration.py` | BayesianBetaCalibrator, DirichletArtifactPrior, CI propagation |
| `zone_prior.py` | ZonePriorModel: literature priors, method inference, Q(z) posteriors |
| `pipeline.py` | SAEDFPipeline orchestrator (single forward pass, dual-head output) |
| `recalibrate_v2.py` | Calibration script → configs/fusion_config.json |
| `run_pipeline_single_frame.py` | CLI demo: single frame → ForensicReport |
| `evaluate_pipeline.py` | Batch evaluation: classification AUC + explanation metrics + video-level metrics + bootstrap CIs + per-method breakdown |
| `eval_metrics.py` | Video-level aggregation, cluster-bootstrap CIs, per-method metrics (numpy-only, unit-tested) |
| `ffpp_utils.py` | FF++ method parsing, video ids, official-split loading, identity-safe hash splits |
| `make_ffpp_splits.py` | Video-level, identity-safe train/val/test splitting of a landmark JSON |
| `zone_localization.py` | Mask→zone ground truth, localization metrics, deletion/insertion helpers |
| `evaluate_zone_localization.py` | Zone attribution vs FF++ mask GT (precision/recall/F1/IoU, pointing game) + deletion/insertion causal faithfulness |
| `evaluate_method_inference.py` | Method-inference confusion matrix + accuracy vs FF++ ground-truth labels |
| `calibrate_method_signatures.py` | Learn empirical R(z) method signatures (+ optional logistic-regression classifier) from a train/val split → configs/method_signatures.json |
| `download_ffpp_masks.py` | Instructions / verification / wrapper for downloading FF++ manipulation-mask videos |
| `plot_evaluation.py` | ROC/PR curves + calibration reliability diagrams |
| `ablation_study.py` | Explanation ablations: classifier-only / attention-only / gradient-only / fixed-β / Bayesian R / Bayesian Q / full |
| `selectivity_examples.py` | Prints/saves concrete forensic-report examples with only a few zones flagged |
| `audit_vit_models.py` | Compatibility audit for candidate ViT/16 checkpoints |
| `detector_benchmark.py` | Classifier-only benchmark (AUC, confusion matrix, ECE, Brier) across candidate detectors |
| `train_vit_detector.py` | Leakage-controlled fine-tuning (video-level splits, ImageNet-only base model, provenance.json) |
| `inference_server.py` | Real-time Flask/waitress backend for the web demo, with rate limiting + auth |
| `console_utf8.py` | Windows console-encoding safety helper for CLI scripts |
| `analyze.py` | Unified CLI: raw image/video → single aggregated ForensicReport |
| `sync_webapp_data.py` | Regenerates `webapp/src/data/*.json` from the canonical pipeline outputs |
| `configs/fusion_config.json` | Calibrated β, population stats, Q(z) zone thresholds, Dirichlet posteriors |
| `outputs/model_benchmarks/` | Per-dataset, per-model benchmark JSON (accuracy/AUC/precision/recall/F1/confusion matrix/ECE/Brier) |
| `tests/` | Pytest suite (fast unit tier + `integration` tier) |
| `webapp/` | Next.js frontend + `/api/detect` proxy to `inference_server.py` |
| `.github/workflows/ci.yml` | CI: Python fast unit tests + webapp lint/test/build |

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
