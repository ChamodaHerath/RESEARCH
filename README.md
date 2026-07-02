# Explainable Deepfake Detection Using Vision Transformers

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
| 7 | Faithfulness verifier — every NL claim (zone, artifact, severity, confidence %, and CI bounds) audited against its source ConceptVector, with a re-verified fallback path | `semantic_patch_mapper.py` |
| 8 | Systematic detector-selection tournament (AUC/ECE/Brier/confusion matrix, 2 datasets, 4 candidate ViT/16 checkpoints) | `detector_benchmark.py`, `audit_vit_models.py` |
| 9 | 6-way ablation study (classifier-only, A(z)-only, S(z)-only, fixed-β, Bayesian R, Bayesian Q) isolating the effect of dual-score fusion and Bayesian zone priors | `ablation_study.py` |

---

## Detector: selection, not assumption

Early iterations of this project used `prithivMLmods/Deep-Fake-Detector-v2-Model`
as the classifier head, which turned out to be barely better than chance
(FF++ AUC 0.43, Celeb-DF AUC 0.56) — the verdict itself was not trustworthy,
independent of the explanation layer's quality. Rather than assume any single
public checkpoint is adequate, `audit_vit_models.py` + `detector_benchmark.py`
run a systematic tournament over drop-in-compatible (224px, 16×16-patch,
binary-label, attention-exposing) ViT checkpoints, evaluated on classifier
metrics alone (no explanation layer, so the comparison isn't confounded by
the fusion/calibration layer). Full per-model result JSON, kept as separate
files per dataset as both are independently reproducible:
`outputs/model_benchmarks/ffpp_benchmark.json` (172 frames) and
`outputs/model_benchmarks/celebdf_benchmark.json` (5,179 frames, cross-dataset).

**FaceForensics++ (in-domain, 172 frames: 86 real / 86 fake)**

| Model | Accuracy | AUC-ROC | Precision | Recall | F1 | ECE | Brier |
|---|---|---|---|---|---|---|---|
| `prithivMLmods/Deep-Fake-Detector-v2-Model` (original default) | 41.9 % | 0.429 | 0.400 | 0.326 | 0.359 | 0.236 | 0.305 |
| `prithivMLmods/Deepfake-Detection-Exp-02-21` | 49.4 % | 0.466 | 0.480 | 0.140 | 0.216 | 0.083 | 0.261 |
| `dima806/deepfake_vs_real_image_detection` | 68.6 % | 0.687 | 0.619 | 0.965 | 0.755 | 0.226 | 0.262 |
| **`hamzenium/ViT-Deepfake-Classifier`** (selected) | **99.4 %** | **1.000** | **1.000** | **0.988** | **0.994** | **0.034** | **0.004** |

**Celeb-DF v2 (cross-dataset, 5,179 frames: 1,780 real / 3,399 fake)**

| Model | Accuracy | AUC-ROC | Precision | Recall | F1 | ECE | Brier |
|---|---|---|---|---|---|---|---|
| `prithivMLmods/Deep-Fake-Detector-v2-Model` (original default) | 45.1 % | 0.558 | 0.728 | 0.262 | 0.386 | 0.357 | 0.383 |
| `prithivMLmods/Deepfake-Detection-Exp-02-21` | 34.4 % | 0.528 | 1.000 | 0.0003 | 0.001 | 0.271 | 0.299 |
| `dima806/deepfake_vs_real_image_detection` | 63.8 % | 0.509 | 0.662 | 0.915 | 0.769 | 0.309 | 0.326 |
| **`hamzenium/ViT-Deepfake-Classifier`** (selected) | **79.1 %** | **0.901** | **0.931** | **0.737** | **0.823** | **0.163** | **0.182** |

`hamzenium/ViT-Deepfake-Classifier` was selected as the production detector —
it is the only candidate that clears a credible AUC bar (> 0.9) on **both**
the in-domain and the cross-dataset partition; every other candidate,
including the "second best" `dima806` checkpoint, is barely better than
chance cross-dataset (AUC 0.51–0.53) despite reasonable-looking FF++ numbers,
which is itself a useful negative result: in-domain accuracy alone is not a
reliable proxy for a candidate detector's real-world credibility.

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
224×224/14×14-patch zone-mapping architecture without a larger refactor; it
was deprioritised once `hamzenium/ViT-Deepfake-Classifier` cleared the
credibility bar on both datasets.

Because the winning checkpoint's public performance already cleared the
credibility bar in-domain **and** cross-dataset, no additional fine-tuning
was performed — see `train_vit_detector.py` if a future iteration needs to
fine-tune a candidate that doesn't clear the bar out of the box.

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
│  Model: hamzenium/ViT-Deepfake-Classifier (HF ViT)              │
│  (configurable via SAEDF_MODEL_NAME; see detector tournament)  │
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
│    MAP β = 1.000  (mean 0.917)  [95% CI: 0.733 – 0.998]        │
│    R(z) = β·A_norm(z) + (1−β)·S_norm(z)                       │
│    (see "On β = 1.0" below — this is a genuine, reported       │
│    finding, not a bug: it changes per detector.)                │
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
└─────────────────────────────────────────────────────────────────┘
```

---

## Evaluation results

Full dual-head pipeline evaluation (classification + explanation), not just
the classifier alone. See `evaluate_pipeline.py`, results in
`outputs/evaluation_results_ffpp.json` / `outputs/evaluation_results_celebdf.json`,
plots in `outputs/plots/` (`roc_curve.png`, `pr_curve.png`, `calibration_curve.png`).

| Metric | FF++ (in-domain, 172 frames) | Celeb-DF (cross-dataset, 5179 frames) |
|---|---|---|
| Accuracy | 99.4 % | 79.1 % |
| AUC-ROC | 1.000 | 0.901 |
| Average precision | 1.000 | 0.941 |
| Precision / Recall / F1 | 1.00 / 0.988 / 0.994 | 0.931 / 0.737 / 0.822 |
| Confusion matrix (TP/FP/TN/FN) | 85 / 0 / 86 / 1 | 2504 / 186 / 1594 / 895 |
| ECE | 0.034 | 0.163 |
| Brier score | 0.004 | 0.182 |
| Mean flagged zones / fake frame | 2.97 / 9 | 1.97 / 9 |
| Faithfulness pass rate | 100 % | 100 % |

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

## On β = 1.0 (an honest finding, not a bug)

Recalibrating fusion for `hamzenium/ViT-Deepfake-Classifier` produces
**MAP β = 1.000** (mean 0.917, 95% CI [0.733, 0.998]) — the Bayesian
posterior places almost all its mass on pure attention A(z), effectively
zeroing out the gradient-rollout signal S(z). This is a real, reproducible
result of the calibration procedure, not a sign it's broken: it means this
particular detector's CLS attention is, on its own, highly discriminative
between real/fake zone patterns on the calibration set. This is reported
transparently rather than forced back toward a "dual-score" answer, because
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

- **Faithfulness ≠ forensic correctness.** `verify_faithfulness()` checks
  that a generated NL sentence's zone/artifact/severity claims match the
  `ConceptVector` that produced it. It does **not** verify that the
  `ConceptVector` itself corresponds to a real, human-visible artefact in the
  image — that would require a human/forensic-expert study, which is future
  work.
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

---

## Testing & CI

```
tests/
  test_zone_mapping.py                  — zone-patch matrix M: shape, bounds, coverage
  test_fusion_math.py                   — R(z)=βA+(1−β)S at β∈{0,0.5,1}, population norm
  test_credible_intervals.py            — Option 4 CI bounds/clipping, β posterior recovery
  test_zone_prior.py                    — Q(z) bounds/monotonicity, method inference, Unknown/Novel
  test_nl_verifier.py                   — NL template + faithfulness verifier, valid/invalid cases
  test_selectivity_and_pipeline.py      — zone-threshold selectivity, full predict_label_and_explanation()
  test_fusion_config.py                 — configs/fusion_config.json load/parse
  test_api_schemas.py                   — inference_server.py response-shaping helpers
  test_inference_server_integration.py  — live /health, /infer/image, /infer/video, rate limit, API key
  test_console_utf8.py                  — Windows console-encoding safety helper

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
- **Production WSGI server** — `python inference_server.py --production`
  serves via `waitress` instead of the Flask development server.

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

Real output from `selectivity_examples.py` (see above) — nothing below is
hand-typed or illustrative-only:

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

### 2. Audit and benchmark candidate detectors (optional — already run)

```bash
python audit_vit_models.py
python detector_benchmark.py --json outputs/ffpp_landmarks_test.json
python detector_benchmark.py --json outputs/celebdf_landmarks_test.json
```

### 3. Calibrate the fusion pipeline

```bash
python recalibrate_v2.py \
    --json outputs/ffpp_landmarks_cal.json \
    --model-name hamzenium/ViT-Deepfake-Classifier \
    --output configs/fusion_config.json
```

### 4. Run the dual-head pipeline on one frame

```bash
python run_pipeline_single_frame.py \
    --json outputs/ffpp_landmarks_test.json \
    --key "Deepfakes/000_003.mp4::f000000" \
    --output outputs/report.json
```

### 5. Evaluate over a full test set (+ ablations + plots)

```bash
python evaluate_pipeline.py --json outputs/ffpp_landmarks_test.json --output outputs/evaluation_results_ffpp.json
python ablation_study.py --json outputs/ffpp_landmarks_test.json --output outputs/ablation_results_ffpp.json
python plot_evaluation.py --results outputs/evaluation_results_ffpp.json outputs/evaluation_results_celebdf.json \
    --labels "FF++ (in-domain)" "Celeb-DF (cross-dataset)" --output-dir outputs/plots
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
| `evaluate_pipeline.py` | Batch evaluation: classification AUC + explanation metrics + raw predictions |
| `plot_evaluation.py` | ROC/PR curves + calibration reliability diagrams |
| `ablation_study.py` | Explanation ablations: classifier-only / attention-only / gradient-only / fixed-β / Bayesian R / Bayesian Q / full |
| `selectivity_examples.py` | Prints/saves concrete forensic-report examples with only a few zones flagged |
| `audit_vit_models.py` | Compatibility audit for candidate ViT/16 checkpoints |
| `detector_benchmark.py` | Classifier-only benchmark (AUC, confusion matrix, ECE, Brier) across candidate detectors |
| `train_vit_detector.py` | Fine-tuning script for a candidate detector (used only if none clear the credibility bar) |
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
