# SAEDF: Semantic Attribution and Explainable DeepFake Detection

This document provides a deep, comprehensive overview of the research, algorithms, and system architecture underlying the SAEDF (Semantic Attribution and Explainable DeepFake) pipeline.

---

## 1. Entire System Process Step-by-Step

The system acts as a unified dual-head orchestrator, taking an image and generating both a classification decision and a rich semantic explanation from a **single forward pass**.

1. **Input & Preprocessing**: The system takes a face image (pre-cropped to 224x224) and its corresponding 68 facial landmarks (e.g., from `dlib`).
2. **ViT Forward Pass**: The image is passed through a Vision Transformer (ViT) deepfake detection model. The model outputs:
   - `fake_probability` (softmax of classification logits).
   - `alpha` ($\alpha$): The last-layer CLS-to-patch attention weights (shape 196).
   - `phi` ($\phi$): The blurred gradient-rollout patch scores representing SHAP/gradient attributions (shape 196).
3. **Semantic Patch Mapping**: The 68 landmarks are grouped into 9 semantic facial zones (e.g., left eye, mouth, chin/jaw). The system constructs a continuous overlap-fraction matrix $M$ mapping the 196 ViT patches to these 9 zones.
4. **Zone-Level Score Extraction**: Using $M$, the patch-level $\alpha$ and $\phi$ vectors are aggregated into area-normalized zone-level scores $A(z)$ and $S(z)$.
5. **Population Normalization**: $A(z)$ and $S(z)$ are Z-score normalized against real-face population statistics and squashed via a sigmoid function to produce $A_{norm}(z)$ and $S_{norm}(z) \in [0, 1]$.
6. **Bayesian Dual-Score Fusion**: The normalized scores are linearly fused using a Bayesian-calibrated weight $\beta$ to produce the final fused anomaly score $R(z)$.
7. **Manipulation Method Inference**: The pattern of $R(z)$ across all 9 zones is compared to literature-based zone signatures using cosine similarity to infer the most likely deepfake manipulation method (e.g., Deepfakes vs. Face2Face).
8. **Posterior Zone Updates & Artifact Attribution**:
   - The Bayesian zone prior $Q(z)$ is updated using the inferred method.
   - A Dirichlet artifact prior assigns specific artifact types (e.g., "blending boundary artifact") to the flagged zones based on $R(z)$.
9. **Forensic Report Generation**: The system filters zones exceeding a severity threshold and generates a structured JSON report containing the classification, verified Natural Language (NL) explanations, and 95% credible intervals for the anomaly scores.

---

## 2. Maths Behind Every Section

The pipeline bridges raw neural network signals with statistical confidence:

- **Zone-Level Aggregation**:
  To ensure large zones don't dominate small zones, the scores are area-normalized:
  $$ A(z) = \frac{\sum_i M[i,z] \cdot \alpha[i]}{\sum_i M[i,z]} \quad \text{and} \quad S(z) = \frac{\sum_i M[i,z] \cdot \phi[i]}{\sum_i M[i,z]} $$

- **Normalization**:
  $$ A_{norm}(z) = \sigma\left( \frac{A(z) - \mu_A(z)}{std_A(z)} \right) \quad \text{and} \quad S_{norm}(z) = \sigma\left( \frac{S(z) - \mu_S(z)}{std_S(z)} \right) $$

- **Bayesian Dual-Score Fusion**:
  $$ R(z) = \beta A_{norm}(z) + (1-\beta) S_{norm}(z) $$
  Where $\beta$ is a calibrated weight parameter representing the optimal blend of attention and gradient signals.

- **Method Inference**:
  The system computes cosine similarity between the observed zone anomaly vector $R$ and the prior-mean signature $sig_m$ for each known manipulation method $m$:
  $$ \cos(R, sig_m) = \frac{R \cdot sig_m}{||R|| \times ||sig_m||} $$
  These similarities are converted to a probability distribution via softmax with a temperature parameter.

- **Bayesian Zone Posterior $Q(z)$**:
  Given a literature-based prior $Beta(\alpha_m(z), \beta_m(z))$ for method $m$, the posterior mean $Q(z|m)$ after observing $R(z)$ with evidence weight $w$ is:
  $$ Q(z|m) = \frac{\alpha_m(z) + R(z) \cdot w}{\alpha_m(z) + \beta_m(z) + w} $$

---

## 3. Maths Behind the Semantic Mapping

The mapping matrix $M \in \mathbb{R}^{196 \times 9}$ handles the transition from the $14 \times 14$ ViT patch grid to the 9 semantic facial zones:

1. **Convex Hull Generation**: For each zone $z$ (e.g., mouth), the corresponding `dlib` landmarks are extracted, and a 2D geometric convex hull is formed.
2. **Continuous Overlap Calculation**: For every $16 \times 16$ pixel ViT patch $i$, the algorithm computes the exact geometric intersection area between the patch bounding box and the zone's convex hull.
3. **Fractional Assignment**:
   $$ M[i,z] = \frac{\text{Area}(patch_i \cap \text{convex\_hull}_z)}{256} $$
   Unlike traditional binary masks where a patch either belongs to a zone or doesn't, this continuous fraction ensures patches spanning multiple zones get partial credit. This mathematically eliminates the "large zone bias" where the chin/jaw (72 patches) inherently drowns out the left eye (0.75 patches).

---

## 4. How Bayesian Statistics is Used

Bayesian statistics form the core of the system's reasoning engine, deployed in three key areas:

1. **Fusion Weight Calibration ($\beta$)**: We place a uniform $Beta(1,1)$ prior over $\beta \in [0, 1]$. Using binary cross-entropy on a calibration set, we compute the posterior distribution of $\beta$. This gives us the MAP (Maximum A Posteriori) estimate and allows us to propagate uncertainty into 95% Credible Intervals for $R(z)$.
2. **Dirichlet Artifact Prior**: We encode deepfake literature into a Dirichlet prior $\alpha_k^{prior} = K - k$ for $K$ ranked artifacts. The prior is updated using pseudo-counts from fake frames. At inference, $R(z)$ acts as a temperature scalar: a high $R(z)$ concentrates probability mass on the most common artifact (e.g., "blending seam"), while a borderline $R(z)$ smoothly spreads uncertainty across alternatives.
3. **Beta-Binomial Zone Priors**: Different deepfakes leave different spatial signatures (e.g., Face2Face affects the mouth, FaceSwap affects the jaw). We encode these literature findings as Beta priors $Beta(\alpha, \beta)$. The observed $R(z)$ score acts as soft binomial evidence, updating the prior to yield the principled posterior probability $Q(z)$.

---

## 5. Advantages of Bayesian Statistics vs. Shapley Values

While Shapley values (SHAP) are popular for feature attribution, the Bayesian approach provides massive advantages for this forensic architecture:

- **Uncertainty Quantification**: Shapley values provide a deterministic point estimate. In a legal or journalistic context, point estimates can be misleading. The Bayesian approach yields full posterior distributions, outputting **95% Credible Intervals**. This tells the user not just *what* the model thinks, but *how confident* it is.
- **Integration of Domain Knowledge**: Shapley values are entirely data-driven and agnostic to the domain. Bayesian priors allow us to inject decades of forensic literature directly into the math (e.g., knowing that Face2Face specifically alters the mouth).
- **Graceful Handling of Borderline Cases**: In low-signal scenarios, Shapley values aggressively attribute noise, resulting in chaotic explanations. The Dirichlet prior acts as a regularizer—when $R(z)$ is weak, it smoothly spreads probability mass, accurately representing structural doubt rather than forcing a false positive.
- **Computational Efficiency**: Calculating exact Shapley values requires $O(2^N)$ combinatorial permutations. Even approximations (like gradient rollout) are expensive. The Bayesian Beta-Binomial updates and grid calibrations are mathematically closed-form and virtually instantaneous, allowing the explainer to run in real-time.

*(Note: The system extracts SHAP-like signals via the gradient rollout $\phi$, but immediately fuses them into the robust Bayesian probabilistic framework).*

---

## 6. System Evaluation Plan

The evaluation framework (`evaluate_pipeline.py`) deeply tests both heads of the architecture across three dimensions:

1. **Classification Metrics**: Standard machine learning evaluation (Accuracy, AUC-ROC, Precision, Recall, F1-score) to ensure detection performance remains state-of-the-art.
2. **Explanation Metrics**:
   - **Faithfulness Pass Rate**: Measures the percentage of generated Natural Language (NL) explanations that mathematically align with the classifier's internal logic.
   - **Mean Flagged Zones**: Tracks how many zones are correctly highlighted per fake frame.
   - **Zone Separation**: Calculates the difference in mean $R(z)$ scores between true Fake and true Real frames per zone.
   - **Flagging Frequency**: Distributions of which zones trigger most frequently.
3. **Bayesian Metrics**:
   - **Mean 95% CI Width**: Evaluates the average spread of the credible intervals. Narrower intervals prove the system's certainty.
   - **$\beta$ Posterior Summary**: Outputs the MAP, Mean, and CI limits of the fusion weight.

---

## 7. Highlighting System Evaluation Against Current SOTA

To highlight your evaluation plan against current systems, emphasize the **quantification of interpretability**:

- **Moving Beyond AUC**: Current systems only report AUC-ROC. This project introduces the **Faithfulness Pass Rate**, proving empirically that the explanation head is not hallucinating, but is fundamentally tethered to the classification decision.
- **Spatial Separation Proof**: Traditional systems output a noisy $224 \times 224$ heatmap. By measuring **Zone Separation** (mean $R_{fake}(z) - mean R_{real}(z)$), your evaluation mathematically proves that the model recognizes structured facial anatomy, not just random pixel noise.
- **Measuring Uncertainty**: Highlight the **Mean 95% CI width**. No current heatmap-based deepfake detector can tell you its margin of error for an attribution. This metric showcases a level of forensic reliability that SOTA models lack.

---

## 8. Advantages Over Current Systems

- **Semantic, Not Spatial**: Raw heatmaps suffer from "confirmation bias" (humans see what they want to see in a blob of color). SAEDF translates attention into semantic concepts ("Right Eye", "Mouth") and links them to exact artifacts ("iris texture inconsistency"), drastically reducing cognitive load for the reviewer.
- **Zero-Shot Method Inference**: The system can infer *which* deepfake tool (Deepfakes, FaceSwap, NeuralTextures, Face2Face) was used simply by analyzing the $R(z)$ pattern across the 9 zones, without needing to train a massive multi-class classifier.
- **Dual-Head Efficiency**: Generates both the detection and the complex semantic explanation from a **single forward pass** of the ViT. It avoids the massive computational overhead of running secondary explainer models (like LIME or heavy LLM-based captioners).
- **Geometric Continuous Mapping**: Most patch-mapping algorithms use binary thresholds. The $M$ matrix calculates continuous geometric overlap fractions, completely neutralizing the "large zone bias" that plagues other methods.

---

## 9. Other Important Points to Mention

- **Model Agnosticism**: The SAEDF framework is highly modular. While calibrated on a specific ViT, the semantic mapping and Bayesian fusion layers can be bolted onto *any* spatial deepfake detector that outputs patch-level attention or gradients.
- **Auditable Forensic Output**: The end product is a structured, JSON-serializable `ForensicReport`. Because the explanations are template-driven from Bayesian probabilities, they are 100% deterministic and legally auditable, unlike LLM-generated explanations which are prone to hallucination and non-determinism.