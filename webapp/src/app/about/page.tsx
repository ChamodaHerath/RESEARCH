import {
  BookOpenText,
  Target,
  FlaskConical,
  Database,
  Layers,
  ShieldQuestion,
  Quote,
} from "lucide-react";
import { SectionHeading, GlassCard, SectionBadge } from "@/components/ui/glass-card";

const OBJECTIVES = [
  {
    icon: Target,
    title: "Detect Manipulated Media",
    description:
      "Design a Vision Transformer-based classifier capable of distinguishing authentic media from deepfake-manipulated images and video with high accuracy.",
  },
  {
    icon: ShieldQuestion,
    title: "Move Beyond Black-Box Scores",
    description:
      "Replace opaque probability outputs with structured, human-readable forensic explanations grounded in attention analysis.",
  },
  {
    icon: Layers,
    title: "Semantic Region Mapping",
    description:
      "Map transformer attention weights onto interpretable facial-forensic zones — eyes, nose, mouth, jaw, and forehead.",
  },
  {
    icon: FlaskConical,
    title: "Cross-Dataset Generalization",
    description:
      "Evaluate robustness across multiple manipulation techniques, compression levels, and unseen generative methods.",
  },
];

// Frame counts are the exact n_total values this project actually evaluated,
// taken from outputs/evaluation_results.json and outputs/celebdf_evaluation_results.json.
// No DFDC evaluation was performed, so DFDC is intentionally not listed here.
const DATASETS = [
  { name: "FaceForensics++", detail: "172 labeled test frames evaluated (86 real / 86 fake)" },
  { name: "Celeb-DF v2", detail: "5,179 labeled test frames evaluated (1,780 real / 3,399 fake)" },
];

export default function AboutPage() {
  return (
    <div className="section-pad container-app flex flex-col gap-20">
      <div className="flex flex-col items-center gap-5 text-center">
        <SectionBadge>About the Research</SectionBadge>
        <h1 className="max-w-3xl font-display text-3xl font-bold text-white sm:text-4xl md:text-5xl">
          Rethinking deepfake detection as a{" "}
          <span className="text-gradient">forensic explanation</span> problem
        </h1>
        <p className="max-w-2xl text-slate-400">
          Most deepfake detectors output a single number. This research asks a different
          question: can a model explain <em>why</em> it believes media has been manipulated —
          in terms a forensic analyst, journalist, or judge can actually use?
        </p>
      </div>

      {/* Abstract */}
      <GlassCard className="mx-auto flex max-w-4xl flex-col gap-5 p-8 sm:p-10">
        <div className="flex items-center gap-2 text-primary">
          <BookOpenText className="h-5 w-5" />
          <span className="text-xs font-semibold uppercase tracking-widest">Research Abstract</span>
        </div>
        <p className="text-sm  align-justify leading-relaxed text-slate-300 sm:text-base">
        The rapid growth of hyper-realistic deepfakes presents a serious threat to digital trust, public communication, and forensic evidence. Although modern deepfake detectors can achieve strong classification performance, many operate as black-box systems that provide only a binary prediction without explaining the visual evidence behind the decision. This research addresses that limitation by proposing an explainable deepfake detection framework based on Vision Transformers (ViTs) and semantic facial-region analysis.
The proposed system uses a pretrained Vision Transformer classifier to estimate the probability that a face image is manipulated, while an explainability pipeline converts internal model evidence into human-readable forensic reasoning. Facial landmarks are extracted using a 68-point landmark model and mapped onto semantically meaningful facial zones such as the eyes, mouth, nose, and jawline. The framework combines two sources of ViT evidence: CLS-to-patch attention and grad-weighted attention rollout. These signals are aggregated over landmark-defined zones and fused using a Bayesian calibration module that estimates zone-level manipulation evidence with uncertainty-aware confidence intervals.
A key contribution of this work is the generation of dual-head forensic reports. The first head provides the classification result, fake probability, confidence tier, and model identity. The second head provides semantic explanations when the zone posterior Q(z) exceeds a real-face 85th-percentile calibrated threshold on Fake verdicts. Causal occlusion is measured as an optional badge (whether blanking the zone drops P(fake)), not as a hard delete of threshold-clearing claims. Zone fusion uses the β posterior mean so both attention and gradient evidence contribute.
The system is evaluated on FaceForensics++ and Celeb-DF, demonstrating strong in-domain and cross-dataset performance using the selected ViT detector. By combining transformer-based detection with Bayesian semantic explanation, this research provides an interpretable deepfake detection framework suitable for forensic analysts, researchers, and high-stakes decision-making contexts where both accuracy and accountability are required.

        </p>
      </GlassCard>

      {/* Objectives */}
      <div className="flex flex-col gap-10">
        <SectionHeading
          eyebrow="Research Objectives"
          title="Core goals of this research"
        />
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
          {OBJECTIVES.map((obj) => {
            const Icon = obj.icon;
            return (
              <GlassCard key={obj.title} className="flex gap-4 p-6">
                <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-primary/20 to-secondary/20 text-blue-300">
                  <Icon className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="font-display text-base font-semibold text-white">{obj.title}</h3>
                  <p className="mt-1.5 text-sm leading-relaxed text-slate-400">{obj.description}</p>
                </div>
              </GlassCard>
            );
          })}
        </div>
      </div>

      {/* Problem Statement */}
      <div className="grid grid-cols-1 gap-10 lg:grid-cols-2 lg:items-center">
        <div className="flex flex-col gap-4">
          <SectionBadge>Problem Statement</SectionBadge>
          <h2 className="font-display text-2xl font-bold text-white sm:text-3xl">
            Accuracy alone is no longer enough
          </h2>
          <p className="text-sm leading-relaxed text-slate-400 sm:text-base">
            Deepfake detectors are increasingly deployed in high-stakes contexts — content
            moderation, journalism verification, and legal proceedings — where a bare
            &ldquo;96% fake&rdquo; score provides no actionable evidence. Stakeholders need to
            know <em>which</em> regions were manipulated, <em>what kind</em> of artifact was
            found, and <em>how confident</em> the system is in each individual finding. Without
            this, detection systems remain difficult to audit, challenge, or trust.
          </p>
        </div>
        <GlassCard glow className="flex flex-col gap-4 p-8">
          <Quote className="h-6 w-6 text-secondary" />
          <p className="text-sm italic leading-relaxed text-slate-300 sm:text-base">
            &ldquo;A detector that cannot explain its own reasoning is of limited use to a
            forensic examiner who must justify a conclusion in a report — or a courtroom.&rdquo;
          </p>
          <span className="text-xs font-medium text-slate-500">— Research Motivation</span>
        </GlassCard>
      </div>

      {/* Datasets */}
      <div className="flex flex-col gap-10">
        <SectionHeading
          eyebrow="Data & Evaluation"
          title="Evaluated on real held-out benchmark partitions"
          description="The ViT classifier backbone (hamzenium/ViT-Deepfake-Classifier) is a pretrained third-party checkpoint, currently set to this Hugging Face ViT/16 model at the user's request rather than trained from scratch by this research. This project's contribution is the calibrated attention/gradient fusion and Bayesian zone-explanation layer built on top of it, plus the methodology used to choose a credible backbone in the first place."
        />
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
          {DATASETS.map((ds) => (
            <GlassCard key={ds.name} className="flex flex-col gap-3 p-6">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-white/[0.05] text-accent ring-1 ring-white/10">
                <Database className="h-5 w-5" />
              </div>
              <h3 className="font-display text-base font-semibold text-white">{ds.name}</h3>
              <p className="text-sm text-slate-400">{ds.detail}</p>
            </GlassCard>
          ))}
        </div>
      </div>
    </div>
  );
}
