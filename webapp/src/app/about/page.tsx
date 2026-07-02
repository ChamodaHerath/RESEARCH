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
        <p className="text-sm leading-relaxed text-slate-300 sm:text-base">
          The proliferation of generative adversarial networks and diffusion-based synthesis has
          made photorealistic deepfake content trivially accessible, posing significant risks to
          digital trust, journalism, and legal evidence integrity. While existing deepfake
          detectors report strong classification accuracy, they largely function as black boxes,
          offering little insight into the specific visual evidence supporting a verdict. This
          research proposes an{" "}
          <span className="font-semibold text-white">
            Explainable Deepfake Detection framework built on Vision Transformers
          </span>
          , which combines patch-based self-attention encoding with a semantic patch-mapping
          module that translates raw attention weights into interpretable, region-level forensic
          explanations. By aligning attention rollout maps with facial landmark zones and
          cross-referencing lighting, texture, and temporal consistency signals, the system
          produces structured forensic reports that mirror the reasoning process of a human
          digital-forensics analyst — improving transparency, auditability, and trust in
          automated deepfake detection systems.
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
          description="The ViT classifier backbone (hamzenium/ViT-Deepfake-Classifier) is a pretrained third-party checkpoint, selected from a systematic tournament of candidates (see README.md) rather than trained from scratch by this research. This project's contribution is the calibrated attention/gradient fusion and Bayesian zone-explanation layer built on top of it, plus the methodology used to choose a credible backbone in the first place."
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
