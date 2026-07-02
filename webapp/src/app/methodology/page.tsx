import { Cpu, Layers, Eye, Network, GitBranch } from "lucide-react";
import { SectionHeading, SectionBadge, GlassCard } from "@/components/ui/glass-card";
import { WorkflowDiagram } from "@/components/methodology/workflow-diagram";

const ARCHITECTURE_NOTES = [
  {
    icon: Cpu,
    title: "Vision Transformer Backbone",
    description:
      "A ViT-Base/16 encoder processes 16×16 image patches with learned positional embeddings across 12 transformer layers and 12 attention heads.",
  },
  {
    icon: Eye,
    title: "Dual Signal Extraction: A(z) and S(z)",
    description:
      "One forward pass yields two real signals from the same model: raw last-layer CLS→patch attention α (for A(z)), and a grad-weighted attention rollout φ across all 12 encoder layers, back-propagated from the fake-class logit (for S(z)).",
  },
  {
    icon: Network,
    title: "Semantic Patch Mapper",
    description:
      "A custom module (semantic_patch_mapper.py) builds a continuous overlap-fraction matrix M between the 196 ViT patches and 9 dlib-landmark-derived facial zones, then aggregates α and φ into zone-level scores A(z) and S(z).",
  },
  {
    icon: Layers,
    title: "Bayesian Calibration",
    description:
      "The fusion weight β is not hand-tuned: it is the MAP estimate of a Beta-distributed posterior fitted on 924 calibration frames (see configs/fusion_config.json), together with a Dirichlet posterior over per-zone artifact types.",
  },
  {
    icon: GitBranch,
    title: "Dual-Score Fusion R(z)",
    description:
      "R(z) = β·A_norm(z) + (1−β)·S_norm(z), where A(z) and S(z) are first population-normalised (z-score vs. real-frame baselines) before fusion — not raw spatial/temporal/frequency heuristics.",
  },
];

export default function MethodologyPage() {
  return (
    <div className="section-pad container-app flex flex-col gap-20">
      <div className="flex flex-col items-center gap-5 text-center">
        <SectionBadge>System Design</SectionBadge>
        <h1 className="max-w-3xl font-display text-3xl font-bold text-white sm:text-4xl md:text-5xl">
          A nine-stage explainable forensic pipeline
        </h1>
        <p className="max-w-2xl text-slate-400">
          Every stage of the pipeline is designed to preserve interpretability — from raw pixels
          to a structured forensic report.
        </p>
      </div>

      <WorkflowDiagram />

      <div className="flex flex-col gap-10">
        <SectionHeading
          eyebrow="Model Architecture"
          title="Under the hood"
          description="Key architectural components that enable explainable, forensic-grade classification."
        />
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {ARCHITECTURE_NOTES.map((note) => {
            const Icon = note.icon;
            return (
              <GlassCard key={note.title} className="flex flex-col gap-3 p-6">
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-primary/20 to-secondary/20 text-blue-300">
                  <Icon className="h-5 w-5" />
                </div>
                <h3 className="font-display text-base font-semibold text-white">{note.title}</h3>
                <p className="text-sm leading-relaxed text-slate-400">{note.description}</p>
              </GlassCard>
            );
          })}
        </div>
      </div>
    </div>
  );
}
