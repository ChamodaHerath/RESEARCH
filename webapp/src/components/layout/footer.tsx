import Link from "next/link";
import { ScanFace, Mail, GraduationCap } from "lucide-react";
import { GithubIcon } from "@/components/icons/github-icon";

export function Footer() {
  return (
    <footer className="relative border-t border-white/10 bg-[#050816]/60">
      <div className="container-app grid grid-cols-1 gap-10 py-14 md:grid-cols-4">
        <div className="flex flex-col gap-4 md:col-span-2">
          <Link href="/" className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-[#3B82F6] to-[#8B5CF6]">
              <ScanFace className="h-4.5 w-4.5 text-white" />
            </span>
            <span className="font-display text-base font-bold text-white">
              Explain<span className="text-gradient">Deep</span>
            </span>
          </Link>
          <p className="max-w-sm text-sm leading-relaxed text-slate-400">
            Extended Explainable Vision Transformer Framework For Deepfake Detection — an AI-powered
            digital forensic research platform combining transformer-based attention
            analysis with human-readable semantic explanations.
          </p>
          <div className="flex items-center gap-3 pt-2">
            <a
              href="https://github.com/ChamodaHerath"
              target="_blank"
              rel="noreferrer"
              className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 text-slate-300 transition-colors hover:border-primary/50 hover:text-white"
            >
              <GithubIcon className="h-4 w-4" />
            </a>
            <a
              href="mailto:herathhmcd.21@uom.lk"
              className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 text-slate-300 transition-colors hover:border-primary/50 hover:text-white"
            >
              <Mail className="h-4 w-4" />
            </a>
          </div>
        </div>

        <div className="flex flex-col gap-3 text-sm">
          <h4 className="font-display text-xs font-semibold uppercase tracking-[0.2em] text-slate-300">
            Research
          </h4>
          <Link href="/about" className="text-slate-400 hover:text-white">About Research</Link>
          <Link href="/methodology" className="text-slate-400 hover:text-white">Methodology</Link>
          <Link href="/results" className="text-slate-400 hover:text-white">Results</Link>
          <Link href="/demo" className="text-slate-400 hover:text-white">Live Demo</Link>
        </div>

        <div className="flex flex-col gap-3 text-sm">
          <h4 className="font-display text-xs font-semibold uppercase tracking-[0.2em] text-slate-300">
            Project Details
          </h4>
          <div className="flex items-start gap-2 text-slate-400">
            <GraduationCap className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
            <span>Department of Computational Mathematics </span>
           
            
          </div>
          <span className="text-slate-400">Supervisor: Dr. P.D.Thalagala</span>
          <span className="text-slate-400">Student: Herath H.M.C.D 215528D</span>
          <span className="text-slate-400">Research Year: 2026</span>
        </div>
      </div>

      <div className="border-t border-white/10 py-6">
        <div className="container-app flex flex-col items-center justify-between gap-3 text-xs text-slate-500 sm:flex-row">
          <span>© {new Date().getFullYear()} ExplainDeep Research Project. All rights reserved.</span>
          <span>Built for academic research &amp; digital forensic transparency.</span>
        </div>
      </div>
    </footer>
  );
}
