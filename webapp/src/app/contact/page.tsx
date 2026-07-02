import { Mail, GraduationCap, User, Calendar } from "lucide-react";
import { SectionBadge, GlassCard } from "@/components/ui/glass-card";
import { GithubIcon } from "@/components/icons/github-icon";
import { ContactForm } from "@/components/contact/contact-form";

const INFO = [
  { icon: GraduationCap, label: "University", value: "University of Moratuwa" },
  { icon: User, label: "Supervisor", value: "Dr. P.D.Thalagala, Dept. of Computational Mathematics" },
  { icon: User, label: "Student Researcher", value: "Herath H.M.C.D 215528D" },
  { icon: Calendar, label: "Research Year", value: "2026" },
  
];

export default function ContactPage() {
  return (
    <div className="section-pad container-app flex flex-col gap-16">
      <div className="flex flex-col items-center gap-5 text-center">
        <SectionBadge>Get in Touch</SectionBadge>
        <h1 className="max-w-3xl font-display text-3xl font-bold text-white sm:text-4xl md:text-5xl">
          Questions, collaboration, or feedback?
        </h1>
        <p className="max-w-2xl text-slate-400">
          Reach out for academic collaboration, dataset access requests, or feedback on the
          research methodology.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-[1fr_1.2fr]">
        <div className="flex flex-col gap-5">
          <GlassCard className="flex flex-col gap-4 p-6">
            <h3 className="font-display text-base font-semibold text-white">Project Details</h3>
            {INFO.map((item) => {
              const Icon = item.icon;
              return (
                <div key={item.label} className="flex items-start gap-3">
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-white/[0.05] text-blue-300 ring-1 ring-white/10">
                    <Icon className="h-4 w-4" />
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-wide text-slate-500">{item.label}</p>
                    <p className="text-sm text-slate-200">{item.value}</p>
                  </div>
                </div>
              );
            })}
          </GlassCard>

          <GlassCard className="flex flex-col gap-4 p-6">
            <h3 className="font-display text-base font-semibold text-white">Direct Contact</h3>
            <a
              href="mailto:research@explaindeep.ai"
              className="flex items-center gap-3 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-slate-200 transition-colors hover:border-primary/40 hover:bg-white/[0.06]"
            >
              <Mail className="h-4 w-4 text-primary" /> herathhmcd.21@uom.lk
            </a>
            <a
              href="https://github.com/ChamodaHerath"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-3 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-slate-200 transition-colors hover:border-primary/40 hover:bg-white/[0.06]"
            >
              <GithubIcon className="h-4 w-4 text-primary" /> github.com/ChamodaHerath
            </a>
          </GlassCard>
        </div>

        <ContactForm />
      </div>
    </div>
  );
}
