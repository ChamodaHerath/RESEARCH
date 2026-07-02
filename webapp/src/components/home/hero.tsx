"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, ShieldCheck, Sparkles, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SectionBadge } from "@/components/ui/glass-card";
import { HeroIllustration } from "./hero-illustration";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, delay: i * 0.1, ease: "easeOut" as const },
  }),
};

export function Hero() {
  return (
    <section className="relative overflow-hidden pb-16 pt-16 sm:pt-24 lg:pt-28">
      <div className="container-app grid grid-cols-1 items-center gap-16 lg:grid-cols-2">
        <div className="flex flex-col gap-7">
          <motion.div variants={fadeUp} initial="hidden" animate="visible" custom={0}>
            <SectionBadge>Digital Forensics · Explainable AI</SectionBadge>
          </motion.div>

          <motion.h1
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            custom={1}
            className="font-display text-4xl font-extrabold leading-[1.08] tracking-tight text-white sm:text-5xl lg:text-[3.4rem]"
          >
            Explainable Deepfake Detection using{" "}
            <span className="animated-gradient-text">Vision Transformers</span>
          </motion.h1>

          <motion.p
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            custom={2}
            className="max-w-xl text-base leading-relaxed text-slate-400 sm:text-lg"
          >
            A digital-forensic platform that detects manipulated images and video —
            then explains <span className="text-slate-200">why</span>. Instead of a
            single probability score, our Vision Transformer pipeline generates
            semantic, human-readable forensic explanations grounded in real attention
            analysis, facial-region anomaly mapping, and calibrated Bayesian zone
            fusion — every run in the live demo is a genuine model inference, not a
            scripted playback.
          </motion.p>

          <motion.div
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            custom={3}
            className="flex flex-wrap items-center gap-4 pt-2"
          >
            <Button asChild size="lg">
              <Link href="/demo">
                Start Detection <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
            <Button asChild variant="outline" size="lg">
              <Link href="/about">Learn More</Link>
            </Button>
          </motion.div>

          <motion.div
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            custom={4}
            className="flex flex-wrap items-center gap-x-8 gap-y-4 pt-6 text-sm text-slate-400"
          >
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-primary" />
              Forensic-grade transparency
            </div>
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-secondary" />
              Semantic explanations
            </div>
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-accent" />
              Real-time attention analysis
            </div>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, ease: "easeOut", delay: 0.2 }}
        >
          <HeroIllustration />
        </motion.div>
      </div>
    </section>
  );
}
