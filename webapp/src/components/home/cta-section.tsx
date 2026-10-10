"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { GithubIcon } from "@/components/icons/github-icon";

export function CtaSection() {
  return (
    <section className="section-pad relative">
      <div className="container-app">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="glass glow-border relative overflow-hidden px-8 py-16 text-center sm:px-16"
        >
          <div className="pointer-events-none absolute -top-24 left-1/2 h-72 w-[36rem] -translate-x-1/2 rounded-full bg-gradient-to-r from-primary/25 via-secondary/25 to-accent/25 blur-[100px]" />
          <div className="relative flex flex-col items-center gap-6">
            <h2 className="font-display text-3xl font-bold text-white sm:text-4xl">
              See explainable forensic AI in action
            </h2>
            <p className="max-w-xl text-slate-400">
              Upload a sample image or video and watch the full detection pipeline run
              live — from patch embedding to a downloadable forensic report.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-4">
              <Button asChild size="lg">
                <Link href="/demo">
                  Launch Live Demo <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg">
                <a href="https://github.com/ChamodaHerath" target="_blank" rel="noreferrer">
                  <GithubIcon className="h-4 w-4" /> View on GitHub
                </a>
              </Button>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
