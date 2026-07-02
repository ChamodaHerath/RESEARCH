"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, CheckCircle2 } from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import { Button } from "@/components/ui/button";

export function ContactForm() {
  const [submitted, setSubmitted] = React.useState(false);
  const [loading, setLoading] = React.useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    window.setTimeout(() => {
      setLoading(false);
      setSubmitted(true);
    }, 1100);
  };

  return (
    <GlassCard glow className="p-6 sm:p-8">
      <AnimatePresence mode="wait">
        {submitted ? (
          <motion.div
            key="success"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex h-full min-h-[360px] flex-col items-center justify-center gap-4 text-center"
          >
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-400">
              <CheckCircle2 className="h-8 w-8" />
            </div>
            <h3 className="font-display text-lg font-semibold text-white">Message Sent</h3>
            <p className="max-w-xs text-sm text-slate-400">
              Thanks for reaching out — this is a research demo, so no message was actually
              transmitted, but the form flow is fully functional.
            </p>
            <Button variant="outline" onClick={() => setSubmitted(false)}>
              Send Another
            </Button>
          </motion.div>
        ) : (
          <motion.form
            key="form"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            onSubmit={handleSubmit}
            className="flex flex-col gap-5"
          >
            <h3 className="font-display text-lg font-semibold text-white">Send a Message</h3>
            <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
              <Field label="Full Name" placeholder="Jane Doe" required />
              <Field label="Email Address" type="email" placeholder="jane@example.com" required />
            </div>
            <Field label="Subject" placeholder="Research collaboration inquiry" required />
            <div className="flex flex-col gap-2">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Message
              </label>
              <textarea
                required
                rows={6}
                placeholder="Tell us about your inquiry..."
                className="rounded-xl border border-white/10 bg-white/[0.04] px-4 py-3 text-sm text-slate-100 outline-none transition-colors placeholder:text-slate-600 focus:border-primary/60"
              />
            </div>
            <Button type="submit" size="lg" className="self-start" disabled={loading}>
              {loading ? "Sending..." : "Send Message"} <Send className="h-4 w-4" />
            </Button>
          </motion.form>
        )}
      </AnimatePresence>
    </GlassCard>
  );
}

function Field({
  label,
  ...props
}: { label: string } & React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <div className="flex flex-col gap-2">
      <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
        {label}
      </label>
      <input
        {...props}
        className="h-12 rounded-xl border border-white/10 bg-white/[0.04] px-4 text-sm text-slate-100 outline-none transition-colors placeholder:text-slate-600 focus:border-primary/60"
      />
    </div>
  );
}
