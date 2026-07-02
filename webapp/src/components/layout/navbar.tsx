"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { ScanFace, Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { GithubIcon } from "@/components/icons/github-icon";
import { cn } from "@/lib/utils";

const NAV_LINKS = [
  { href: "/", label: "Home" },
  { href: "/about", label: "About Research" },
  { href: "/demo", label: "Demo" },
  { href: "/methodology", label: "Methodology" },
  { href: "/results", label: "Results" },
  { href: "/contact", label: "Contact" },
];

export function Navbar() {
  const pathname = usePathname();
  const [scrolled, setScrolled] = React.useState(false);
  const [open, setOpen] = React.useState(false);

  React.useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={cn(
        "sticky top-0 z-50 w-full transition-all duration-300",
        scrolled ? "border-b border-white/10 bg-[#050816]/80 backdrop-blur-xl" : "border-b border-transparent bg-transparent"
      )}
    >
      <div className="container-app flex h-18 items-center justify-between py-3">
        <Link href="/" className="group flex items-center gap-3">
          <span className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-[#3B82F6] to-[#8B5CF6] shadow-[0_0_20px_-2px_rgba(59,130,246,0.8)]">
            <ScanFace className="h-5 w-5 text-white" strokeWidth={2.2} />
            <span className="absolute inset-0 rounded-xl animate-pulse-ring border border-blue-400/60" />
          </span>
          <span className="flex flex-col leading-tight">
            <span className="font-display text-[15px] font-bold tracking-wide text-white">
              Explain<span className="text-gradient">Deep</span>
            </span>
            <span className="text-[10px] font-medium uppercase tracking-[0.18em] text-slate-400">
              Vision Transformer Forensics
            </span>
          </span>
        </Link>

        <nav className="hidden items-center gap-1 lg:flex">
          {NAV_LINKS.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  "relative rounded-lg px-4 py-2 text-sm font-medium transition-colors",
                  active ? "text-white" : "text-slate-400 hover:text-white"
                )}
              >
                {active && (
                  <motion.span
                    layoutId="nav-active"
                    className="absolute inset-0 rounded-lg bg-white/[0.06] ring-1 ring-white/10"
                    transition={{ type: "spring", duration: 0.5 }}
                  />
                )}
                <span className="relative z-10">{link.label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="hidden items-center gap-3 lg:flex">
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="flex h-10 w-10 items-center justify-center rounded-lg border border-white/10 bg-white/[0.03] text-slate-300 transition-colors hover:border-primary/50 hover:text-white"
          >
            <GithubIcon className="h-4 w-4" />
          </a>
          <Button asChild size="sm">
            <Link href="/demo">Start Detection</Link>
          </Button>
        </div>

        <button
          className="flex h-10 w-10 items-center justify-center rounded-lg border border-white/10 text-slate-200 lg:hidden"
          onClick={() => setOpen((v) => !v)}
          aria-label="Toggle menu"
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden border-t border-white/10 bg-[#050816]/95 backdrop-blur-xl lg:hidden"
          >
            <div className="container-app flex flex-col gap-1 py-4">
              {NAV_LINKS.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setOpen(false)}
                  className={cn(
                    "rounded-lg px-4 py-3 text-sm font-medium",
                    pathname === link.href ? "bg-white/[0.06] text-white" : "text-slate-400"
                  )}
                >
                  {link.label}
                </Link>
              ))}
              <div className="mt-2 flex items-center gap-3 px-4">
                <a
                  href="https://github.com"
                  target="_blank"
                  rel="noreferrer"
                  className="flex h-10 w-10 items-center justify-center rounded-lg border border-white/10 text-slate-300"
                >
                  <GithubIcon className="h-4 w-4" />
                </a>
                <Button asChild size="sm" className="flex-1">
                  <Link href="/demo">Start Detection</Link>
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
