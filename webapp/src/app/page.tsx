import { Hero } from "@/components/home/hero";
import { StatsSection } from "@/components/home/stats-section";
import { MethodologySection } from "@/components/home/methodology-section";
import { HighlightsSection } from "@/components/home/highlights-section";
import { CtaSection } from "@/components/home/cta-section";

export default function Home() {
  return (
    <>
      <Hero />
      <StatsSection />
      <MethodologySection />
      <HighlightsSection />
      <CtaSection />
    </>
  );
}
