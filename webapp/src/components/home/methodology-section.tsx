import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { SectionHeading } from "@/components/ui/glass-card";
import { WorkflowDiagram } from "@/components/methodology/workflow-diagram";
import { Button } from "@/components/ui/button";

export function MethodologySection() {
  return (
    <section id="methodology" className="section-pad relative">
      <div className="container-app flex flex-col gap-12">
        <SectionHeading
          eyebrow="System Pipeline"
          title="From raw media to forensic verdict"
          description="A nine-stage pipeline transforms a single image or video frame into an explainable forensic decision."
        />
        <WorkflowDiagram />
        <div className="flex justify-center">
          <Button asChild variant="outline" size="lg">
            <Link href="/methodology">
              View Full Methodology <ArrowRight className="h-4 w-4" />
            </Link>
          </Button>
        </div>
      </div>
    </section>
  );
}
