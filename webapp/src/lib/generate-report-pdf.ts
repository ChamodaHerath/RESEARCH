import jsPDF from "jspdf";
import type { DetectionResponse } from "./types";

function faithfulnessPct(result: DetectionResponse): string {
  const v = result.explanation_validation?.live;
  if (v) return `${(v.faithfulness_pass_rate * 100).toFixed(0)}%`;
  const fromBreakdown = result.confidence_breakdown.find((c) =>
    c.metric.toLowerCase().includes("faithfulness")
  );
  if (fromBreakdown) return `${fromBreakdown.value}%`;
  return result.semantic_explanations.length > 0 ? "100%" : "—";
}

function whyZeroZones(result: DetectionResponse): string {
  const live = result.explanation_validation?.live;
  const isFake = result.prediction === "Fake";
  const confPct = result.confidence;
  const pFakePct = isFake ? confPct : Math.max(0, 100 - confPct);

  const elevated = [...result.facial_regions]
    .filter(
      (r) =>
        r.clears_threshold === true ||
        r.flagged === true ||
        (r.threshold != null && r.anomalyScore > r.threshold) ||
        r.anomalyScore > 0.65
    )
    .sort((a, b) => b.anomalyScore - a.anomalyScore)
    .slice(0, 3);

  const zoneList =
    elevated.length > 0
      ? elevated
          .map(
            (r) =>
              `${r.label} (${(r.anomalyScore * 100).toFixed(1)}%` +
              (r.threshold != null
                ? `; thr ${(r.threshold * 100).toFixed(1)}%`
                : "") +
              `)`
          )
          .join("; ")
      : null;

  const caveat =
    ` Fake confidence is ${pFakePct.toFixed(1)}%. ` +
    `Because the research policy withholds natural-language claims under these conditions, ` +
    `we cannot fully claim those zones are corrupted.`;

  if (!isFake) {
    return (
      `No semantic sentences were emitted (verdict is Real).` +
      (zoneList
        ? ` Elevated region scores for inspection: ${zoneList}.`
        : ` Region scores are shown for inspection only.`) +
      ` Fake confidence is ${pFakePct.toFixed(1)}%. We cannot claim facial zones are corrupted.`
    );
  }

  if (pFakePct <= 70) {
    return (
      `No semantic sentences were emitted.` +
      (zoneList
        ? ` Candidate / elevated zones: ${zoneList}.`
        : ` No zone clearly exceeded its calibrated threshold.`) +
      caveat
    );
  }

  if (
    live?.causal_gate_enabled &&
    (live.causal_zones_tested ?? 0) > 0 &&
    live.causal_claims_retained === 0
  ) {
    return (
      `No semantic sentences were retained: candidate zones were dropped by the causal gate ` +
      `(occlusion did not reduce P(fake) by ≥ ${live.causal_min_drop ?? 0.03}).` +
      (zoneList ? ` Zones considered: ${zoneList}.` : "") +
      caveat
    );
  }

  return (
    `No semantic sentences were emitted.` +
    (zoneList
      ? ` Elevated region scores: ${zoneList}.`
      : ` No region crossed the calibrated Q(z) threshold.`) +
    caveat
  );
}

export function generateReportPdf(result: DetectionResponse) {
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 48;
  let y = 56;

  const isFake = result.prediction === "Fake";
  const ranked = [...result.facial_regions].sort(
    (a, b) => b.anomalyScore - a.anomalyScore
  );
  const flaggedCount = ranked.filter((r) => r.flagged).length;

  doc.setFillColor(5, 8, 22);
  doc.rect(0, 0, pageWidth, 90, "F");
  doc.setTextColor(255, 255, 255);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(18);
  doc.text("ExplainDeep — Digital Forensic Report", margin, 42);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(10);
  doc.setTextColor(180, 190, 210);
  doc.text("Explainable Deepfake Detection using Vision Transformers", margin, 60);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(isFake ? 239 : 34, isFake ? 68 : 197, isFake ? 68 : 94);
  doc.text(
    `${result.prediction.toUpperCase()} · ${result.confidence.toFixed(2)}%`,
    pageWidth - margin,
    42,
    { align: "right" }
  );

  y = 118;

  function ensureSpace(needed = 60) {
    if (y + needed > 760) {
      doc.addPage();
      y = 56;
    }
  }

  function sectionTitle(title: string) {
    ensureSpace(40);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12.5);
    doc.setTextColor(30, 41, 59);
    doc.text(title, margin, y);
    doc.setDrawColor(59, 130, 246);
    doc.setLineWidth(1.2);
    doc.line(margin, y + 5, pageWidth - margin, y + 5);
    y += 22;
  }

  function keyValueRow(label: string, value: string) {
    ensureSpace(24);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(9.5);
    doc.setTextColor(90, 100, 120);
    doc.text(label, margin, y);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(20, 25, 35);
    doc.text(String(value), margin + 160, y, {
      maxWidth: pageWidth - margin * 2 - 160,
    });
    y += 18;
  }

  function paragraph(text: string, size = 9.5) {
    doc.setFont("helvetica", "normal");
    doc.setFontSize(size);
    doc.setTextColor(50, 58, 72);
    const lines = doc.splitTextToSize(text, pageWidth - margin * 2);
    ensureSpace(lines.length * 12 + 8);
    doc.text(lines, margin, y);
    y += lines.length * 12 + 8;
  }

  // 1. Case information
  sectionTitle("1. Case Information");
  keyValueRow("Case ID", result.report.caseId);
  keyValueRow("Date", result.report.date);
  keyValueRow("Time", result.report.time);
  keyValueRow("Model Version", result.report.modelVersion);
  keyValueRow("Processing Time", result.processing_time);
  keyValueRow("Input File", result.report.inputFile);
  keyValueRow("Hash (SHA-256)", result.report.hashValue);
  y += 6;

  // 2. Classification head
  sectionTitle("2. Classification Head Result");
  keyValueRow("Verdict", result.prediction.toUpperCase());
  keyValueRow("Confidence", `${result.confidence.toFixed(2)}%`);
  keyValueRow("Risk Level", result.riskLevel);
  y += 6;

  // 3. Heatmap + facial regions
  sectionTitle("3. Heatmap & Detected Facial Regions");
  paragraph(
    "Attention heatmap is included in the on-screen report. Region scores and calibrated thresholds below (higher score = more likely manipulated)."
  );

  // Embed heatmap if it is a data URL
  const heatmapSrc = result.heatmap || result.attention_map;
  if (heatmapSrc?.startsWith("data:image")) {
    try {
      ensureSpace(200);
      const imgW = 220;
      const imgH = 220;
      doc.addImage(heatmapSrc, "JPEG", margin, y, imgW, imgH);
      y += imgH + 12;
    } catch {
      paragraph("(Heatmap image could not be embedded in PDF; see on-screen report.)", 8.5);
    }
  }

  const maskGt = result.mask_ground_truth;
  if (maskGt?.available && maskGt.overlay?.startsWith("data:image")) {
    paragraph(
      "Manipulation-mask ground truth (green = true edited pixels from FaceForensics++):",
      8.5
    );
    try {
      ensureSpace(200);
      const imgW = 220;
      const imgH = 220;
      doc.addImage(maskGt.overlay, "JPEG", margin, y, imgW, imgH);
      y += imgH + 8;
      if (maskGt.mask_source) {
        paragraph(
          `Source: ${maskGt.mask_source} · coverage ${(maskGt.mask_coverage * 100).toFixed(1)}%`,
          8
        );
      }
    } catch {
      paragraph("(Mask GT image could not be embedded in PDF; see on-screen report.)", 8.5);
    }
  }

  ranked.forEach((r) => {
    const status = r.flagged
      ? "FLAGGED"
      : r.anomalyScore > 0.65
        ? "high"
        : r.anomalyScore > 0.35
          ? "moderate"
          : "low";
    const thr =
      r.threshold != null ? ` thr ${(r.threshold * 100).toFixed(1)}%` : "";
    keyValueRow(
      r.label,
      `${(r.anomalyScore * 100).toFixed(1)}%${thr} (${status})`
    );
  });
  y += 6;

  // 4. Semantic explanation
  sectionTitle("4. Semantic Explanation");
  keyValueRow("Faithfulness score", faithfulnessPct(result));
  keyValueRow("Zones flagged", String(flaggedCount));
  keyValueRow("NL claims", String(result.semantic_explanations.length));
  y += 4;

  doc.setFont("helvetica", "bold");
  doc.setFontSize(10);
  doc.setTextColor(30, 41, 59);
  ensureSpace(24);
  doc.text("Sentences", margin, y);
  y += 16;

  if (result.semantic_explanations.length === 0) {
    paragraph(whyZeroZones(result));
  } else {
    result.semantic_explanations.forEach((exp) => {
      paragraph(
        `• ${exp.title} — ${exp.description} (match ${(exp.score * 100).toFixed(0)}%; faithfulness PASS)`
      );
    });
  }

  doc.setFontSize(8);
  doc.setTextColor(150, 155, 165);
  doc.text(
    `Generated by ExplainDeep · ${new Date().toISOString()}`,
    margin,
    doc.internal.pageSize.getHeight() - 30
  );

  doc.save(`${result.report.caseId}-forensic-report.pdf`);
}
