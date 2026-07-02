import jsPDF from "jspdf";
import type { DetectionResponse } from "./types";

export function generateReportPdf(result: DetectionResponse) {
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 48;
  let y = 56;

  const isFake = result.prediction === "Fake";

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

  y = 118;

  function sectionTitle(title: string) {
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
    doc.setFont("helvetica", "bold");
    doc.setFontSize(9.5);
    doc.setTextColor(90, 100, 120);
    doc.text(label, margin, y);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(20, 25, 35);
    doc.text(String(value), margin + 150, y, { maxWidth: pageWidth - margin * 2 - 150 });
    y += 18;
  }

  sectionTitle("1. Case Information");
  keyValueRow("Case ID", result.report.caseId);
  keyValueRow("Date", result.report.date);
  keyValueRow("Time", result.report.time);
  keyValueRow("Model Version", result.report.modelVersion);
  keyValueRow("Processing Time", result.processing_time);
  keyValueRow("Input File", result.report.inputFile);
  keyValueRow("Hash (SHA-256)", result.report.hashValue);
  y += 8;

  sectionTitle("2. Prediction Summary");
  keyValueRow("Overall Verdict", result.prediction.toUpperCase());
  keyValueRow("Confidence", `${result.confidence.toFixed(2)}%`);
  keyValueRow("Risk Level", result.riskLevel);
  y += 8;

  if (result.classification_reasoning) {
    sectionTitle("2b. Theoretical Basis — Why Real/Fake?");
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9.5);
    doc.setTextColor(50, 58, 72);
    const summaryLines = doc.splitTextToSize(
      result.classification_reasoning.summary,
      pageWidth - margin * 2
    );
    doc.text(summaryLines, margin, y);
    y += summaryLines.length * 12 + 8;
    result.classification_reasoning.steps.forEach((step) => {
      if (y > 720) {
        doc.addPage();
        y = 56;
      }
      doc.setFont("helvetica", "bold");
      doc.setFontSize(9.5);
      doc.setTextColor(30, 41, 59);
      doc.text(`${step.stage}  [${step.value}]`, margin, y);
      y += 14;
      if (step.formula) {
        doc.setFont("courier", "normal");
        doc.setFontSize(8.5);
        doc.setTextColor(90, 100, 120);
        doc.text(step.formula, margin, y);
        y += 12;
      }
      doc.setFont("helvetica", "normal");
      doc.setFontSize(9);
      doc.setTextColor(50, 58, 72);
      const interpLines = doc.splitTextToSize(step.interpretation, pageWidth - margin * 2);
      doc.text(interpLines, margin, y);
      y += interpLines.length * 11 + 8;
    });
    y += 4;
  }

  sectionTitle("3. Evidence");
  doc.setFont("helvetica", "bold");
  doc.setFontSize(10);
  doc.setTextColor(30, 41, 59);
  doc.text("Detected Anomalies / Semantic Explanations:", margin, y);
  y += 16;
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9.5);
  doc.setTextColor(50, 58, 72);
  result.semantic_explanations.forEach((exp) => {
    const base = `• [${exp.severity.toUpperCase()}] ${exp.title} — ${exp.description} (match ${(exp.score * 100).toFixed(0)}%)`;
    const theoryBlock = exp.theory
      ? `\n  Why: ${exp.theory.summary}\n  ${exp.theory.steps.map((s) => `${s.label}: ${s.detail}`).join("\n  ")}`
      : "";
    const lines = doc.splitTextToSize(base + theoryBlock, pageWidth - margin * 2);
    if (y + lines.length * 12 > 760) {
      doc.addPage();
      y = 56;
    }
    doc.text(lines, margin, y);
    y += lines.length * 12 + 4;
  });

  y += 6;
  doc.setFont("helvetica", "bold");
  doc.setFontSize(10);
  doc.setTextColor(30, 41, 59);
  doc.text("Facial Region Anomaly Scores:", margin, y);
  y += 16;
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9.5);
  doc.setTextColor(50, 58, 72);
  result.facial_regions.forEach((r) => {
    doc.text(`• ${r.label}: ${(r.anomalyScore * 100).toFixed(0)}%`, margin, y);
    y += 14;
  });

  if (y > 700) {
    doc.addPage();
    y = 56;
  }
  y += 8;
  sectionTitle("4. Attention Analysis & Confidence Table");
  result.confidence_breakdown.forEach((c) => {
    keyValueRow(c.metric, `${c.value}%`);
  });

  if (y > 680) {
    doc.addPage();
    y = 56;
  }
  y += 8;
  sectionTitle("5. Recommendations");
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9.5);
  doc.setTextColor(50, 58, 72);
  const recommendation = isFake
    ? "This media exhibits strong indicators of synthetic manipulation. It is recommended to treat this content as unverified, flag it for manual forensic review, and avoid redistribution until authenticity is confirmed through independent verification."
    : "No significant indicators of manipulation were detected. The media is assessed as authentic with the confidence stated above. Standard verification practices are still recommended for high-stakes use cases.";
  const recLines = doc.splitTextToSize(recommendation, pageWidth - margin * 2);
  doc.text(recLines, margin, y);
  y += recLines.length * 13 + 14;

  sectionTitle("6. Disclaimer");
  doc.setFont("helvetica", "italic");
  doc.setFontSize(8.5);
  doc.setTextColor(110, 118, 132);
  const disclaimer =
    "This report is generated by an automated academic research prototype for demonstration purposes as part of a final year research project on Explainable Deepfake Detection using Vision Transformers. It is not a certified legal or forensic instrument and should not be used as sole evidence in judicial, regulatory, or security-critical decision making without independent expert verification.";
  const discLines = doc.splitTextToSize(disclaimer, pageWidth - margin * 2);
  doc.text(discLines, margin, y);

  doc.setFontSize(8);
  doc.setTextColor(150, 155, 165);
  doc.text(
    `Generated by ExplainDeep Research Platform · ${new Date().toISOString()}`,
    margin,
    doc.internal.pageSize.getHeight() - 30
  );

  doc.save(`${result.report.caseId}-forensic-report.pdf`);
}
