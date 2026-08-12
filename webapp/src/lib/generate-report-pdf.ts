import jsPDF from "jspdf";
import type { DetectionResponse } from "./types";
import { sanitizePdfText } from "./pdf-text";

const MARGIN = 48;
const CONTENT_BOTTOM = 780;
const LINE_HEIGHT = 12;

function faithfulnessPct(result: DetectionResponse): string {
  const v = result.explanation_validation?.live;
  if (v) return `${(v.faithfulness_pass_rate * 100).toFixed(0)}%`;
  const fromBreakdown = result.confidence_breakdown.find((c) =>
    c.metric.toLowerCase().includes("faithfulness")
  );
  if (fromBreakdown) return `${fromBreakdown.value}%`;
  return result.semantic_explanations.length > 0 ? "100%" : "N/A";
}

function whyZeroZones(result: DetectionResponse): string {
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
              ")"
          )
          .join("; ")
      : null;

  if (!isFake) {
    return (
      "No semantic sentences were emitted (verdict is Real)." +
      (zoneList
        ? ` Elevated region scores for inspection: ${zoneList}.`
        : " Region scores are shown for inspection only.") +
      ` Fake confidence is ${pFakePct.toFixed(1)}%. We cannot claim facial zones are corrupted.`
    );
  }

  if (pFakePct <= 70) {
    return (
      "No semantic sentences were emitted." +
      (zoneList
        ? ` Candidate / elevated zones: ${zoneList}.`
        : " No zone clearly exceeded its calibrated threshold.") +
      ` Fake confidence is ${pFakePct.toFixed(1)}% (needs > 70% for NL claims).`
    );
  }

  return (
    "No semantic sentences were emitted." +
    (zoneList
      ? ` Elevated region scores: ${zoneList}.`
      : " No region crossed the calibrated Q(z) threshold.") +
    ` Fake confidence is ${pFakePct.toFixed(1)}%.`
  );
}

function imageFormat(dataUrl: string): "JPEG" | "PNG" {
  return dataUrl.startsWith("data:image/png") ? "PNG" : "JPEG";
}

function zoneStatus(r: DetectionResponse["facial_regions"][number]): string {
  if (r.flagged) return "FLAGGED";
  if (r.clears_threshold) return "clears thr";
  if (r.anomalyScore > 0.65) return "high";
  if (r.anomalyScore > 0.35) return "moderate";
  return "low";
}

export function generateReportPdf(result: DetectionResponse) {
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  let y = 56;

  const isFake = result.prediction === "Fake";
  const ranked = [...result.facial_regions].sort(
    (a, b) => b.anomalyScore - a.anomalyScore
  );
  const flaggedCount = ranked.filter((r) => r.flagged).length;
  const live = result.explanation_validation?.live;
  const maskVal = result.explanation_validation?.mask_ground_truth_validation;

  function ensureSpace(needed = 60) {
    if (y + needed > CONTENT_BOTTOM) {
      doc.addPage();
      y = 56;
    }
  }

  function writeText(
    text: string | string[],
    x: number,
    yPos: number,
    options?: Parameters<jsPDF["text"]>[3]
  ) {
    if (Array.isArray(text)) {
      doc.text(text.map((line) => sanitizePdfText(line)), x, yPos, options);
      return;
    }
    doc.text(sanitizePdfText(text), x, yPos, options);
  }

  function sectionTitle(title: string) {
    ensureSpace(44);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.setTextColor(30, 41, 59);
    writeText(title, MARGIN, y);
    doc.setDrawColor(59, 130, 246);
    doc.setLineWidth(1);
    doc.line(MARGIN, y + 6, pageWidth - MARGIN, y + 6);
    y += 24;
  }

  function keyValueRow(label: string, value: string) {
    ensureSpace(22);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(9);
    doc.setTextColor(90, 100, 120);
    writeText(label, MARGIN, y);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(20, 25, 35);
    const lines = doc.splitTextToSize(
      sanitizePdfText(value),
      pageWidth - MARGIN * 2 - 150
    );
    writeText(lines, MARGIN + 150, y);
    y += Math.max(16, lines.length * LINE_HEIGHT);
  }

  function paragraph(text: string, size = 9) {
    doc.setFont("helvetica", "normal");
    doc.setFontSize(size);
    doc.setTextColor(50, 58, 72);
    const lines = doc.splitTextToSize(
      sanitizePdfText(text),
      pageWidth - MARGIN * 2
    );
    ensureSpace(lines.length * LINE_HEIGHT + 10);
    writeText(lines, MARGIN, y);
    y += lines.length * LINE_HEIGHT + 10;
  }

  function drawTable(
    headers: string[],
    colWidths: number[],
    rows: string[][]
  ) {
    const rowHeight = 18;
    const tableWidth = colWidths.reduce((a, b) => a + b, 0);
    ensureSpace(rowHeight * (rows.length + 1) + 12);

    let x = MARGIN;
    doc.setFillColor(241, 245, 249);
    doc.rect(x, y - 11, tableWidth, rowHeight, "F");
    doc.setFont("helvetica", "bold");
    doc.setFontSize(8);
    doc.setTextColor(71, 85, 105);
    headers.forEach((header, i) => {
      writeText(header, x + 4, y);
      x += colWidths[i];
    });
    y += rowHeight - 4;

    rows.forEach((row, rowIdx) => {
      ensureSpace(rowHeight + 4);
      if (rowIdx % 2 === 0) {
        doc.setFillColor(248, 250, 252);
        doc.rect(MARGIN, y - 11, tableWidth, rowHeight, "F");
      }
      x = MARGIN;
      doc.setFont("helvetica", "normal");
      doc.setFontSize(8);
      doc.setTextColor(30, 41, 59);
      row.forEach((cell, i) => {
        const cellLines = doc.splitTextToSize(
          sanitizePdfText(cell),
          colWidths[i] - 8
        );
        writeText(cellLines[0] ?? "", x + 4, y);
        x += colWidths[i];
      });
      y += rowHeight - 2;
    });
    y += 8;
  }

  // Header band
  doc.setFillColor(5, 8, 22);
  doc.rect(0, 0, pageWidth, 88, "F");
  doc.setTextColor(255, 255, 255);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(17);
  writeText("ExplainDeep - Digital Forensic Report", MARGIN, 40);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9.5);
  doc.setTextColor(180, 190, 210);
  writeText(
    "Extended Explainable Vision Transformer Framework For Deepfake Detection",
    MARGIN,
    58
  );
  doc.setFont("helvetica", "bold");
  doc.setTextColor(isFake ? 239 : 34, isFake ? 68 : 197, isFake ? 68 : 94);
  writeText(
    `${result.prediction.toUpperCase()} | ${result.confidence.toFixed(2)}%`,
    pageWidth - MARGIN,
    40,
    { align: "right" }
  );

  y = 112;

  // 1. Case information
  sectionTitle("1. Case Information");
  keyValueRow("Case ID", result.report.caseId);
  keyValueRow("Date / Time", `${result.report.date} ${result.report.time}`);
  keyValueRow("Model", result.model);
  keyValueRow("Model Version", result.report.modelVersion);
  keyValueRow("Mode", result.mode);
  keyValueRow("Processing Time", result.processing_time);
  keyValueRow("Input File", result.report.inputFile);
  keyValueRow("Hash (SHA-256)", result.report.hashValue);
  y += 4;

  // 2. Classification head
  sectionTitle("2. Classification Head Result");
  keyValueRow("Verdict", result.prediction.toUpperCase());
  keyValueRow("Confidence", `${result.confidence.toFixed(2)}%`);
  keyValueRow("Risk Level", result.riskLevel);
  if (isFake) {
    const corrupted = ranked.filter((r) => r.flagged);
    keyValueRow(
      "Corrupted facial zones",
      corrupted.length > 0
        ? corrupted
            .map((r) => `${r.label} (${(r.anomalyScore * 100).toFixed(1)}%)`)
            .join("; ")
        : "None retained after research gates"
    );
  }

  if (result.confidence_breakdown.length > 0) {
    y += 4;
    paragraph("Pipeline confidence breakdown:", 8.5);
    drawTable(
      ["Metric", "Value"],
      [pageWidth - MARGIN * 2 - 80, 80],
      result.confidence_breakdown.map((c) => [c.metric, `${c.value}%`])
    );
  }
  y += 4;

  // 2b. Classification reasoning
  if (result.classification_reasoning) {
    sectionTitle("2b. Why This Verdict (Classification Reasoning)");
    paragraph(result.classification_reasoning.summary);
    result.classification_reasoning.steps.forEach((step, i) => {
      ensureSpace(48);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(9);
      doc.setTextColor(30, 41, 59);
      writeText(`Step ${i + 1}: ${step.stage}`, MARGIN, y);
      y += LINE_HEIGHT + 2;
      if (step.formula) {
        doc.setFont("helvetica", "normal");
        doc.setFontSize(8.5);
        doc.setTextColor(71, 85, 105);
        writeText(`Formula: ${step.formula}`, MARGIN + 12, y);
        y += LINE_HEIGHT;
      }
      doc.setFont("helvetica", "normal");
      doc.setFontSize(8.5);
      doc.setTextColor(51, 65, 85);
      writeText(`Value: ${step.value}`, MARGIN + 12, y);
      y += LINE_HEIGHT;
      paragraph(step.interpretation, 8.5);
      y += 2;
    });
  }

  // 3. Visual evidence + zones
  sectionTitle("3. Heatmap, Mask GT and Zone Scores");
  paragraph(
    "Attention heatmap and manipulation-mask ground truth (when available) are embedded below. Zone table lists R(z), Q(z), calibrated thresholds, and flag status."
  );

  const heatmapSrc = result.heatmap || result.attention_map;
  const maskGt = result.mask_ground_truth;
  const hasHeatmap = heatmapSrc?.startsWith("data:image");
  const hasMask =
    maskGt?.available && maskGt.overlay?.startsWith("data:image");

  if (hasHeatmap || hasMask) {
    ensureSpace(190);
    const imgSize = hasHeatmap && hasMask ? 170 : 200;
    const gap = 16;
    let imgX = MARGIN;

    if (hasHeatmap) {
      try {
        doc.addImage(
          heatmapSrc,
          imageFormat(heatmapSrc),
          imgX,
          y,
          imgSize,
          imgSize
        );
        doc.setFont("helvetica", "normal");
        doc.setFontSize(7.5);
        doc.setTextColor(100, 116, 139);
        writeText("Attention heatmap", imgX, y + imgSize + 12);
        imgX += imgSize + gap;
      } catch {
        paragraph("(Heatmap image could not be embedded in PDF.)", 8);
      }
    }

    if (hasMask) {
      try {
        doc.addImage(
          maskGt.overlay,
          imageFormat(maskGt.overlay),
          imgX,
          y,
          imgSize,
          imgSize
        );
        doc.setFont("helvetica", "normal");
        doc.setFontSize(7.5);
        doc.setTextColor(100, 116, 139);
        writeText("Mask GT overlay", imgX, y + imgSize + 12);
      } catch {
        paragraph("(Mask GT image could not be embedded in PDF.)", 8);
      }
    }

    y += imgSize + 24;

    if (hasMask) {
      paragraph(
        `Mask source: ${maskGt.mask_source} | coverage ${(maskGt.mask_coverage * 100).toFixed(1)}% | frame ${maskGt.frame_index}`,
        8
      );
    }
  }

  drawTable(
    ["Region", "Score", "R(z)", "Q(z)", "Threshold", "Status"],
    [118, 52, 48, 48, 58, 72],
    ranked.map((r) => [
      r.label,
      `${(r.anomalyScore * 100).toFixed(1)}%`,
      r.R != null ? `${(r.R * 100).toFixed(1)}%` : "N/A",
      r.Q != null ? `${(r.Q * 100).toFixed(1)}%` : "N/A",
      r.threshold != null ? `${(r.threshold * 100).toFixed(1)}%` : "N/A",
      r.flag_reason
        ? `${zoneStatus(r)} (${r.flag_reason})`
        : zoneStatus(r),
    ])
  );

  // 4. Semantic explanation
  sectionTitle("4. Semantic Explanation");
  keyValueRow("Faithfulness score", faithfulnessPct(result));
  keyValueRow("Zones claimed", String(flaggedCount));
  keyValueRow("NL claims", String(result.semantic_explanations.length));
  if (live) {
    keyValueRow(
      "Causal badge",
      live.causal_gate_enabled
        ? `measured (min drop ${live.causal_min_drop ?? "N/A"}; verified ${live.causal_claims_retained}/${live.causal_zones_tested})`
        : "not measured"
    );
  }
  paragraph(
    "Faithfulness: share of NL sentences that match their source zone/artifact/confidence numbers.",
    8
  );

  ensureSpace(24);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(9.5);
  doc.setTextColor(30, 41, 59);
  writeText("Sentences", MARGIN, y);
  y += 16;

  if (result.semantic_explanations.length === 0) {
    paragraph(whyZeroZones(result));
  } else {
    result.semantic_explanations.forEach((exp, idx) => {
      ensureSpace(36);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(9);
      doc.setTextColor(30, 41, 59);
      writeText(`${idx + 1}. ${exp.title}`, MARGIN, y);
      y += LINE_HEIGHT + 2;
      paragraph(exp.description, 8.5);
      let meta = `Match ${(exp.score * 100).toFixed(0)}% | severity ${exp.severity}`;
      if (exp.causal_drop != null) {
        meta += ` | causal dP(fake)=-${exp.causal_drop.toFixed(4)}`;
      }
      paragraph(meta, 8);
      if (exp.theory) {
        paragraph(
          `Theory: ${exp.theory.summary} | R=${exp.theory.scores.R.toFixed(4)}` +
            (exp.theory.scores.Q != null
              ? ` | Q=${exp.theory.scores.Q.toFixed(4)}`
              : ""),
          8
        );
      }
      y += 4;
    });
  }

  // 5. Live validation
  if (live) {
    sectionTitle("5. Explanation Validation (This Request)");
    keyValueRow("Claims", String(live.n_claims));
    keyValueRow("Zones causally tested", String(live.causal_zones_tested));
    keyValueRow(
      "Causal-verified (badge)",
      live.causal_retention_rate != null
        ? `${live.causal_claims_retained} (${(live.causal_retention_rate * 100).toFixed(0)}%)`
        : String(live.causal_claims_retained)
    );
    paragraph(live.note, 8.5);
    if (live.per_claim.length > 0) {
      drawTable(
        ["Zone", "Artifact", "Causal drop", "Causal pass"],
        [110, 110, 90, 80],
        live.per_claim.map((c) => [
          c.zone,
          c.artifact_type,
          c.causal_drop != null ? c.causal_drop.toFixed(4) : "N/A",
          c.causal_passed != null ? String(c.causal_passed) : "N/A",
        ])
      );
    }
  }

  if (maskVal) {
    sectionTitle("6. Offline FF++ Mask-GT Research Metrics");
    keyValueRow("Dataset", maskVal.dataset);
    if (maskVal.n_frames_evaluated != null) {
      keyValueRow("Frames evaluated", String(maskVal.n_frames_evaluated));
    }
    if (maskVal.precision != null) {
      keyValueRow("Precision", `${(maskVal.precision * 100).toFixed(2)}%`);
    }
    if (maskVal.pointing_game_accuracy != null) {
      keyValueRow(
        "Pointing-game accuracy",
        `${(maskVal.pointing_game_accuracy * 100).toFixed(2)}%`
      );
    }
    if (maskVal.zone_score_auc_micro != null) {
      keyValueRow(
        "Zone-score AUC (micro)",
        maskVal.zone_score_auc_micro.toFixed(4)
      );
    }
    paragraph(maskVal.scope_note, 8.5);
  }

  if (result.explanation_validation?.novel_contribution_note) {
    paragraph(result.explanation_validation.novel_contribution_note, 8.5);
  }

  if (result.mode === "video" && result.timeline.length > 0) {
    sectionTitle("7. Video Frame Timeline");
    drawTable(
      ["Frame", "Time", "P(fake)", "Note"],
      [52, 52, 58, pageWidth - MARGIN * 2 - 162],
      result.timeline.map((t) => [
        String(t.frame),
        `${t.timeSec.toFixed(2)}s`,
        `${(t.suspicion * 100).toFixed(1)}%`,
        t.note,
      ])
    );
  }

  // Page footers
  const totalPages = doc.getNumberOfPages();
  for (let page = 1; page <= totalPages; page += 1) {
    doc.setPage(page);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(8);
    doc.setTextColor(148, 155, 165);
    const footerY = pageHeight - 28;
    writeText(
      `ExplainDeep Forensic Report | ${result.report.caseId} | Page ${page} of ${totalPages}`,
      MARGIN,
      footerY
    );
    writeText(new Date().toISOString(), pageWidth - MARGIN, footerY, {
      align: "right",
    });
  }

  doc.save(`${result.report.caseId}-forensic-report.pdf`);
}
