/** Map Unicode punctuation and symbols to WinAnsi-safe ASCII for jsPDF built-in fonts. */
const PDF_CHAR_REPLACEMENTS: ReadonlyArray<[string, string]> = [
  ["\u2014", " - "], // em dash
  ["\u2013", "-"], // en dash
  ["\u2212", "-"], // minus sign
  ["\u2022", "-"], // bullet
  ["\u00b7", " | "], // middle dot
  ["\u2026", "..."], // ellipsis
  ["\u2018", "'"],
  ["\u2019", "'"],
  ["\u201c", '"'],
  ["\u201d", '"'],
  ["\u0394", "Delta"], // Greek Delta
  ["\u03b2", "beta"],
  ["\u03b1", "alpha"],
  ["\u00d7", "x"], // multiplication sign
  ["\u2265", ">="],
  ["\u2264", "<="],
  ["\u2192", "->"],
  ["\u2190", "<-"],
  ["\u00a0", " "], // non-breaking space
  ["\u00a7", "Sec."],
  ["\u00b1", "+/-"],
  ["\u2032", "'"],
  ["\u2033", '"'],
];

/**
 * Normalize text for jsPDF standard fonts (Helvetica), which only render WinAnsi
 * reliably. Backend/UI strings often contain em dashes, bullets, Greek letters,
 * and math symbols that otherwise appear as garbled glyphs in the PDF.
 */
export function sanitizePdfText(
  input: string | number | null | undefined
): string {
  if (input == null) return "N/A";
  let text = String(input);
  for (const [from, to] of PDF_CHAR_REPLACEMENTS) {
    text = text.split(from).join(to);
  }
  text = text.replace(/[^\u0020-\u007E\u00A0-\u00FF]/g, (ch) => {
    const ascii = ch.normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
    return /^[\u0020-\u007E\u00A0-\u00FF]+$/.test(ascii) ? ascii : "";
  });
  text = text.replace(/[ \t]+/g, " ").trim();
  return text || "N/A";
}
