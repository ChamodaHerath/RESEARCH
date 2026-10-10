import { describe, it, expect } from "vitest";
import { sanitizePdfText } from "./pdf-text";

describe("sanitizePdfText", () => {
  it("replaces common Unicode punctuation with ASCII", () => {
    expect(sanitizePdfText("ExplainDeep — Report · Case 1")).toBe(
      "ExplainDeep - Report | Case 1"
    );
    expect(sanitizePdfText("• bullet — dash")).toBe("- bullet - dash");
    expect(sanitizePdfText("ΔP(fake)=−0.0123")).toBe(
      "DeltaP(fake)=-0.0123"
    );
  });

  it("normalizes math and Greek symbols from backend reasoning", () => {
    expect(
      sanitizePdfText("R(z) = β·A_norm(z) + (1−β)·S_norm(z); P(fake|x) ≥ 0.5")
    ).toBe(
      "R(z) = beta | A_norm(z) + (1-beta) | S_norm(z); P(fake|x) >= 0.5"
    );
    expect(sanitizePdfText("224×224 face crop")).toBe("224x224 face crop");
  });

  it("handles nullish values", () => {
    expect(sanitizePdfText(null)).toBe("N/A");
    expect(sanitizePdfText(undefined)).toBe("N/A");
  });
});
