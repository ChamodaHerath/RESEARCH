import { NextRequest, NextResponse } from "next/server";
import type { DetectionMode } from "@/lib/types";

export const runtime = "nodejs";

// Real Python inference backend (inference_server.py) — wraps the actual
// SAEDFPipeline (ViT classifier + attention/gradient fusion + Bayesian zone
// priors). This route performs NO computation and fabricates NOTHING: it is
// a pure proxy. Every field in the response was computed live from the
// uploaded file by the Python process.
const INFERENCE_BACKEND_URL =
  process.env.INFERENCE_BACKEND_URL ?? "http://127.0.0.1:8000";

function isDetectionMode(value: unknown): value is DetectionMode {
  return value === "image" || value === "video";
}

export async function POST(req: NextRequest) {
  let mode: DetectionMode = "image";
  try {
    const formData = await req.formData();
    const file = formData.get("file");
    const modeRaw = formData.get("mode");
    mode = isDetectionMode(modeRaw) ? modeRaw : "image";

    if (!(file instanceof File)) {
      return NextResponse.json({ error: "No file uploaded." }, { status: 400 });
    }

    const backendForm = new FormData();
    backendForm.append("file", file, file.name);

    const endpoint = mode === "video" ? "/infer/video" : "/infer/image";

    const backendRes = await fetch(`${INFERENCE_BACKEND_URL}${endpoint}`, {
      method: "POST",
      body: backendForm,
    });

    const payload = await backendRes.json();

    if (!backendRes.ok) {
      return NextResponse.json(
        {
          error:
            payload?.error ??
            "The real detection backend could not process this file.",
          code: payload?.code ?? "BACKEND_ERROR",
        },
        { status: backendRes.status }
      );
    }

    return NextResponse.json(payload, { status: 200 });
  } catch (err) {
    const message =
      err instanceof Error && err.message.includes("fetch")
        ? "Could not reach the real inference backend. Make sure inference_server.py is running (python inference_server.py --port 8000)."
        : "Failed to process detection request.";
    return NextResponse.json({ error: message, code: "BACKEND_UNREACHABLE" }, { status: 502 });
  }
}
