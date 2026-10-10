import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { NextRequest } from "next/server";

// Silences noisy console.error inside the route's catch-block during the
// "backend unreachable" test.
vi.spyOn(console, "error").mockImplementation(() => {});

function makeRequest(formData: FormData): NextRequest {
  return new NextRequest("http://localhost/api/detect", {
    method: "POST",
    body: formData,
  });
}

function makeUploadFormData(mode?: string): FormData {
  const fd = new FormData();
  fd.append("file", new File(["fake-bytes"], "sample.jpg", { type: "image/jpeg" }));
  if (mode !== undefined) fd.append("mode", mode);
  return fd;
}

describe("POST /api/detect", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("returns 400 when no file is uploaded", async () => {
    const { POST } = await import("./route");
    const fd = new FormData();
    fd.append("mode", "image");
    const res = await POST(makeRequest(fd));
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toMatch(/no file uploaded/i);
  });

  it("proxies to /infer/image by default and forwards a successful payload", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ prediction: "Fake", confidence: 91.2 }),
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await import("./route");
    const res = await POST(makeRequest(makeUploadFormData()));

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toEqual({ prediction: "Fake", confidence: 91.2 });

    const [url] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/infer/image");
  });

  it("proxies to /infer/video when mode=video", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ prediction: "Real", confidence: 12.0 }),
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await import("./route");
    const res = await POST(makeRequest(makeUploadFormData("video")));

    expect(res.status).toBe(200);
    const [url] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/infer/video");
  });

  it("falls back to image mode for an invalid mode value", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ prediction: "Real", confidence: 5.0 }),
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await import("./route");
    await POST(makeRequest(makeUploadFormData("not-a-real-mode")));

    const [url] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/infer/image");
  });

  it("forwards a non-ok backend response with its status and error payload", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({ error: "No face detected in frame.", code: "NO_FACE_DETECTED" }),
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await import("./route");
    const res = await POST(makeRequest(makeUploadFormData()));

    expect(res.status).toBe(422);
    const body = await res.json();
    expect(body.code).toBe("NO_FACE_DETECTED");
  });

  it("returns 502 with a helpful message when the backend is unreachable", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError("fetch failed"));
    global.fetch = fetchMock as unknown as typeof fetch;

    const { POST } = await import("./route");
    const res = await POST(makeRequest(makeUploadFormData()));

    expect(res.status).toBe(502);
    const body = await res.json();
    expect(body.code).toBe("BACKEND_UNREACHABLE");
    expect(body.error).toMatch(/inference_server\.py/i);
  });
});
