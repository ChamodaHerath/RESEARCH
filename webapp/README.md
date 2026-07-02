# ExplainDeep — Explainable Deepfake Detection using Vision Transformers

A premium, dark-themed digital forensics web platform demonstrating an explainable
deepfake detection system powered by Vision Transformers. Built for a final year
research project, designed to look and feel like a commercial cybersecurity /
AI research platform (in the spirit of CrowdStrike Falcon, Microsoft Security,
and NVIDIA AI research showcases).

## Tech Stack

- **Framework:** Next.js 16 (App Router) + React 19 + TypeScript
- **Styling:** Tailwind CSS v4, custom glassmorphism design system
- **Animation:** Framer Motion
- **Charts:** Recharts
- **UI Primitives:** Radix UI (Select, Dialog, Tabs, Tooltip)
- **Uploads:** react-dropzone
- **PDF Reports:** jsPDF (client-side forensic report generation)

## Pages

| Route          | Description                                                        |
| -------------- | ------------------------------------------------------------------- |
| `/`            | Landing page — hero, stats, methodology preview, research highlights |
| `/about`       | Research abstract, objectives, problem statement, datasets           |
| `/demo`        | Live 3-column detection console (upload, terminal, results, report)  |
| `/methodology` | Full 9-stage pipeline diagram + architecture notes                   |
| `/results`     | Benchmark charts, confusion matrix, baseline comparison table        |
| `/contact`     | Project/team info + contact form                                     |

## Backend API — real inference, no mock data

`POST /api/detect` accepts a `multipart/form-data` payload:

```
file:  <image or video file>
mode:  "image" | "video"
```

This route is a thin proxy (`src/app/api/detect/route.ts`) to a real Python
inference server (`inference_server.py` at the repository root). There is no
mock/simulated detection logic anywhere in this app — every field in the
`DetectionResponse` (see `src/lib/types.ts`) is computed live from the
uploaded file by the actual research pipeline:

- Face detection (MTCNN) + 68-point landmarks (dlib)
- A real forward pass of the pretrained ViT deepfake classifier
  (`hamzenium/ViT-Deepfake-Classifier` — selected via a systematic
  benchmark tournament over candidate checkpoints, see the repo root
  `README.md`)
- Real CLS-attention (`A(z)`) and grad-weighted attention-rollout (`S(z)`)
  zone scores, fused via the calibrated Bayesian `β` from
  `configs/fusion_config.json`
- Real Bayesian zone-prior posteriors (`Q(z)`) and Dirichlet artifact-type
  posteriors
- A real grad-rollout heatmap rendered onto the actual aligned face crop
  (returned as a base64 JPEG — not a static placeholder image)

If no face can be detected in the uploaded file, the backend returns a 422
error instead of fabricating a result, and the UI surfaces that error.

Homepage statistics and the `/results` page are likewise sourced directly
from the real evaluation JSON files produced by `evaluate_pipeline.py`
(`src/data/eval-ffpp.json`, `src/data/eval-celebdf.json`,
`src/data/fusion-config.json`, wired through `src/lib/real-eval.ts`) — nothing
there is hand-typed or hardcoded either. These three files are regenerated
from the canonical pipeline outputs with `python sync_webapp_data.py` (run
from the repository root) whenever the detector, calibration, or evaluation
results change — never edit them by hand.

## Getting Started

Start the real inference backend first (from the repository root, with the
project's Python venv):

```bash
python inference_server.py --port 8000
```

Then, in `webapp/`:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). If the inference backend
isn't running, `/demo` will show a real error instead of fake results.

## Build

```bash
npm run build
npm run start
```
