# Workbook Vision Lab (Prototype)

A React + TypeScript + Vite prototype that mirrors a Qanda/Gauss-style flow:
**upload/capture a workbook photo -> mock-detect problem regions -> solve with multi-LLM streaming -> align step-by-step reasoning**. All detection and LLM calls are mocked for now, with TODO placeholders where backend integrations will go.

## Features
- Upload or capture a photo (mobile-friendly capture input).
- Immediate preview with localStorage persistence.
- Mock bounding-box overlay + selectable cropped regions.
- Solve view with **two tabs**:
  - **Streaming**: raw incremental output per model.
  - **Aligned Steps**: structured steps + alignment colors.
- Step alignment coloring computed locally (no hard-coded tags).

## Tech Stack
- React + TypeScript
- Vite
- Tailwind CSS (configured in `tailwind.config.js` + `postcss.config.js`)

## Local Development
From the repo root:

```bash
cd frontend
npm install
npm run dev
```

## Remote Development (Docker + Port Forwarding)
From the repo root:

```bash
cd frontend
npm run dev:remote
```

External URL: `http://<SERVER_IP_OR_DOMAIN>:20225` (mapped to Vite `5173`).

See `docs/remote-dev.md` for full remote workflow and HMR troubleshooting.

## Project Structure
- `src/App.tsx`: Flow orchestration + state management.
- `src/components/ImageUploader.tsx`: Upload/capture + preview.
- `src/components/DetectorView.tsx`: Bounding boxes + crop selection.
- `src/components/StreamingView.tsx`: Streaming panels + live text output.
- `src/components/SolveView.tsx`: Aligned steps (strategy + accordion + final answer).
- `src/components/SelectedProblemCard.tsx`: Shared selected-problem context block.
- `src/llmStreamClient.ts`: Streaming interface types.
- `src/llmStreamClient.mock.ts`: Mock streaming implementation.
- `src/llmStreamClient.backend.ts`: Backend streaming placeholder (SSE/NDJSON).
- `src/llmStreamClient.index.ts`: One-line switch (mock vs backend).
- `src/postprocess.ts`: Parse `[STRATEGY] / [STEP] / [FINAL_ANSWER]` markers.
- `src/mockData.ts`: Mock images, detections, and solution content.
- `src/mockApi.ts`: Mock detector API with TODOs for backend integration.

## Mock Data & TODOs
Mock data lives in `src/mockData.ts`:
- `Detection` includes `{ id, x, y, w, h, label }` (pixel coordinates; relative 0..1 also supported).
- `ModelResult` includes `strategy` (required) and `steps[]` where each step has `title` + `body`.
- Alignment is **not stored** in mock data; it is computed locally at runtime.

Backend integration TODOs:
- `src/mockApi.ts`: `detectProblems(imageDataUrl)` -> replace with real detector endpoint.
- `src/llmStreamClient.backend.ts`: replace mock stream with SSE / fetch-streaming / WebSocket.
  - Expected stream chunk format: `{ modelId, kind, text?, event?, errorMessage? }`
  - `createLLMStreamClient()` in `src/llmStreamClient.index.ts` is the switch.

## Streaming + Postprocess + Alignment
- Streaming interface defined in `src/llmStreamClient.ts` with a mock implementation.
- Mock streaming uses lightweight markers:
  - `[STRATEGY]`, `[STEP title="..."]`, `[FINAL_ANSWER]`
- `src/postprocess.ts` parses streamed text into structured steps.
- Alignment is computed in `src/alignment.ts` using token overlap; threshold is adjustable.
- This whole pipeline is a placeholder for future backend streaming + embedding-based alignment.

Dataset demo (local files):
- `public/data/outputs_parsing/2024_math_odd/page_001/page_001.json` is loaded on startup.
- `public/data/2024_math_odd/page_001.png` is used as the base image.
- Replace or extend these files to point at other pages/years.

## UI Flow (Text Overview)
1) **Home / Upload** -> pick or capture a workbook image -> preview
2) **Detect/Crop** -> mock bounding boxes overlay + selectable crops
3) **Solve** -> Streaming tab (raw output) + Aligned Steps tab (strategy/steps/answer)
