import type { LLMStreamClient, ModelId, SolveRequest, StreamChunk } from "./llmStreamClient";
import { getMockResults } from "./mockData.ts";

const randomDelay = (min = 120, max = 280) =>
  min + Math.floor(Math.random() * (max - min));

const toStructuredText = (result: {
  strategy: string;
  steps: { title: string; body: string }[];
  finalAnswer: string;
}) => {
  return JSON.stringify({
    model_name: "mock",
    steps: result.steps.map((step, idx) => ({
      step_idx: idx,
      title: step.title,
      content: step.body,
    })),
    final_answer: Number(result.finalAnswer) || 0,
  });
};

const splitIntoChunks = (text: string, size = 24) => {
  const chunks: string[] = [];
  let cursor = 0;
  while (cursor < text.length) {
    chunks.push(text.slice(cursor, cursor + size));
    cursor += size;
  }
  return chunks;
};

const emit = (
  onChunk: (chunk: StreamChunk) => void,
  chunk: StreamChunk,
) => {
  onChunk({ ...chunk, timestampMs: Date.now() });
};

export const mockStreamClient: LLMStreamClient = {
  async startSolveStream(
    req: SolveRequest,
    opts: { signal?: AbortSignal; onChunk: (chunk: StreamChunk) => void },
  ) {
    const { signal, onChunk } = opts;
    const results = getMockResults(req.problemId);
    const resultByModel = new Map<ModelId, (typeof results)[number]>();
    results.forEach((result) => {
      resultByModel.set(result.modelId, result);
    });
    const cancelled = () => signal?.aborted;

    const scheduleTasks: Array<Promise<void>> = req.models.map((model) => {
      return new Promise((resolve) => {
        const modelId = model.modelId;
        emit(onChunk, { modelId, kind: "event", event: "start" });

        const result = resultByModel.get(modelId);
        const structured = result
          ? toStructuredText(result)
          : JSON.stringify({ model_name: "mock", steps: [], final_answer: 0 });

        const chunks = splitIntoChunks(structured, 24);
        let idx = 0;

        const pushNext = () => {
          if (cancelled()) {
            emit(onChunk, {
              modelId,
              kind: "event",
              event: "error",
              errorMessage: "Stream cancelled",
            });
            resolve(undefined);
            return;
          }
          if (idx >= chunks.length) {
            emit(onChunk, { modelId, kind: "event", event: "done" });
            resolve(undefined);
            return;
          }
          emit(onChunk, { modelId, kind: "text", text: chunks[idx] });
          idx += 1;
          setTimeout(pushNext, randomDelay());
        };

        setTimeout(pushNext, randomDelay(80, 140));
      });
    });

    if (signal) {
      signal.addEventListener("abort", () => {
        req.models.forEach((model) => {
          emit(onChunk, {
            modelId: model.modelId,
            kind: "event",
            event: "error",
            errorMessage: "Stream aborted",
          });
        });
      });
    }

    await Promise.all(scheduleTasks);
  },
};
