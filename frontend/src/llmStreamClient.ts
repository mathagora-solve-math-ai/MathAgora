/** Model identifier (e.g. "openai/gpt-5-codex"). */
export type ModelId = string;

/** Input modality for solve: text-only, image crop only, or both. */
export type SolveModality = "text" | "image" | "image+text";

/** Request payload for starting a solve stream. */
export type SolveRequest = {
  problemId: string;
  problemLabel: string;
  cropImageDataUrl?: string;
  problemText: string;
  modality: SolveModality;
  models: { modelId: ModelId; displayName: string }[];
};

/** Stream chunk kinds. */
export type StreamChunk =
  | { modelId: ModelId; kind: "event"; event: "start"; timestampMs?: number }
  | { modelId: ModelId; kind: "event"; event: "done"; timestampMs?: number }
  | { modelId: ModelId; kind: "event"; event: "error"; errorMessage?: string; timestampMs?: number }
  | { modelId: ModelId; kind: "text"; text?: string; timestampMs?: number }
  | { modelId: ModelId; kind: "token"; text?: string; timestampMs?: number };

export type LLMStreamClient = {
  startSolveStream(
    req: SolveRequest,
    opts: { signal?: AbortSignal; onChunk: (chunk: StreamChunk) => void },
  ): Promise<void>;
};
