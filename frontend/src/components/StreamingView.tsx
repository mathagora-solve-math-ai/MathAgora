type ModelMeta = {
  modelId: string;
  displayName: string;
  version?: string;
  temperature?: number;
  latencyMs?: number;
};

type StreamState = {
  status: "idle" | "streaming" | "done" | "error" | "stopped";
  partialText: string;
  startedAtMs?: number;
  finishedAtMs?: number;
  errorMessage?: string;
};

type StreamingViewProps = {
  models: ModelMeta[];
  streamStates: Record<string, StreamState>;
  onStopModel: (modelId: string) => void;
  onStopAll: () => void;
};

const statusLabel = (status: StreamState["status"]) => {
  switch (status) {
    case "streaming":
      return "Streaming";
    case "done":
      return "Done";
    case "error":
      return "Error";
    case "stopped":
      return "Stopped";
    default:
      return "Idle";
  }
};

export default function StreamingView({
  models,
  streamStates,
  onStopModel,
  onStopAll,
}: StreamingViewProps) {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-slate-900">Streaming</div>
          <p className="text-sm text-slate-600">
            Raw generation output arrives incrementally per model. Use the aligned
            tab for structured steps.
          </p>
        </div>
        <button
          className="rounded-full border border-slate-300 px-4 py-1.5 text-sm font-semibold text-slate-700 transition hover:border-slate-500 hover:bg-slate-50 active:scale-[0.98]"
          type="button"
          onClick={onStopAll}
        >
          Stop All
        </button>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        {models.map((model) => {
          const state = streamStates[model.modelId] ?? {
            status: "idle",
            partialText: "",
          };
          const isStreaming = state.status === "streaming";
          const waitingForFirstToken =
            (state.status === "idle" || state.status === "streaming") &&
            !state.partialText;
          return (
            <div
              key={model.modelId}
              className="flex h-full flex-col rounded-2xl border border-amber-100 bg-white p-4 shadow-soft"
            >
              <div className="flex items-start justify-between">
                <div>
                  <div className="text-lg font-semibold text-slate-900">
                    {model.displayName}
                  </div>
                  <div className="text-xs text-slate-500">{model.version}</div>
                </div>
                <div className="text-right text-xs text-slate-500">
                  <div>{model.latencyMs}ms</div>
                  <div>Temp {model.temperature}</div>
                </div>
              </div>

              <div className="mt-2 flex items-center justify-between text-xs text-slate-500">
                <span>{statusLabel(state.status)}</span>
                <button
                  type="button"
                  className="rounded-full border border-slate-200 px-3 py-1 text-xs font-semibold text-slate-600 transition hover:border-slate-400 hover:bg-slate-50"
                  onClick={() => onStopModel(model.modelId)}
                >
                  Stop
                </button>
              </div>

              <div className="mt-3 flex-1 rounded-xl border border-slate-200 bg-slate-950/90 p-3 font-mono text-[12px] leading-relaxed text-slate-100">
                {waitingForFirstToken ? (
                  <div className="flex h-full flex-col items-center justify-center gap-3 text-slate-300">
                    <span className="h-5 w-5 animate-spin rounded-full border-2 border-slate-400 border-t-transparent" />
                    <span className="text-xs">
                      {state.status === "idle"
                        ? "요청 준비 중..."
                        : "모델 응답 대기 중..."}
                    </span>
                  </div>
                ) : (
                  <div className="whitespace-pre-wrap">
                    {state.partialText}
                    {isStreaming ? "\n▍" : ""}
                  </div>
                )}
                {state.errorMessage ? (
                  <div className="mt-2 text-xs text-rose-300">
                    {state.errorMessage}
                  </div>
                ) : null}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
