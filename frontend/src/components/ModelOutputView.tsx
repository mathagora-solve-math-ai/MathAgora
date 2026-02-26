import { useState } from "react";
import { parseStructuredSolution } from "../postprocess.ts";
import { MathText } from "./MathText.tsx";
import { ANSWER_PALETTE } from "../theme/flowmapPalette.ts";

/** Model logo path by provider prefix; GPT and GPT-Codex both use openai. */
function getModelLogo(modelId: string): string {
  if (modelId.startsWith("openai/")) return "/logos/openai.png";
  if (modelId.startsWith("anthropic/")) return "/logos/claude.png";
  if (modelId.startsWith("google/")) return "/logos/gemini.jpeg";
  if (modelId.startsWith("x-ai/")) return "/logos/grok.png";
  return "/logos/openai.png";
}

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

type Props = {
  models: ModelMeta[];
  streamStates: Record<string, StreamState>;
  onStopModel: (modelId: string) => void;
  onStopAll: () => void;
};

const StepCard = ({
  title,
  body,
  index,
}: {
  title: string;
  body: string;
  index: number;
}) => {
  const [open, setOpen] = useState(true);
  return (
    <div className="overflow-hidden rounded-lg border border-slate-200 bg-white text-sm">
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-2 text-left transition hover:bg-slate-50"
        onClick={() => setOpen((p) => !p)}
      >
        <span className="flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-slate-100 text-[10px] font-bold text-slate-500">
          {index + 1}
        </span>
        <span className="flex-1 text-xs font-medium leading-tight text-slate-800">
          {title || <span className="italic text-slate-400">...</span>}
        </span>
        <span className="text-[10px] text-slate-300">{open ? "▾" : "▸"}</span>
      </button>
      {open && body ? (
        <div className="border-t border-slate-100 px-3 pb-3 pt-2 text-xs leading-relaxed text-slate-600">
          <MathText text={body} />
        </div>
      ) : null}
    </div>
  );
};

export default function ModelOutputView({
  models,
  streamStates,
  onStopModel,
  onStopAll,
}: Props) {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div className="text-sm font-semibold text-slate-900">Model Output</div>
        <button
          className="rounded-full border border-slate-300 px-4 py-1.5 text-sm font-semibold text-slate-700 transition hover:border-slate-500 hover:bg-slate-50 active:scale-[0.98]"
          type="button"
          onClick={onStopAll}
        >
          Stop All
        </button>
      </div>
      <div className="overflow-x-auto">
      {(() => {
        const STEP_ROW_MIN_H = 72;
        const parsedByModel = models.map((model) => {
          const state = streamStates[model.modelId] ?? { status: "idle" as const, partialText: "" };
          return { model, state, parsed: parseStructuredSolution(state.partialText) };
        });
        const maxSteps = Math.max(1, ...parsedByModel.map(({ parsed }) => parsed.steps.length));
        const stepsBlockMinH = maxSteps * STEP_ROW_MIN_H;
        const finalBlockH = 56;
        const headerH = 72;
        const totalColumnMinH = headerH + stepsBlockMinH + 8 + finalBlockH;

        /** Same value → same color. 빈 값/— 는 회색, 나머지는 정렬 후 동일 팔레트. */
        const rawValues = parsedByModel.map((p) => p.parsed.finalAnswer?.trim() || "").filter(Boolean);
        const uniqueNonEmpty = [...new Set(rawValues)].sort((a, b) => a.localeCompare(b));
        const answerValueToColor = new Map<string, { border: string; bg: string }>();
        uniqueNonEmpty.forEach((v, i) => {
          answerValueToColor.set(v, ANSWER_PALETTE[i % ANSWER_PALETTE.length]);
        });
        const EMPTY_ANSWER_STYLE = { border: "#e2e8f0", bg: "#f1f5f9" };

        /** Min/max duration among done models (for speed badges). */
        const durationsMs = parsedByModel
          .map(({ state }) =>
            state.startedAtMs != null && state.finishedAtMs != null
              ? state.finishedAtMs - state.startedAtMs
              : null,
          )
          .filter((ms): ms is number => ms != null && ms > 0);
        const minDurationMs = durationsMs.length > 0 ? Math.min(...durationsMs) : null;
        const maxDurationMs = durationsMs.length > 0 ? Math.max(...durationsMs) : null;

        return (
          <div
            className="grid gap-4"
            style={{
              gridTemplateColumns: `repeat(${Math.max(models.length, 1)}, minmax(220px, 1fr))`,
              alignItems: "stretch",
              minHeight: totalColumnMinH,
            }}
          >
            {parsedByModel.map(({ model, state, parsed }) => {
              const isStreaming = state.status === "streaming";
              const isWaiting = (state.status === "idle" || isStreaming) && !state.partialText;
              const showStructured =
                state.status === "done" ||
                state.status === "stopped" ||
                parsed.steps.length > 0;

              return (
                <div
                  key={model.modelId}
                  className="flex min-h-full flex-col rounded-2xl border border-amber-100 bg-white p-4 shadow-soft"
                  style={{ minHeight: totalColumnMinH - 32 }}
                >
                  {/* Header */}
                  <div className="mb-3 flex flex-shrink-0 items-start justify-between">
                    <div className="flex items-center gap-2">
                      <img
                        src={getModelLogo(model.modelId)}
                        alt=""
                        className="h-7 w-7 shrink-0 rounded object-contain"
                      />
                      <div>
                        <div className="text-base font-semibold text-slate-900">
                          {model.displayName}
                        </div>
                        <div className="text-xs text-slate-400">{model.version}</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {isStreaming && (
                        <span className="h-3 w-3 animate-spin rounded-full border-2 border-slate-400 border-t-transparent" />
                      )}
                      {state.status === "done" && (() => {
                        const durMs =
                          state.startedAtMs != null && state.finishedAtMs != null
                            ? state.finishedAtMs - state.startedAtMs
                            : null;
                        const ratio =
                          durMs != null && minDurationMs != null && durMs > 0
                            ? durMs / minDurationMs
                            : null;
                        const isFastest = ratio != null && ratio <= 1.05;
                        const isSlowest =
                          durMs != null && maxDurationMs != null && durMs >= maxDurationMs * 0.95 && !isFastest;
                        return (
                          <span className="flex items-center gap-1.5">
                            <span className="text-xs font-semibold text-emerald-600">Done</span>
                            {durMs != null && (
                              <span className="text-[11px] font-medium text-slate-500">
                                {(durMs / 1000).toFixed(1)}s
                              </span>
                            )}
                            {ratio != null && (
                              <span
                                className={`whitespace-nowrap rounded-full px-1.5 py-0.5 text-[10px] font-semibold ${
                                  isFastest
                                    ? "bg-emerald-100 text-emerald-700"
                                    : isSlowest
                                      ? "bg-rose-100 text-rose-600"
                                      : "bg-slate-100 text-slate-500"
                                }`}
                              >
                                {isFastest ? "⚡ Fastest" : `${ratio.toFixed(1)}× slower`}
                              </span>
                            )}
                          </span>
                        );
                      })()}
                      {state.status === "error" && (
                        <span className="text-xs font-semibold text-rose-500">Error</span>
                      )}
                      {state.status === "stopped" && (
                        <span className="text-xs font-semibold text-slate-400">Stopped</span>
                      )}
                      <button
                        type="button"
                        className="rounded-full border border-slate-200 px-2.5 py-1 text-xs font-semibold text-slate-600 transition hover:border-slate-400 hover:bg-slate-50"
                        onClick={() => onStopModel(model.modelId)}
                      >
                        Stop
                      </button>
                    </div>
                  </div>

                  {/* Content */}
                  {isWaiting ? (
                    <div className="flex flex-col items-center justify-center gap-2 py-8 text-slate-400">
                      <span className="h-5 w-5 animate-spin rounded-full border-2 border-slate-300 border-t-transparent" />
                      <span className="text-xs">
                        {state.status === "idle" ? "Preparing request..." : "Waiting for response..."}
                      </span>
                    </div>
                  ) : showStructured ? (
                    <div className="flex min-h-0 flex-1 flex-col">
                      {/* Steps */}
                      <div className="flex flex-col gap-2">
                        {parsed.steps.length === 0 ? (
                          <div className="rounded-lg border border-dashed border-slate-200 px-3 py-4 text-center text-xs italic text-slate-400">
                            No solution steps
                          </div>
                        ) : null}
                        {parsed.steps.map((step, idx) => (
                          <div key={idx} style={{ minHeight: STEP_ROW_MIN_H }}>
                            <StepCard
                              index={idx}
                              title={step.title}
                              body={step.body}
                            />
                          </div>
                        ))}
                      </div>

                      {/* Spacer — pushes Answer block to bottom */}
                      <div className="flex-1" />

                      {/* Answer block */}
                      {(() => {
                        const raw = parsed.finalAnswer?.trim() || "";
                        const displayAnswer = raw || "—";
                        const color = raw ? answerValueToColor.get(raw) : null;
                        const { bg, border } = color ?? EMPTY_ANSWER_STYLE;
                        const borderColor = color ? `${border}99` : border;
                        return (
                          <div
                            className="mt-3 flex-shrink-0 rounded-xl border px-4 py-3 shadow-md"
                            style={{ backgroundColor: bg, borderColor }}
                          >
                            <div className="mb-1 text-[10px] font-bold uppercase tracking-widest text-slate-500">
                              Answer
                            </div>
                            <div className="text-base font-semibold text-slate-900">
                              {displayAnswer}
                            </div>
                          </div>
                        );
                      })()}
                      {state.errorMessage ? (
                        <div className="mt-2 flex-shrink-0 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-600">
                          {state.errorMessage}
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <div className="flex-1 rounded-xl border border-slate-200 bg-slate-950/90 p-3 font-mono text-[11px] leading-relaxed text-slate-100 overflow-auto max-h-72">
                      <div className="whitespace-pre-wrap">
                        {state.partialText}
                        {isStreaming ? "▍" : ""}
                      </div>
                      {state.errorMessage ? (
                        <div className="mt-2 text-xs text-rose-300">{state.errorMessage}</div>
                      ) : null}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        );
      })()}
      </div>
    </div>
  );
}
