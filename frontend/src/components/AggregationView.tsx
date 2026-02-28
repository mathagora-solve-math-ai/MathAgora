import { useState } from "react";
import { MathText } from "./MathText.tsx";

export type AggregationData = {
  steps: { step_idx: number; title: string; content: string }[];
  final_answer: number | string | null;
  rationale: string;
  confidence: "high" | "medium" | "low";
};

export type ModelAnswerInfo = {
  name: string;
  modelId: string;
  answer: string;
};

type Props = {
  data: AggregationData | null;
  isGenerating: boolean;
  onRegenerate: () => void;
  modelAnswers?: ModelAnswerInfo[];
};

const base = (import.meta.env?.BASE_URL ?? "/").replace(/\/$/, "");
function getModelLogo(modelId: string): string {
  if (modelId.startsWith("openai/")) return `${base}/logos/openai.png`;
  if (modelId.startsWith("anthropic/")) return `${base}/logos/claude.png`;
  if (modelId.startsWith("google/")) return `${base}/logos/gemini.jpeg`;
  if (modelId.startsWith("x-ai/")) return `${base}/logos/grok.png`;
  return `${base}/logos/openai.png`;
}

export default function AggregationView({ data, isGenerating, onRegenerate, modelAnswers }: Props) {
  const [expanded, setExpanded] = useState(true);

  if (isGenerating && !data) {
    return (
      <div className="flex items-center gap-3 rounded-2xl border-2 border-amber-200 bg-white px-5 py-4 shadow-lg">
        <span className="h-5 w-5 shrink-0 animate-spin rounded-full border-[2.5px] border-amber-400 border-t-transparent" />
        <span className="text-sm font-medium text-amber-700">
          Aggregating — synthesizing the best answer from all model solutions…
        </span>
      </div>
    );
  }

  if (!data) return null;

  const displayAnswer =
    data.final_answer !== null && data.final_answer !== undefined
      ? String(data.final_answer).trim() || "—"
      : "—";

  const hasAnswer = displayAnswer !== "—";

  return (
    <div className="flex flex-col gap-4 rounded-2xl border-2 border-amber-200 bg-white p-5 shadow-lg">
      {/* ── Header ── */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-violet-600 text-white shadow-sm">
            <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
              <path
                fillRule="evenodd"
                d="M10 1a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-1.5 0v-1.5A.75.75 0 0 1 10 1ZM5.05 3.05a.75.75 0 0 1 1.06 0l1.062 1.06A.75.75 0 1 1 6.11 5.173L5.05 4.11a.75.75 0 0 1 0-1.06Zm9.9 0a.75.75 0 0 1 0 1.06l-1.06 1.062a.75.75 0 0 1-1.062-1.061L13.89 3.05a.75.75 0 0 1 1.06 0ZM10 7a3 3 0 1 0 0 6 3 3 0 0 0 0-6Zm-7 3a.75.75 0 0 1 .75-.75h1.5a.75.75 0 0 1 0 1.5h-1.5A.75.75 0 0 1 3 10Zm12.25-.75h1.5a.75.75 0 0 1 0 1.5h-1.5a.75.75 0 0 1 0-1.5ZM5.05 14.888a.75.75 0 0 1 0-1.06l1.06-1.062a.75.75 0 0 1 1.062 1.061l-1.061 1.062a.75.75 0 0 1-1.061 0Zm8.838 0a.75.75 0 0 1-1.061 0l-1.062-1.061a.75.75 0 1 1 1.061-1.062l1.062 1.061a.75.75 0 0 1 0 1.062ZM10 17.25a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-1.5 0v-1.5A.75.75 0 0 1 10 17.25Z"
                clipRule="evenodd"
              />
            </svg>
          </div>
          <div>
            <h3 className="text-base font-semibold text-slate-900">Final Answer</h3>
            <p className="text-xs text-slate-500">Synthesized from all model solutions</p>
          </div>
        </div>
        <span className="text-xs text-slate-400">Claude Opus 4.5 · aggregation</span>
      </div>

      {/* ── Rationale ── */}
      {data.rationale ? (
        <p className="rounded-xl bg-slate-50 px-4 py-2.5 text-sm leading-relaxed text-slate-600 border border-slate-100">
          <span className="font-medium text-slate-700">Rationale: </span>
          <MathText text={data.rationale} />
        </p>
      ) : null}

      {/* ── Steps toggle ── */}
      {data.steps?.length > 0 ? (
        <>
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="flex w-full items-center justify-between rounded-xl border border-amber-100 bg-amber-50/60 px-4 py-2.5 text-left text-sm font-medium text-slate-700 transition hover:bg-amber-50"
          >
            <span>Synthesized solution ({data.steps.length} steps)</span>
            <svg
              viewBox="0 0 20 20"
              fill="currentColor"
              className={`h-4 w-4 shrink-0 transition-transform ${expanded ? "rotate-180" : ""}`}
            >
              <path
                fillRule="evenodd"
                d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z"
                clipRule="evenodd"
              />
            </svg>
          </button>

          {expanded ? (
            <ol className="flex flex-col gap-2">
              {data.steps.map((step, idx) => {
                return (
                  <li
                    key={step.step_idx ?? idx}
                    className="flex gap-3 rounded-lg border border-slate-100 bg-white px-3 py-2.5 shadow-sm"
                  >
                    <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-slate-100 text-xs font-semibold text-slate-500">
                      {idx + 1}
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-semibold text-slate-800">
                        <MathText text={step.title} />
                      </p>
                      <p className="mt-0.5 text-sm leading-relaxed text-slate-600">
                        <MathText text={step.content} />
                      </p>
                    </div>
                  </li>
                );
              })}
            </ol>
          ) : null}
        </>
      ) : null}

      {/* ── Answer + model agreement ── */}
      <div className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 shadow-md">
        <div className="flex items-center justify-between gap-4">
          {/* Answer value */}
          <div className="flex items-center gap-3">
            <span className="text-[10px] font-bold uppercase tracking-widest text-amber-600">
              Answer
            </span>
            <span className="text-2xl font-bold text-slate-900">{displayAnswer}</span>
          </div>

          {/* Model agreement logos */}
          {modelAnswers && modelAnswers.length > 0 && hasAnswer && (
            <div className="flex items-center gap-1">
              {modelAnswers.map((m) => {
                const matches = m.answer !== "" && m.answer === displayAnswer;
                return (
                  <div
                    key={m.name}
                    className="relative"
                    title={`${m.name}: ${m.answer || "—"}`}
                    style={{
                      opacity: matches ? 1 : 0.2,
                      filter: matches ? "none" : "grayscale(1)",
                      transition: "opacity 0.2s, filter 0.2s",
                    }}
                  >
                    <img
                      src={getModelLogo(m.modelId)}
                      alt={m.name}
                      className="h-7 w-7 rounded-full border border-slate-200 object-contain bg-white"
                    />
                    {matches && (
                      <span className="absolute -right-0.5 -top-0.5 flex h-3.5 w-3.5 items-center justify-center rounded-full bg-amber-500 text-[8px] font-bold text-white shadow">
                        ✓
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* ── Footer: regenerate ── */}
      <div className="flex items-center justify-end border-t border-amber-100 pt-3">
        <button
          type="button"
          onClick={onRegenerate}
          disabled={isGenerating}
          className="rounded-full border border-amber-200 px-3.5 py-1 text-xs font-medium text-amber-700 transition hover:border-amber-400 hover:bg-amber-50 disabled:opacity-50"
        >
          {isGenerating ? "Regenerating…" : "Regenerate"}
        </button>
      </div>
    </div>
  );
}
