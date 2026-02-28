import { useMemo, useState } from "react";
import type { AlignedModelResult, AlignedStep } from "../alignment.ts";
import { MathText } from "./MathText.tsx";

type SolveViewProps = {
  results: AlignedModelResult[];
  showAlignmentDebug?: boolean;
};

const palette = [
  {
    bg: "bg-amber-100",
    border: "border-amber-300",
    text: "text-amber-900",
  },
  {
    bg: "bg-emerald-100",
    border: "border-emerald-300",
    text: "text-emerald-900",
  },
  {
    bg: "bg-sky-100",
    border: "border-sky-300",
    text: "text-sky-900",
  },
  {
    bg: "bg-rose-100",
    border: "border-rose-300",
    text: "text-rose-900",
  },
  {
    bg: "bg-lime-100",
    border: "border-lime-300",
    text: "text-lime-900",
  },
];

const StepAccordion = ({
  step,
  index,
  colorClass,
  showDebug,
}: {
  step: AlignedStep;
  index: number;
  colorClass: { bg: string; border: string; text: string } | null;
  showDebug?: boolean;
}) => {
  const [open, setOpen] = useState(index === 0);
  return (
    <div
      className={`rounded-lg border px-3 py-2 text-sm ${
        colorClass
          ? `${colorClass.bg} ${colorClass.border} ${colorClass.text}`
          : "border-slate-200 bg-white text-slate-700"
      }`}
    >
      <button
        type="button"
        className="flex w-full items-center justify-between gap-2 text-left"
        onClick={() => setOpen((prev) => !prev)}
      >
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-slate-500">
            {index + 1}
          </span>
          <p className="font-medium">{step.title}</p>
        </div>
        <div className="flex items-center gap-2">
          {step.alignmentGroupId ? (
            <span className="rounded-full bg-white/60 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide">
              {step.alignmentGroupId}
              {showDebug && step.similarityHint !== undefined
                ? ` ${step.similarityHint.toFixed(2)}`
                : ""}
            </span>
          ) : null}
          <span className="text-slate-400">{open ? "▾" : "▸"}</span>
        </div>
      </button>
      {open ? (
        <p className="mt-2 text-sm text-slate-700">
          <MathText text={step.body} />
        </p>
      ) : null}
    </div>
  );
};

export default function SolveView({
  results,
  showAlignmentDebug,
}: SolveViewProps) {
  const alignmentMap = useMemo(() => {
    const groupIds = Array.from(
      new Set(
        results
          .flatMap((result) => result.steps)
          .map((step) => step.alignmentGroupId)
          .filter((id): id is string => Boolean(id)),
      ),
    );

    return new Map(
      groupIds.map((id, index) => [id, palette[index % palette.length]]),
    );
  }, [results]);

  return (
    <div className="grid gap-4 lg:grid-cols-5">
      {results.map((result) => (
        <div
          key={result.modelId}
          className="flex h-full flex-col rounded-2xl border border-amber-100 bg-white p-4 shadow-soft"
        >
          <div className="flex items-start justify-between">
            <div>
              <div className="text-lg font-semibold text-slate-900">
                {result.modelName}
              </div>
              <div className="text-xs text-slate-500">{result.version}</div>
            </div>
            <div className="text-right text-xs text-slate-500">
              <div>{result.latencyMs}ms</div>
              <div>Temp {result.temperature}</div>
            </div>
          </div>

          <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-3">
            <div className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
              Solution Strategy
            </div>
            <div className="mt-2 text-sm text-slate-700">
              <MathText text={result.strategy} />
            </div>
          </div>

          <div className="mt-4 flex flex-1 flex-col gap-2">
            {result.steps.map((step, index) => (
              <StepAccordion
                key={step.id}
                step={step}
                index={index}
                showDebug={showAlignmentDebug}
                colorClass={
                  step.alignmentGroupId
                    ? alignmentMap.get(step.alignmentGroupId) || null
                    : null
                }
              />
            ))}
          </div>

          <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-3">
            <div className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
              Answer
            </div>
            <div className="mt-1 text-sm font-semibold text-slate-900">
              <MathText text={result.finalAnswer} />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
