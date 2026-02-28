import { useState } from "react";
import type { Detection } from "../mockData";
import { MathText } from "./MathText";

const DOCUMENT_TYPE_LABELS: Record<string, { short: string; description: string }> = {
  csat: {
    short: "CSAT",
    description: "College Scholastic Ability Test — Mathematics (Korean national exam)",
  },
  sat: {
    short: "SAT",
    description: "SAT Math Module (U.S. college admission test)",
  },
};

type DetectorViewProps = {
  imageUrl: string;
  detections: Detection[];
  documentType: "csat" | "sat" | null;
  selectedId: string | null;
  onSelect: (id: string) => void;
  imageSize: { width: number; height: number };
  checkedIds?: Set<string>;
  onToggleCheck?: (id: string) => void;
  solveModality?: "text" | "image" | "image+text";
  onChangeSolveModality?: (modality: "text" | "image" | "image+text") => void;
};

const cropStyle = (
  imageUrl: string,
  detection: Detection,
  imageSize: { width: number; height: number },
) => {
  if (imageSize.width <= 1 || imageSize.height <= 1) return {};
  const isRelative =
    detection.x <= 1 && detection.y <= 1 && detection.w <= 1 && detection.h <= 1;
  const x = isRelative ? detection.x : detection.x / imageSize.width;
  const y = isRelative ? detection.y : detection.y / imageSize.height;
  const w = isRelative ? detection.w : detection.w / imageSize.width;
  const h = isRelative ? detection.h : detection.h / imageSize.height;

  return {
    backgroundImage: `url(${imageUrl})`,
    backgroundSize: `${100 / w}% ${100 / h}%`,
    backgroundPosition: `${(-x / w) * 100}% ${(-y / h) * 100}%`,
  } as const;
};

export default function DetectorView({
  imageUrl,
  detections,
  documentType,
  selectedId,
  onSelect,
  checkedIds = new Set(),
  onToggleCheck,
  imageSize,
  solveModality = "image",
  onChangeSolveModality,
}: DetectorViewProps) {
  const [labelHover, setLabelHover] = useState(false);
  const isReady = imageSize.width > 1 && imageSize.height > 1;
  const docMeta = documentType ? DOCUMENT_TYPE_LABELS[documentType] : null;
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h2 className="text-xl font-semibold text-slate-900">Page-to-Problem Segmentation</h2>
        <p className="text-sm text-slate-600">
          Detected problem regions are shown on the image. Select a region to
          compare multi-model solutions.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
        <div className="rounded-2xl border border-amber-100 bg-white p-4 shadow-sm">
          <div className="relative w-full rounded-xl border border-amber-100 bg-slate-50">
            {docMeta ? (
              <div
                className="absolute left-3 top-3 z-20"
                onMouseEnter={() => setLabelHover(true)}
                onMouseLeave={() => setLabelHover(false)}
              >
                {labelHover ? (
                  <div className="absolute bottom-full left-0 mb-1.5 whitespace-nowrap rounded-lg border border-slate-200 bg-white px-3 py-2 text-left text-xs font-normal text-slate-700 shadow-lg">
                    {docMeta.description}
                  </div>
                ) : null}
                <span className="rounded-md border border-amber-600/80 bg-amber-500/95 px-2.5 py-1 text-xs font-bold uppercase tracking-wide text-white shadow-sm">
                  {docMeta.short}
                </span>
              </div>
            ) : null}
            <div className="relative overflow-hidden rounded-xl">
            <img
              src={imageUrl}
              alt="Detected workbook"
              className="h-auto w-full object-contain"
            />
            <div className="absolute inset-0" aria-hidden="true">
              {isReady &&
                detections.map((detection) => {
                const isSelected = detection.id === selectedId;
                const isRelative =
                  detection.x <= 1 &&
                  detection.y <= 1 &&
                  detection.w <= 1 &&
                  detection.h <= 1;
                const left = isRelative
                  ? detection.x * 100
                  : (detection.x / imageSize.width) * 100;
                const top = isRelative
                  ? detection.y * 100
                  : (detection.y / imageSize.height) * 100;
                const width = isRelative
                  ? detection.w * 100
                  : (detection.w / imageSize.width) * 100;
                const height = isRelative
                  ? detection.h * 100
                  : (detection.h / imageSize.height) * 100;
                return (
                  <button
                    key={detection.id}
                    type="button"
                    onClick={() => onSelect(detection.id)}
                    className={`absolute border-2 transition ${
                      isSelected
                        ? "border-amber-500 bg-amber-200/30"
                        : "border-slate-400/70 bg-slate-200/10 hover:border-amber-400"
                    }`}
                    style={{
                      left: `${left}%`,
                      top: `${top}%`,
                      width: `${width}%`,
                      height: `${height}%`,
                    }}
                  >
                    <span className="absolute left-0 top-0 -translate-y-full rounded-none bg-slate-900 px-1.5 py-0.5 text-[10px] font-semibold text-white">
                      {detection.id}
                    </span>
                    <span className="sr-only">Select {detection.label}</span>
                  </button>
                );
              })}
            </div>
            </div>
          </div>
          <div className="mt-3 text-xs text-slate-500">
            {detections.length
              ? `${detections.length} problem regions detected`
              : "No detections yet. Run Detect Problems to populate this view."}
          </div>
        </div>

        <div className="rounded-2xl border border-amber-100 bg-white p-4 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
              Cropped Problems
            </div>
            {onChangeSolveModality ? (
              <div className="flex flex-wrap gap-1.5">
                {(["image", "text", "image+text"] as const).map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => onChangeSolveModality(m)}
                    className={`rounded-full border px-3 py-1 text-xs font-semibold transition ${
                      solveModality === m
                        ? "border-amber-500 bg-amber-100 text-amber-900"
                        : "border-slate-200 bg-white text-slate-600 hover:border-amber-300"
                    }`}
                  >
                    {m === "image+text" ? "Image+Text" : m === "image" ? "Image" : "Text"}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
          <p className="mt-2 text-xs text-slate-500">
            Check problems to solve (multiple allowed). Use modality to choose input for solve.
          </p>
          <div className="mt-3 grid gap-3">
            {detections.map((detection) => {
              const isSelected = detection.id === selectedId;
              const isChecked = onToggleCheck ? checkedIds.has(detection.id) : false;
              return (
                <div
                  key={detection.id}
                  role="button"
                  tabIndex={0}
                  onClick={() => { onToggleCheck?.(detection.id); onSelect(detection.id); }}
                  onKeyDown={(e) => e.key === "Enter" && (onToggleCheck?.(detection.id), onSelect(detection.id))}
                  className={`flex flex-col gap-3 rounded-xl border p-3 text-left transition cursor-pointer ${
                    isSelected
                      ? "border-amber-400 bg-amber-50"
                      : "border-slate-200 bg-white hover:border-amber-300"
                  }`}
                >
                  <div className="flex items-start gap-2">
                    {onToggleCheck ? (
                      <span
                        aria-hidden="true"
                        className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded border-2 transition pointer-events-none ${
                          isChecked
                            ? "border-amber-500 bg-amber-500 text-white"
                            : "border-slate-300 bg-white"
                        }`}
                      >
                        {isChecked ? "✓" : null}
                      </span>
                    ) : null}
                    <div className="min-w-0 flex-1">
                      {(solveModality === "image" || solveModality === "image+text") && (
                        detection.cropUrl ? (
                          <div className="flex h-40 w-full items-center justify-center rounded border border-amber-100 bg-white p-2">
                            <img
                              src={detection.cropUrl}
                              alt={`Crop ${detection.id}`}
                              className="h-full w-full object-contain"
                            />
                          </div>
                        ) : (
                          <div
                            className="h-40 w-full rounded border border-amber-100 bg-slate-100"
                            style={cropStyle(imageUrl, detection, imageSize)}
                          />
                        )
                      )}
                      {(solveModality === "text" || solveModality === "image+text") && (
                        <div className="mt-2 flex items-start gap-2 text-sm text-slate-600">
                          <span className="rounded bg-slate-900 px-1.5 py-0.5 text-[10px] font-semibold text-white">
                            {detection.id}
                          </span>
                          <div className="min-w-0 flex-1 leading-relaxed">
                            <MathText text={detection.text || detection.label || ""} />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
