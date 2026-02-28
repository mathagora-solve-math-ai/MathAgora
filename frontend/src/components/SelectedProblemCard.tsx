import type { Detection } from "../mockData";
import { MathText } from "./MathText";

type SolveModality = "text" | "image" | "image+text";

const MODALITY_LABELS: Record<SolveModality, string> = {
  text: "Input: Text",
  image: "Input: Image",
  "image+text": "Input: Image + Text",
};

type SelectedProblemCardProps = {
  imageUrl: string;
  detection: Detection | null;
  problemText: string;
  modality: SolveModality;
  imageSize: { width: number; height: number };
  cropImageDataUrl: string | null;
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

export default function SelectedProblemCard({
  imageUrl,
  detection,
  problemText,
  modality,
  imageSize,
  cropImageDataUrl,
}: SelectedProblemCardProps) {
  return (
    <div className="rounded-2xl border border-amber-100 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
          Selected Problem
        </div>
        <span className="rounded-md border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-amber-700">
          {MODALITY_LABELS[modality]}
        </span>
      </div>
      <div className="mt-3 flex flex-col gap-3">
        {(modality === "image" || modality === "image+text") && (
          <div className="h-48 w-full overflow-hidden rounded-xl border border-amber-100 bg-slate-50">
            {cropImageDataUrl ? (
              <img
                src={cropImageDataUrl}
                alt="Problem crop"
                className="h-full w-full object-contain"
              />
            ) : detection ? (
              <div
                className="h-full w-full"
                style={cropStyle(imageUrl, detection, imageSize)}
              />
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-slate-400">
                No crop available
              </div>
            )}
          </div>
        )}
        {(modality === "text" || modality === "image+text") && (
          problemText ? (
            <div className="rounded-lg border border-slate-200 bg-slate-50/50 px-3 py-2 text-sm text-slate-700">
              <MathText text={problemText} />
            </div>
          ) : (
            <p className="text-sm italic text-slate-400">No problem text</p>
          )
        )}
      </div>
    </div>
  );
}
