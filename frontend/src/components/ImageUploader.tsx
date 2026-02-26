import { useRef, useState, type ChangeEvent, type DragEvent } from "react";
import type { UploadedImage } from "../mockData";
import type { DatasetIndex } from "../dataset";

type ImageUploaderProps = {
  image: UploadedImage | null;
  onFileSelected: (file: File) => void;
  datasetYear: string;
  datasetPage: string;
  onChangeDatasetYear: (value: string) => void;
  onChangeDatasetPage: (value: string) => void;
  datasetIndex?: DatasetIndex | null;
};

export default function ImageUploader({
  image,
  onFileSelected,
  datasetYear,
  datasetPage,
  onChangeDatasetYear,
  onChangeDatasetPage,
  datasetIndex,
}: ImageUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    onFileSelected(file);
    event.currentTarget.value = "";
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) onFileSelected(file);
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
      <div className="flex flex-col gap-4">
        <h2 className="text-lg font-semibold text-slate-900">Capture or Upload</h2>

        {/* ── Upload / Camera block (same card style as Sample) ── */}
        <div className="flex flex-col gap-3 rounded-2xl border border-amber-100 bg-white px-4 py-4">
          {/* Drop zone — filled state when user has uploaded a file */}
          {image?.source === "upload" ? (
            <div className="flex items-center gap-3 rounded-xl border-2 border-amber-300 bg-amber-50/60 px-4 py-3">
              {/* Thumbnail */}
              <div className="h-16 w-16 shrink-0 overflow-hidden rounded-lg border border-amber-200 bg-white">
                <img
                  src={image.dataUrl}
                  alt="preview"
                  className="h-full w-full object-contain"
                />
              </div>
              {/* Info */}
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-semibold text-slate-800">{image.name}</p>
                {image.sizeBytes != null && (
                  <p className="mt-0.5 text-xs text-slate-500">
                    {image.sizeBytes >= 1024 * 1024
                      ? `${(image.sizeBytes / 1024 / 1024).toFixed(1)} MB`
                      : `${Math.round(image.sizeBytes / 1024)} KB`}
                  </p>
                )}
              </div>
              {/* Change button */}
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="shrink-0 rounded-full border border-amber-300 bg-white px-3 py-1.5 text-xs font-semibold text-amber-700 transition hover:bg-amber-100"
              >
                Change
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleInputChange}
              />
            </div>
          ) : (
            <div
              onClick={() => fileInputRef.current?.click()}
              onDragEnter={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={(e) => {
                if (!e.currentTarget.contains(e.relatedTarget as Node)) setIsDragging(false);
              }}
              onDrop={handleDrop}
              className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-8 text-center transition select-none
                ${isDragging
                  ? "border-amber-400 bg-amber-50"
                  : "border-amber-200 bg-amber-50/40 hover:border-amber-300 hover:bg-amber-50/70"
                }`}
            >
              <div className={`flex h-14 w-14 items-center justify-center rounded-2xl transition
                ${isDragging ? "bg-amber-200 text-amber-700" : "bg-amber-100 text-amber-500"}`}>
                <svg viewBox="0 0 24 24" className="h-7 w-7" fill="none" stroke="currentColor" strokeWidth="1.6">
                  <path d="M12 16V4" />
                  <path d="M8 8l4-4 4 4" />
                  <path d="M4 20h16" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-700">
                  {isDragging ? "Drop image here" : "Drag & drop or click to upload"}
                </p>
                <p className="mt-0.5 text-xs text-slate-500">PNG · JPG · WEBP</p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleInputChange}
              />
            </div>
          )}

          {/* Divider */}
          <div className="flex items-center gap-3">
            <div className="h-px flex-1 bg-slate-100" />
            <span className="text-xs text-slate-400">or</span>
            <div className="h-px flex-1 bg-slate-100" />
          </div>

          {/* Camera button */}
          <label className="flex cursor-pointer items-center justify-center gap-2.5 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-semibold text-slate-700 transition hover:border-amber-300 hover:bg-amber-50 active:scale-[0.98]">
            <svg viewBox="0 0 24 24" className="h-5 w-5 shrink-0 text-slate-500" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M4 7h3l2-3h6l2 3h3v12H4z" />
              <circle cx="12" cy="13" r="3.2" />
            </svg>
            Take a photo with camera
            <input
              ref={cameraInputRef}
              type="file"
              accept="image/*"
              capture="environment"
              className="hidden"
              onChange={handleInputChange}
            />
          </label>
        </div>

        {/* ── Sample block ── */}
        <div className="flex flex-col gap-3 rounded-2xl border border-amber-100 bg-white px-4 py-3">
          <div className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
            Sample
          </div>
          {(() => {
            const isSample = image?.source !== "upload";
            return (
              <>
                <div className="flex flex-wrap gap-2">
                  {Object.keys(datasetIndex?.years ?? { [datasetYear]: {} })
                    .sort()
                    .reverse()
                    .map((year) => (
                      <button
                        key={year}
                        type="button"
                        onClick={() => onChangeDatasetYear(year)}
                        className={`rounded-full border px-4 py-1 text-sm font-semibold transition ${
                          isSample && year === datasetYear
                            ? "border-slate-900 bg-slate-900 text-white"
                            : "border-slate-200 bg-white text-slate-600 hover:border-slate-400"
                        }`}
                      >
                        {year}
                      </button>
                    ))}
                </div>
                <div className="flex flex-wrap gap-2">
                  {Object.keys(datasetIndex?.years?.[datasetYear]?.pages ?? {}).length ? (
                    Object.keys(datasetIndex?.years?.[datasetYear]?.pages ?? {}).map((page) => (
                      <button
                        key={page}
                        type="button"
                        onClick={() => onChangeDatasetPage(page)}
                        className={`rounded-full border px-3 py-1 text-xs font-semibold transition ${
                          isSample && page === datasetPage
                            ? "border-amber-400 bg-amber-100 text-amber-900"
                            : "border-slate-200 bg-white text-slate-600 hover:border-amber-300"
                        }`}
                      >
                        page_{page}
                      </button>
                    ))
                  ) : (
                    <button
                      type="button"
                      className={`rounded-full border px-3 py-1 text-xs font-semibold ${
                        isSample
                          ? "border-amber-400 bg-amber-100 text-amber-900"
                          : "border-slate-200 bg-white text-slate-600"
                      }`}
                    >
                      page_{datasetPage}
                    </button>
                  )}
                </div>
              </>
            );
          })()}
        </div>

        {image?.source === "dataset" ? (
          <div className="rounded-2xl border border-dashed border-amber-200 bg-amber-50/40 px-4 py-3 text-xs text-amber-900">
            {`Sample image loaded from /data (${image?.datasetMeta?.year ?? datasetYear} page_${image?.datasetMeta?.page ?? datasetPage}). Use Detect to show parsed regions.`}
          </div>
        ) : null}
      </div>

      {/* ── Preview ── */}
      <div className="rounded-2xl border border-amber-100 bg-white p-4 shadow-sm">
        <div className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
          Preview
        </div>
        <div className="mt-3 overflow-hidden rounded-xl border border-amber-100 bg-slate-50">
          {image?.dataUrl ? (
            <img
              src={image.dataUrl}
              alt="Workbook preview"
              className="h-[32rem] w-full object-contain"
            />
          ) : (
            <div className="flex h-[32rem] items-center justify-center text-sm text-slate-400">
              No image yet
            </div>
          )}
        </div>
        <div className="mt-3 text-xs text-slate-500">
          {image?.name ? `File: ${image.name}` : "Awaiting upload or capture."}
        </div>
      </div>
    </div>
  );
}
