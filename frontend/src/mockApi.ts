import {
  getMockDetections,
  getMockResults,
  type DetectResult,
  type ModelResult,
} from "./mockData";

const randomDelay = () => 700 + Math.floor(Math.random() * 500);

/** Backend base URL for detect/classify. When set, detectProblems and classifyDocument call the real API.
 * In dev, defaults to http://localhost:8000 so miniserver (BACKEND_PORT=8000) works without .env. */
const DETECT_API_BASE = (() => {
  const env = import.meta.env as Record<string, string | undefined>;
  const url = env.VITE_DETECT_API_URL?.trim();
  if (url) return url.replace(/\/$/, "");
  if (env.DEV) return "http://localhost:8000";
  return "";
})();

/** Classify document image as CSAT or SAT (model first, OCR fallback when confidence < 0.8). */
export async function classifyDocument(
  imageDataUrl: string,
): Promise<{ label: "csat" | "sat"; confidence: number }> {
  if (DETECT_API_BASE) {
    const res = await fetch(`${DETECT_API_BASE}/api/document/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_base64: imageDataUrl.startsWith("data:") ? imageDataUrl : `data:image/png;base64,${imageDataUrl}`,
      }),
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `Classify failed: ${res.status}`);
    }
    const data = await res.json();
    return { label: data.label ?? "csat", confidence: Number(data.confidence) ?? 0 };
  }
  return new Promise((resolve) => {
    setTimeout(() => resolve({ label: "csat", confidence: 0.9 }), randomDelay());
  });
}

/** Fetch image from URL and return as data URL (for dataset images). */
export function imageUrlToDataUrl(url: string): Promise<string> {
  return fetch(url)
    .then((r) => {
      if (!r.ok) throw new Error(`Failed to load image: ${r.status}`);
      return r.blob();
    })
    .then(
      (blob) =>
        new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(String(reader.result));
          reader.onerror = () => reject(new Error("Failed to read image"));
          reader.readAsDataURL(blob);
        }),
    );
}

/** Crop a region from the full page image (used when pre-generated crop is not available, e.g. SAT sample). */
export function cropImageFromPage(
  pageImageUrlOrDataUrl: string,
  x: number,
  y: number,
  w: number,
  h: number,
): Promise<string> {
  const dataUrlPromise = pageImageUrlOrDataUrl.startsWith("data:")
    ? Promise.resolve(pageImageUrlOrDataUrl)
    : imageUrlToDataUrl(pageImageUrlOrDataUrl);

  return dataUrlPromise.then((dataUrl) => {
    return new Promise<string>((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const tw = img.naturalWidth;
        const th = img.naturalHeight;
        const x1 = Math.max(0, Math.min(x, tw - 1));
        const y1 = Math.max(0, Math.min(y, th - 1));
        const w1 = Math.max(1, Math.min(w, tw - x1));
        const h1 = Math.max(1, Math.min(h, th - y1));
        const canvas = document.createElement("canvas");
        canvas.width = w1;
        canvas.height = h1;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          reject(new Error("Canvas 2d not available"));
          return;
        }
        ctx.drawImage(img, x1, y1, w1, h1, 0, 0, w1, h1);
        resolve(canvas.toDataURL("image/png"));
      };
      img.onerror = () => reject(new Error("Failed to load page image for crop"));
      img.src = dataUrl;
    });
  });
}

/** Build page_id for backend from dataset year/page (CSAT: year_math_odd_page_N, SAT: sat_math_odd_page_N). */
function buildPageId(opts: {
  datasetYear?: string;
  datasetPage?: string;
  documentType?: "csat" | "sat" | null;
}): string | null {
  const { datasetYear, datasetPage, documentType } = opts;
  if (!datasetPage) return null;
  if (documentType === "sat") return `sat_math_odd_page_${datasetPage}`;
  if (datasetYear) return `${datasetYear}_math_odd_page_${datasetPage}`;
  return null;
}

/** dataset 이미지 URL에서 year/page 추론 (예: /data/2026_math_odd/page_001.png → 2026, 001). */
function inferDatasetMetaFromUrl(url: string): { year: string; page: string } | null {
  const m = url.match(/\/data\/(\d{4})_math_odd\/page_(\d+)\.png$/);
  if (m) return { year: m[1], page: m[2].padStart(3, "0") };
  const mSat = url.match(/\/data\/sat_math_odd\/page_([^/]+)\.png$/);
  if (mSat) return { year: "sat", page: mSat[1] };
  return null;
}

/** Cache key for detect result: same image + same opts => reuse result to avoid redundant work. */
function detectCacheKey(payloadImage: string, opts: { documentType?: string; pageId?: string } | undefined): string {
  const optPart = opts ? `${opts.documentType ?? ""}_${opts.pageId ?? ""}` : "";
  if (payloadImage.length > 12000) {
    return payloadImage.slice(0, 600) + "_" + payloadImage.length + "_" + optPart;
  }
  return payloadImage + optPart;
}

const detectResultCache = new Map<string, DetectResult>();

/** Detect 결과 캐시만 제거. */
export function clearDetectCache(): void {
  detectResultCache.clear();
}

/** 2026 샘플용 캐시만 제거 (잘못된 값이 고정된 경우 대비). */
export function clearDetectCacheFor2026(): void {
  const prefix = "2026_math_odd_page_";
  for (const key of detectResultCache.keys()) {
    if (key.includes(prefix)) detectResultCache.delete(key);
  }
}

/** 프론트 Detect 캐시 + 백엔드 converter 캐시 전부 제거. */
export async function clearAllCaches(): Promise<void> {
  detectResultCache.clear();
  const base = (import.meta.env as Record<string, string>).VITE_DETECT_API_URL?.trim?.()?.replace(/\/$/, "")
    || (import.meta.env.DEV ? "http://localhost:8000" : "");
  if (base) {
    try {
      await fetch(`${base}/api/cache/clear`, { method: "POST" });
    } catch {
      // ignore
    }
  }
}

export async function detectProblems(
  imageDataUrl: string,
  opts?: {
    pageId?: string;
    documentType?: "csat" | "sat" | null;
    datasetYear?: string;
    datasetPage?: string;
  },
): Promise<DetectResult> {
  if (DETECT_API_BASE) {
    try {
      let payloadImage = imageDataUrl;
      if (!payloadImage.startsWith("data:")) {
        // Dataset pages are typically URL paths; convert to data URL before backend call.
        if (payloadImage.startsWith("/") || /^https?:\/\//.test(payloadImage)) {
          payloadImage = await imageUrlToDataUrl(payloadImage);
        } else {
          payloadImage = `data:image/png;base64,${payloadImage}`;
        }
      }
      let pageId = opts?.pageId ?? buildPageId(opts ?? {});
      if (!pageId && imageDataUrl && typeof imageDataUrl === "string" && !imageDataUrl.startsWith("data:") && imageDataUrl.length < 3000) {
        const inferred = inferDatasetMetaFromUrl(imageDataUrl);
        if (inferred)
          pageId =
            inferred.year === "sat"
              ? `sat_math_odd_page_${inferred.page}`
              : `${inferred.year}_math_odd_page_${inferred.page}`;
      }
      const isDatasetSample = !!(opts?.datasetYear && opts?.datasetPage) || !!pageId?.startsWith("2026_math_odd_page_");
      const cacheKey = detectCacheKey(payloadImage, { documentType: opts?.documentType ?? undefined, pageId: pageId ?? undefined });
      const cached = !isDatasetSample ? detectResultCache.get(cacheKey) : undefined;
      if (cached) return cached;

      const res = await fetch(`${DETECT_API_BASE}/api/problems/detect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_base64: payloadImage,
          document_type: opts?.documentType ?? null,
          page_id: pageId,
          save_to_demo_parsing: false,
          run_ocr_per_crop: true,
        }),
      });
      if (!res.ok) {
        const err = await res.text();
        throw new Error(err || `Detect failed: ${res.status}`);
      }
      const data = await res.json();
      const result: DetectResult = {
        documentType: data.documentType ?? null,
        detections: Array.isArray(data.detections) ? data.detections : [],
        imageWidth: data.imageWidth ?? undefined,
        imageHeight: data.imageHeight ?? undefined,
      };
      if (!isDatasetSample) detectResultCache.set(cacheKey, result);
      return result;
    } catch (e) {
      console.error("Detect API error:", e);
      throw e;
    }
  }
  return new Promise((resolve) => {
    setTimeout(
      () =>
        resolve({
          documentType: "csat",
          detections: getMockDetections(),
        }),
      randomDelay(),
    );
  });
}

export async function solveWithModels(problemId: string): Promise<ModelResult[]> {
  // TODO: Replace with real calls to multiple LLMs and aggregation logic.
  return new Promise((resolve) => {
    setTimeout(() => resolve(getMockResults(problemId)), randomDelay());
  });
}
