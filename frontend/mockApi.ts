import {
    getMockDetections,
    getMockResults,
    type DetectResult,
    type ModelResult,
  } from "./mockData";
  
  const randomDelay = () => 700 + Math.floor(Math.random() * 500);
  
  /** Backend base URL for detect/classify. When set, detectProblems and classifyDocument call the real API. */
  const DETECT_API_BASE = (
    import.meta.env as Record<string, string | undefined>
  ).VITE_DETECT_API_URL ?? "";
  
  /** Classify document image as CSAT or SAT (model first, OCR fallback when confidence < 0.8). */
  export async function classifyDocument(
    imageDataUrl: string,
  ): Promise<{ label: "csat" | "sat"; confidence: number }> {
    if (DETECT_API_BASE) {
      const res = await fetch(`${DETECT_API_BASE.replace(/\/$/, "")}/api/document/classify`, {
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
  
  export async function detectProblems(
    imageDataUrl: string,
    opts?: {
      pageId?: string;
      documentType?: "csat" | "sat" | null;
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
        const res = await fetch(`${DETECT_API_BASE.replace(/\/$/, "")}/api/problems/detect`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_base64: payloadImage,
            document_type: opts?.documentType ?? null,
            page_id: opts?.pageId ?? null,
            // Detect/Solve UI path doesn't require demo_parsing artifacts.
            // Disable it to avoid extra per-crop OCR and speed up detect.
            save_to_demo_parsing: false,
            run_ocr_per_crop: false,
          }),
        });
        if (!res.ok) {
          const err = await res.text();
          throw new Error(err || `Detect failed: ${res.status}`);
        }
        const data = await res.json();
        return {
          documentType: data.documentType ?? null,
          detections: Array.isArray(data.detections) ? data.detections : [],
        };
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
  