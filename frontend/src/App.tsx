import { useEffect, useMemo, useRef, useState } from "react";
import ImageUploader from "./components/ImageUploader";
import DetectorView from "./components/DetectorView";
import ModelOutputView from "./components/ModelOutputView";
import SelectedProblemCard from "./components/SelectedProblemCard";
import FlowMapView, { type FlowMapData } from "./components/FlowMapView";
import AggregationView, { type AggregationData } from "./components/AggregationView";
import {
  MOCK_IMAGE_DATA_URL,
  getFallbackDetections,
  type Detection,
  type ModelResult,
  type UploadedImage,
} from "./mockData";
import { clearDetectCache, clearDetectCacheFor2026, cropImageFromPage, detectProblems, imageUrlToDataUrl } from "./mockApi";
import { loadDatasetIndex, loadDatasetPage } from "./dataset";
import type { DatasetIndex } from "./dataset";
import { createLLMStreamClient } from "./llmStreamClient.index";
import type { ModelId, SolveModality, StreamChunk, SolveRequest } from "./llmStreamClient";
import { parseStructuredSolution } from "./postprocess";
import { getSolveCache, getSolveCacheKey, setSolveCache } from "./solveCache";

const MODEL_COLORS = [
  "#22d3ee",
  "#3b82f6",
  "#22c55e",
  "#f59e0b",
  "#f43f5e",
  "#a855f7",
];

const STORAGE_KEY = "qanda_upload_v1";

type Stage = "upload" | "detect" | "solve";

type StreamState = {
  status: "idle" | "streaming" | "done" | "error" | "stopped";
  partialText: string;
  startedAtMs?: number;
  finishedAtMs?: number;
  errorMessage?: string;
};

type ModelMeta = {
  modelId: ModelId;
  displayName: string;
  version: string;
  temperature: number;
  latencyMs: number;
};

/** Per-problem solve state (one entry per checked problem). */
type ProblemSolveState = {
  cropDataUrl: string;
  inputModality: SolveModality;
  modelMetas: ModelMeta[];
  streamStates: Record<ModelId, StreamState>;
  structuredResults: ModelResult[] | null;
  flowmapData: FlowMapData | null;
  isGeneratingFlowmap: boolean;
  aggregationData: AggregationData | null;
  isGeneratingAggregation: boolean;
};

const SOLVE_MODELS: ModelMeta[] = [
  {
    modelId: "openai/gpt-5-codex",
    displayName: "GPT-5-Codex",
    version: "openai/gpt-5-codex",
    temperature: 0.0,
    latencyMs: 0,
  },
  {
    modelId: "openai/gpt-5",
    displayName: "GPT-5",
    version: "openai/gpt-5",
    temperature: 0.0,
    latencyMs: 0,
  },
  {
    modelId: "anthropic/claude-opus-4.5",
    displayName: "Claude Opus 4.5",
    version: "anthropic/claude-opus-4.5",
    temperature: 0.0,
    latencyMs: 0,
  },
  {
    modelId: "google/gemini-3-pro-preview",
    displayName: "Gemini 3 Pro",
    version: "google/gemini-3-pro-preview",
    temperature: 0.0,
    latencyMs: 0,
  },
  {
    modelId: "x-ai/grok-4-fast",
    displayName: "Grok 4 Fast",
    version: "x-ai/grok-4-fast",
    temperature: 0.0,
    latencyMs: 0,
  },
];

const initStreamStates = (models: ModelMeta[]): Record<ModelId, StreamState> =>
  models.reduce<Record<ModelId, StreamState>>((acc, model) => {
    acc[model.modelId] = { status: "idle", partialText: "" };
    return acc;
  }, {});

/** Derive overall status of a problem's solve session for tab indicator. */
function getProblemStatus(ps: ProblemSolveState): "idle" | "streaming" | "done" | "error" {
  if (!ps.modelMetas.length) return "idle";
  const statuses = ps.modelMetas.map((m) => ps.streamStates[m.modelId]?.status ?? "idle");
  if (statuses.some((s) => s === "streaming")) return "streaming";
  if (statuses.every((s) => ["done", "error", "stopped"].includes(s))) {
    return statuses.some((s) => s === "error") ? "error" : "done";
  }
  return "idle";
}

const truncateTabLabel = (label: string, max = 22) =>
  label.length <= max ? label : `${label.slice(0, max).trimEnd()}…`;

const DETECT_STAGES = [
  { label: "Identifying document type", detail: "Checking whether this is a CSAT or SAT exam" },
  { label: "Finding problem regions", detail: "Locating each question on the page" },
  { label: "Reading text from problems", detail: "Extracting text from each cropped region" },
  { label: "Polishing extracted text", detail: "Fixing math notation and formatting" },
] as const;

export default function App() {
  const [stage, setStage] = useState<Stage>("upload");
  const [uploadedImage, setUploadedImage] = useState<UploadedImage | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [documentType, setDocumentType] = useState<"csat" | "sat" | null>(null);
  const [imageSize, setImageSize] = useState({ width: 1, height: 1 });
  const [error, setError] = useState<string | null>(null);

  // Dataset controls
  const [datasetYear, setDatasetYear] = useState("2025");
  const [datasetPage, setDatasetPage] = useState("015");
  const [datasetIndex, setDatasetIndex] = useState<DatasetIndex | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectStageIdx, setDetectStageIdx] = useState(-1);

  // Detect stage: highlighted problem (image bbox) + checked problems (for batch solve)
  const [selectedProblemId, setSelectedProblemId] = useState<string | null>(null);
  const [checkedProblemIds, setCheckedProblemIds] = useState<Set<string>>(new Set());
  const [solveModality, setSolveModality] = useState<SolveModality>("image");

  // Solve stage: per-problem state map + active tab
  const [solveStates, setSolveStates] = useState<Record<string, ProblemSolveState>>({});
  const [activeTabId, setActiveTabId] = useState<string | null>(null);
  const [isSolving, setIsSolving] = useState(false);

  const streamClientRef = useRef(createLLMStreamClient());
  const abortControllersRef = useRef<Record<string, AbortController>>({});
  const stoppedModelsRef = useRef<Record<string, Set<ModelId>>>({});
  const autoAlignedRef = useRef<Record<string, boolean>>({});
  const autoFlowmapRequestedRef = useRef<Record<string, boolean>>({});
  const autoAggregationRequestedRef = useRef<Record<string, boolean>>({});

  // ─── Auto-scroll refs ─────────────────────────────────────────────────────
  const detectProgressRef = useRef<HTMLDivElement>(null);
  const flowmapSectionRef = useRef<HTMLDivElement>(null);
  const aggregationSectionRef = useRef<HTMLDivElement>(null);
  const scrolledForFlowmapRef = useRef(new Set<string>());
  const scrolledForAggRef = useRef(new Set<string>());

  // ─── Dataset loading ─────────────────────────────────────────────────────

  const loadDatasetSelection = async (
    year: string,
    page: string,
    index?: DatasetIndex | null,
  ) => {
    clearDetectCache();
    if (year === "2026") clearDetectCacheFor2026();
    const dataset = await loadDatasetPage({ year, page, index: index ?? undefined });
    setUploadedImage({
      dataUrl: dataset.imageUrl,
      name: `dataset_${dataset.meta.year}_${dataset.meta.set}_page_${dataset.meta.page}.png`,
      updatedAt: Date.now(),
      source: "dataset",
      datasetMeta: dataset.meta,
    });
    setDetections(dataset.detections);
    setDocumentType(dataset.meta.year === "sat" ? "sat" : "csat");
    if (dataset.imageWidth != null && dataset.imageHeight != null) {
      setImageSize({ width: dataset.imageWidth, height: dataset.imageHeight });
    } else {
      setImageSize({ width: 1, height: 1 });
    }
    setSelectedProblemId(dataset.detections[0]?.id ?? null);
    setCheckedProblemIds(new Set());
    setSolveStates({});
    setActiveTabId(null);
    setStage("upload");
    setError(null);
  };

  // 앱 로드 시 캐시 제거 → 새로 동작 (2026 잘못된 고정값 방지를 위해 2026 캐시도 초기화)
  useEffect(() => {
    clearDetectCache();
    clearDetectCacheFor2026();
  }, []);

  // Detect 진행 중 단계별 progress 애니메이션
  useEffect(() => {
    if (!isDetecting) {
      setDetectStageIdx(-1);
      return;
    }
    setDetectStageIdx(0);
    const timers = [
      setTimeout(() => setDetectStageIdx(1), 5000),
      setTimeout(() => setDetectStageIdx(2), 18000),
      setTimeout(() => setDetectStageIdx(3), 35000),
    ];
    return () => timers.forEach(clearTimeout);
  }, [isDetecting]);

  useEffect(() => {
    let hasStoredImage = false;
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const parsed = JSON.parse(stored) as UploadedImage;
        if (parsed?.dataUrl) {
          setUploadedImage(parsed);
          hasStoredImage = true;
          if (parsed.source === "dataset" && parsed.datasetMeta) {
            setDatasetYear(parsed.datasetMeta.year);
            setDatasetPage(parsed.datasetMeta.page);
          }
        }
      } catch {
        // ignore
      }
    }

    loadDatasetIndex()
      .then((index) => {
        setDatasetIndex(index);
        const years = Object.keys(index.years).sort().reverse();
        const fallbackYear = index.years[datasetYear] ? datasetYear : years[0];
        const pages = Object.keys(index.years[fallbackYear]?.pages ?? {}).sort();
        const fallbackPage = index.years[fallbackYear]?.pages?.[datasetPage]
          ? datasetPage
          : pages[0];
        setDatasetYear((prev) => (index.years[prev] ? prev : fallbackYear));
        setDatasetPage((prev) =>
          index.years[fallbackYear]?.pages?.[prev] ? prev : fallbackPage,
        );
        if (!hasStoredImage) {
          return loadDatasetSelection(fallbackYear, fallbackPage, index);
        }
      })
      .catch(() => {
        if (!hasStoredImage) setUploadedImage(null);
      });
  }, []);

  useEffect(() => {
    if (!uploadedImage) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(uploadedImage));
    } catch {
      // QuotaExceededError when image is too large for localStorage — ignore gracefully
    }
  }, [uploadedImage]);

  useEffect(() => {
    if (!uploadedImage?.dataUrl) return;
    const img = new Image();
    img.onload = () =>
      setImageSize({ width: img.naturalWidth, height: img.naturalHeight });
    img.src = uploadedImage.dataUrl;
  }, [uploadedImage]);

  // Purge stale checked IDs when detections change
  useEffect(() => {
    const validIds = new Set(detections.map((d) => d.id));
    setCheckedProblemIds((prev) => {
      const next = new Set([...prev].filter((id) => validIds.has(id)));
      return next.size === prev.size ? prev : next;
    });
  }, [detections]);

  // ─── Auto-align steps when all models finish ──────────────────────────────

  useEffect(() => {
    const needsBuild: string[] = [];
    Object.keys(solveStates).forEach((pid) => {
      if (autoAlignedRef.current[pid]) return;
      const ps = solveStates[pid];
      if (!ps?.modelMetas?.length || ps.structuredResults) return;
      const allDone = ps.modelMetas.every((m) =>
        ["done", "error", "stopped"].includes(ps.streamStates[m.modelId]?.status ?? "idle"),
      );
      if (allDone) needsBuild.push(pid);
    });
    if (!needsBuild.length) return;

    needsBuild.forEach((pid) => {
      autoAlignedRef.current[pid] = true;
    });

    setSolveStates((prev) => {
      const next = { ...prev };
      needsBuild.forEach((pid) => {
        const ps = prev[pid];
        if (!ps) return;
        const results: ModelResult[] = ps.modelMetas.map((model) => {
          const state = ps.streamStates[model.modelId] ?? { status: "idle", partialText: "" };
          const parsed = parseStructuredSolution(state.partialText);
          return {
            modelId: model.modelId,
            modelName: model.displayName,
            version: model.version,
            latencyMs: model.latencyMs,
            temperature: model.temperature,
            strategy: parsed.strategy || "(generating strategy)",
            steps: parsed.steps.map((step, index) => ({
              id: `${model.modelId}_${index}`,
              title: step.title || `Step ${index + 1}`,
              body: step.body,
            })),
            finalAnswer: parsed.finalAnswer || "",
          };
        });
        next[pid] = { ...ps, structuredResults: results };
      });
      return next;
    });
  }, [solveStates]);

  // ─── Auto-generate flow map when all models are done (no button click) ─────
  useEffect(() => {
    const base = (
      (import.meta.env as Record<string, string | undefined>).VITE_SOLVE_API_URL ??
      (import.meta.env as Record<string, string | undefined>).VITE_DETECT_API_URL ??
      ""
    ).replace(/\/$/, "");

    Object.keys(solveStates).forEach((problemId) => {
      const ps = solveStates[problemId];
      if (!ps?.structuredResults || ps.flowmapData != null || ps.isGeneratingFlowmap) return;
      if (autoFlowmapRequestedRef.current[problemId]) return;
      autoFlowmapRequestedRef.current[problemId] = true;

      const detection = detections.find((d) => d.id === problemId);
      const problemText = detection?.text || detection?.label || "";

      setSolveStates((prev) => ({
        ...prev,
        [problemId]: { ...prev[problemId], isGeneratingFlowmap: true },
      }));
      setError(null);

      fetch(`${base}/api/flowmap/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          problem_text: problemText,
          solutions: ps.structuredResults.map((r) => ({
            model_name: r.modelName,
            steps: r.steps.map((step, idx) => ({
              step_idx: idx,
              title: step.title,
              content: step.body,
            })),
          })),
        }),
      })
        .then(async (res) => {
          if (!res.ok) throw new Error((await res.text()) || `Flow map failed: ${res.status}`);
          return res.json() as Promise<FlowMapData>;
        })
        .then((data) => {
          setSolveStates((prev) => {
            const ps = prev[problemId];
            if (ps?.structuredResults) {
              const cacheKey = getSolveCacheKey(problemId, ps.inputModality, uploadedImage ?? null);
              setSolveCache(cacheKey, {
                modality: ps.inputModality,
                structuredResults: ps.structuredResults,
                flowmapData: data,
              });
            }
            return {
              ...prev,
              [problemId]: { ...prev[problemId], flowmapData: data, isGeneratingFlowmap: false },
            };
          });
        })
        .catch((e) => {
          // ref stays true → no infinite retry on failure
          setSolveStates((prev) => ({
            ...prev,
            [problemId]: { ...prev[problemId], isGeneratingFlowmap: false },
          }));
          setError(e instanceof Error ? e.message : "Flow map generation failed.");
        });
    });
  }, [solveStates, detections]);

  // ─── Auto-generate aggregation when flow map is ready ────────────────────
  useEffect(() => {
    const base = (
      (import.meta.env as Record<string, string | undefined>).VITE_SOLVE_API_URL ??
      (import.meta.env as Record<string, string | undefined>).VITE_DETECT_API_URL ??
      ""
    ).replace(/\/$/, "");

    Object.keys(solveStates).forEach((problemId) => {
      const ps = solveStates[problemId];
      if (!ps?.flowmapData || ps.aggregationData != null || ps.isGeneratingAggregation) return;
      if (autoAggregationRequestedRef.current[problemId]) return;
      autoAggregationRequestedRef.current[problemId] = true;

      const detection = detections.find((d) => d.id === problemId);
      const problemText = detection?.text || detection?.label || "";

      setSolveStates((prev) => ({
        ...prev,
        [problemId]: { ...prev[problemId], isGeneratingAggregation: true },
      }));

      fetch(`${base}/api/aggregation/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          problem_text: problemText,
          solutions: (ps.structuredResults ?? []).map((r) => ({
            model_name: r.modelName,
            steps: r.steps.map((step, idx) => ({
              step_idx: idx,
              title: step.title,
              content: step.body,
            })),
            final_answer: r.finalAnswer,
          })),
        }),
      })
        .then(async (res) => {
          if (!res.ok) throw new Error((await res.text()) || `Aggregation failed: ${res.status}`);
          return res.json() as Promise<AggregationData>;
        })
        .then((data) => {
          setSolveStates((prev) => ({
            ...prev,
            [problemId]: { ...prev[problemId], aggregationData: data, isGeneratingAggregation: false },
          }));
        })
        .catch((e) => {
          // ref stays true → no infinite retry on failure; user can manually regenerate
          setSolveStates((prev) => ({
            ...prev,
            [problemId]: { ...prev[problemId], isGeneratingAggregation: false },
          }));
          setError(e instanceof Error ? e.message : "Aggregation failed.");
        });
    });
  }, [solveStates, detections]);

  // ─── Auto-scroll ─────────────────────────────────────────────────────────

  // Scroll to detect progress box when detection starts
  useEffect(() => {
    if (isDetecting) {
      requestAnimationFrame(() =>
        detectProgressRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }),
      );
    }
  }, [isDetecting]);

  // Scroll to flowmap the first time its loading spinner appears for the active tab
  useEffect(() => {
    if (!activeTabId || !solveStates[activeTabId]?.isGeneratingFlowmap) return;
    if (scrolledForFlowmapRef.current.has(activeTabId)) return;
    scrolledForFlowmapRef.current.add(activeTabId);
    requestAnimationFrame(() =>
      flowmapSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }),
    );
  }, [activeTabId, solveStates]);

  // Scroll to aggregation the first time its result is fully ready for the active tab
  useEffect(() => {
    if (!activeTabId) return;
    const ps = solveStates[activeTabId];
    if (!ps?.aggregationData || ps.isGeneratingAggregation) return;
    if (scrolledForAggRef.current.has(activeTabId)) return;
    scrolledForAggRef.current.add(activeTabId);
    requestAnimationFrame(() =>
      aggregationSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }),
    );
  }, [activeTabId, solveStates]);

  // ─── Handlers ────────────────────────────────────────────────────────────

  const handleFileSelected = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      setUploadedImage({
        dataUrl: String(reader.result || ""),
        name: file.name,
        updatedAt: Date.now(),
        isDemo: false,
        source: "upload",
        sizeBytes: file.size,
      });
      setDetections([]);
      setDocumentType(null);
      setSelectedProblemId(null);
      setCheckedProblemIds(new Set());
      setSolveStates({});
      setActiveTabId(null);
      setStage("upload");
      setError(null);
    };
    reader.readAsDataURL(file);
  };

  const handleDatasetYearChange = (value: string) => {
    const pages = Object.keys(datasetIndex?.years?.[value]?.pages ?? {}).sort();
    const nextPage = pages[0] ?? datasetPage;
    setDatasetYear(value);
    setDatasetPage(nextPage);
    if (datasetIndex) {
      loadDatasetSelection(value, nextPage, datasetIndex).catch(() =>
        setError("Dataset page could not be loaded."),
      );
    }
  };

  const handleDatasetPageChange = (value: string) => {
    setDatasetPage(value);
    if (datasetIndex) {
      loadDatasetSelection(datasetYear, value, datasetIndex).catch(() =>
        setError("Dataset page could not be loaded."),
      );
    }
  };

  const handleDetect = async () => {
    if (!uploadedImage?.dataUrl) {
      setError("Upload or capture a workbook photo before detecting problems.");
      return;
    }
    setError(null);
    setIsDetecting(true);
    setSolveStates({});
    setActiveTabId(null);
    try {
      // Always use backend detect so UI text is the converted (post-processor) result.
      const result = await detectProblems(uploadedImage.dataUrl, {
        datasetYear: uploadedImage.source === "dataset" && uploadedImage.datasetMeta
          ? uploadedImage.datasetMeta.year
          : undefined,
        datasetPage: uploadedImage.source === "dataset" && uploadedImage.datasetMeta
          ? uploadedImage.datasetMeta.page
          : undefined,
        documentType: uploadedImage.source === "dataset" && uploadedImage.datasetMeta
          ? (uploadedImage.datasetMeta.year === "sat" ? "sat" : "csat")
          : null,
      });
      const list = result.detections?.length ? result.detections : getFallbackDetections();
      setDetections(list);
      setDocumentType(result.documentType);
      setSelectedProblemId(list[0]?.id ?? null);
      setCheckedProblemIds(new Set());
      setStage("detect");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Detection failed.");
      // Still go to detect stage with fallback regions so the image shows with some selectable areas.
      const fallback = getFallbackDetections();
      setDetections(fallback);
      setSelectedProblemId(fallback[0]?.id ?? null);
      setCheckedProblemIds(new Set());
      setStage("detect");
    } finally {
      setIsDetecting(false);
    }
  };

  /** Resolve crop data URL for a detection. */
  const resolveCropDataUrl = async (
    detection: Detection,
    pageImageUrl: string,
  ): Promise<string> => {
    if (detection.cropUrl) {
      return detection.cropUrl.startsWith("data:")
        ? detection.cropUrl
        : await imageUrlToDataUrl(detection.cropUrl);
    }
    return cropImageFromPage(
      pageImageUrl,
      detection.x,
      detection.y,
      detection.w,
      detection.h,
    );
  };

  const handleStreamChunkForProblem = (problemId: string, chunk: StreamChunk) => {
    setSolveStates((prev) => {
      const ps = prev[problemId];
      if (!ps) return prev;
      if (stoppedModelsRef.current[problemId]?.has(chunk.modelId)) return prev;

      const current = ps.streamStates[chunk.modelId] ?? { status: "idle", partialText: "" };
      let nextStreamState: StreamState;

      if (chunk.kind === "event") {
        if (chunk.event === "start") {
          nextStreamState = {
            ...current,
            status: "streaming",
            startedAtMs: chunk.timestampMs,
            errorMessage: undefined,
          };
        } else if (chunk.event === "done") {
          nextStreamState = { ...current, status: "done", finishedAtMs: chunk.timestampMs };
        } else if (chunk.event === "error") {
          nextStreamState = {
            ...current,
            status: "error",
            errorMessage: chunk.errorMessage ?? "Stream error",
            finishedAtMs: chunk.timestampMs,
          };
        } else {
          return prev;
        }
      } else if (chunk.kind === "text" || chunk.kind === "token") {
        nextStreamState = {
          ...current,
          partialText: `${current.partialText}${chunk.text ?? ""}`,
        };
      } else {
        return prev;
      }

      return {
        ...prev,
        [problemId]: {
          ...ps,
          streamStates: { ...ps.streamStates, [chunk.modelId]: nextStreamState },
        },
      };
    });
  };

  const handleSolveSelected = async () => {
    if (!uploadedImage?.dataUrl) {
      setError("Upload or capture a workbook photo before solving.");
      return;
    }

    const idsToSolve =
      checkedProblemIds.size > 0
        ? [...checkedProblemIds]
        : selectedProblemId
          ? [selectedProblemId]
          : [];

    if (!idsToSolve.length) {
      setError("Select or check at least one problem to solve.");
      return;
    }

    // Restore from localStorage cache when available (same problem + modality + source)
    const cachedByPid = new Map<string, { modality: SolveModality; structuredResults: ModelResult[]; flowmapData: FlowMapData }>();
    idsToSolve.forEach((pid) => {
      const key = getSolveCacheKey(pid, solveModality, uploadedImage);
      const cached = getSolveCache(key);
      if (cached) cachedByPid.set(pid, cached);
    });

    /** Build streamStates so UI shows "done" with parsed steps from cached structuredResults. */
    const streamStatesFromStructuredResults = (structuredResults: ModelResult[]): Record<ModelId, StreamState> => {
      const out: Record<string, StreamState> = {};
      SOLVE_MODELS.forEach((model) => {
        const r = structuredResults.find(
          (x) => x.modelId === model.modelId || x.modelName === model.displayName,
        );
        const partialText = r
          ? JSON.stringify({
              model_name: r.modelName,
              steps: r.steps.map((s, i) => ({ step_idx: i, title: s.title, content: s.body })),
              final_answer: r.finalAnswer,
            })
          : "";
        out[model.modelId] = { status: "done", partialText };
      });
      return out as Record<ModelId, StreamState>;
    };

    // Only run solve for problems not in memory and not in cache
    const pidsToRun = idsToSolve.filter((pid) => {
      if (cachedByPid.has(pid)) return false;
      return (
        !solveStates[pid]?.structuredResults ||
        solveStates[pid]?.inputModality !== solveModality
      );
    });
    pidsToRun.forEach((pid) => {
      abortControllersRef.current[pid]?.abort();
      delete abortControllersRef.current[pid];
      delete stoppedModelsRef.current[pid];
      delete autoAlignedRef.current[pid];
    });

    setSolveStates((prev) => {
      const next: Record<string, ProblemSolveState> = {};
      idsToSolve.forEach((pid) => {
        const cached = cachedByPid.get(pid);
        if (cached) {
          next[pid] = {
            cropDataUrl: "",
            inputModality: cached.modality,
            modelMetas: SOLVE_MODELS,
            streamStates: streamStatesFromStructuredResults(cached.structuredResults),
            structuredResults: cached.structuredResults,
            flowmapData: cached.flowmapData,
            isGeneratingFlowmap: false,
            aggregationData: null,
            isGeneratingAggregation: false,
          };
          return;
        }
        if (prev[pid]?.structuredResults != null) {
          if (prev[pid]?.inputModality === solveModality) {
            next[pid] = prev[pid];
          } else {
            next[pid] = {
              cropDataUrl: "",
              inputModality: solveModality,
              modelMetas: SOLVE_MODELS,
              streamStates: initStreamStates(SOLVE_MODELS),
              structuredResults: null,
              flowmapData: null,
              isGeneratingFlowmap: false,
              aggregationData: null,
              isGeneratingAggregation: false,
            };
            stoppedModelsRef.current[pid] = new Set();
          }
        } else {
          next[pid] = {
            cropDataUrl: prev[pid]?.cropDataUrl ?? "",
            inputModality: solveModality,
            modelMetas: SOLVE_MODELS,
            streamStates: prev[pid]?.streamStates ?? initStreamStates(SOLVE_MODELS),
            structuredResults: null,
            flowmapData: null,
            isGeneratingFlowmap: false,
            aggregationData: null,
            isGeneratingAggregation: false,
          };
          stoppedModelsRef.current[pid] = new Set();
        }
      });
      return next;
    });
    setActiveTabId(idsToSolve[0]);
    setStage("solve");
    setError(null);
    setIsSolving(true);

    // Resolve cropDataUrl for problems restored from cache (so Selected Problem shows the crop)
    cachedByPid.forEach((_, pid) => {
      const detection = detections.find((d) => d.id === pid);
      if (!detection || solveModality === "text") return;
      resolveCropDataUrl(detection, uploadedImage.dataUrl)
        .then((cropDataUrl) => {
          setSolveStates((prev) => {
            const ps = prev[pid];
            if (!ps) return prev;
            return { ...prev, [pid]: { ...ps, cropDataUrl } };
          });
        })
        .catch(() => {});
    });

    if (pidsToRun.length === 0) {
      setIsSolving(false);
      return;
    }

    const streamPromises = pidsToRun.map(async (pid) => {
      const detection = detections.find((d) => d.id === pid);
      if (!detection) return;

      const needsImageInput = solveModality !== "text";
      let cropDataUrl = "";
      if (needsImageInput) {
        try {
          cropDataUrl = await resolveCropDataUrl(detection, uploadedImage.dataUrl);
        } catch {
          setSolveStates((prev) => {
            const ps = prev[pid];
            if (!ps) return prev;
            const failedStates = Object.fromEntries(
              SOLVE_MODELS.map((m) => [
                m.modelId,
                { status: "error" as const, partialText: "", errorMessage: "Failed to load crop image." },
              ]),
            );
            return { ...prev, [pid]: { ...ps, streamStates: failedStates } };
          });
          return;
        }
      }

      setSolveStates((prev) => ({
        ...prev,
        [pid]: { ...prev[pid], cropDataUrl, inputModality: solveModality },
      }));

      const controller = new AbortController();
      abortControllersRef.current[pid] = controller;

      const req: SolveRequest = {
        problemId: pid,
        problemLabel: detection.label || "Problem",
        cropImageDataUrl: cropDataUrl || undefined,
        problemText: (detection.text || detection.label || "").trim(),
        modality: solveModality,
        models: SOLVE_MODELS.map((m) => ({
          modelId: m.modelId,
          displayName: m.displayName,
        })),
      };

      try {
        await streamClientRef.current.startSolveStream(req, {
          signal: controller.signal,
          onChunk: (chunk) => handleStreamChunkForProblem(pid, chunk),
        });
      } catch (e) {
        const message = e instanceof Error ? e.message : "Solve streaming failed.";
        const finishedAtMs = Date.now();
        setSolveStates((prev) => {
          const ps = prev[pid];
          if (!ps) return prev;
          const nextStreamStates = { ...ps.streamStates };
          SOLVE_MODELS.forEach((m) => {
            const s = nextStreamStates[m.modelId];
            if (s && !["done", "error", "stopped"].includes(s.status)) {
              nextStreamStates[m.modelId] = { ...s, status: "error", errorMessage: message, finishedAtMs };
            }
          });
          return { ...prev, [pid]: { ...ps, streamStates: nextStreamStates } };
        });
      }
    });

    await Promise.allSettled(streamPromises);
    setIsSolving(false);
  };

  const handleStopModel = (problemId: string, modelId: ModelId) => {
    stoppedModelsRef.current[problemId]?.add(modelId);
    setSolveStates((prev) => {
      const ps = prev[problemId];
      if (!ps) return prev;
      const current = ps.streamStates[modelId] ?? { status: "idle", partialText: "" };
      return {
        ...prev,
        [problemId]: {
          ...ps,
          streamStates: {
            ...ps.streamStates,
            [modelId]: { ...current, status: "stopped", errorMessage: "Stopped by user" },
          },
        },
      };
    });
  };

  const handleStopProblem = (problemId: string) => {
    abortControllersRef.current[problemId]?.abort();
    setSolveStates((prev) => {
      const ps = prev[problemId];
      if (!ps) return prev;
      const nextStreamStates = { ...ps.streamStates };
      ps.modelMetas.forEach((m) => {
        if (nextStreamStates[m.modelId]?.status === "streaming") {
          nextStreamStates[m.modelId] = {
            ...nextStreamStates[m.modelId],
            status: "stopped",
            errorMessage: "Stopped by user",
          };
        }
      });
      return { ...prev, [problemId]: { ...ps, streamStates: nextStreamStates } };
    });
  };

  const handleStopAll = () => {
    Object.keys(abortControllersRef.current).forEach((pid) =>
      abortControllersRef.current[pid]?.abort(),
    );
    setSolveStates((prev) => {
      const next = { ...prev };
      Object.keys(next).forEach((pid) => {
        const ps = next[pid];
        const nextStreamStates = { ...ps.streamStates };
        ps.modelMetas.forEach((m) => {
          if (nextStreamStates[m.modelId]?.status === "streaming") {
            nextStreamStates[m.modelId] = {
              ...nextStreamStates[m.modelId],
              status: "stopped",
              errorMessage: "Stopped by user",
            };
          }
        });
        next[pid] = { ...ps, streamStates: nextStreamStates };
      });
      return next;
    });
  };

  const handleGenerateFlowmap = async (problemId: string) => {
    const ps = solveStates[problemId];
    if (!ps?.structuredResults) return;

    const detection = detections.find((d) => d.id === problemId);
    const problemText = detection?.text || detection?.label || "";

    const base = (
      (import.meta.env as Record<string, string | undefined>).VITE_SOLVE_API_URL ??
      (import.meta.env as Record<string, string | undefined>).VITE_DETECT_API_URL ??
      ""
    ).replace(/\/$/, "");

    setSolveStates((prev) => ({
      ...prev,
      [problemId]: { ...prev[problemId], isGeneratingFlowmap: true },
    }));
    setError(null);

    try {
      const res = await fetch(`${base}/api/flowmap/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          problem_text: problemText,
          solutions: ps.structuredResults.map((r) => ({
            model_name: r.modelName,
            steps: r.steps.map((step, idx) => ({
              step_idx: idx,
              title: step.title,
              content: step.body,
            })),
          })),
        }),
      });
      if (!res.ok) {
        throw new Error((await res.text()) || `Flow map generation failed: ${res.status}`);
      }
      const data = (await res.json()) as FlowMapData;
      setSolveStates((prev) => {
        const nextPs = { ...prev[problemId], flowmapData: data, isGeneratingFlowmap: false };
        if (nextPs.structuredResults) {
          const key = getSolveCacheKey(problemId, nextPs.inputModality, uploadedImage ?? null);
          setSolveCache(key, {
            modality: nextPs.inputModality,
            structuredResults: nextPs.structuredResults,
            flowmapData: data,
          });
        }
        return { ...prev, [problemId]: nextPs };
      });
    } catch (e) {
      setSolveStates((prev) => ({
        ...prev,
        [problemId]: { ...prev[problemId], isGeneratingFlowmap: false },
      }));
      setError(e instanceof Error ? e.message : "Flow map generation failed.");
    }
  };

  const handleGenerateAggregation = async (problemId: string) => {
    const ps = solveStates[problemId];
    if (!ps?.structuredResults) return;

    const detection = detections.find((d) => d.id === problemId);
    const problemText = detection?.text || detection?.label || "";

    const base = (
      (import.meta.env as Record<string, string | undefined>).VITE_SOLVE_API_URL ??
      (import.meta.env as Record<string, string | undefined>).VITE_DETECT_API_URL ??
      ""
    ).replace(/\/$/, "");

    setSolveStates((prev) => ({
      ...prev,
      [problemId]: { ...prev[problemId], isGeneratingAggregation: true },
    }));
    setError(null);

    try {
      const res = await fetch(`${base}/api/aggregation/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          problem_text: problemText,
          solutions: ps.structuredResults.map((r) => ({
            model_name: r.modelName,
            steps: r.steps.map((step, idx) => ({
              step_idx: idx,
              title: step.title,
              content: step.body,
            })),
            final_answer: r.finalAnswer,
          })),
        }),
      });
      if (!res.ok) throw new Error((await res.text()) || `Aggregation failed: ${res.status}`);
      const data = (await res.json()) as AggregationData;
      setSolveStates((prev) => ({
        ...prev,
        [problemId]: { ...prev[problemId], aggregationData: data, isGeneratingAggregation: false },
      }));
    } catch (e) {
      setSolveStates((prev) => ({
        ...prev,
        [problemId]: { ...prev[problemId], isGeneratingAggregation: false },
      }));
      setError(e instanceof Error ? e.message : "Aggregation failed.");
    }
  };

  const handleToggleCheck = (id: string) => {
    setCheckedProblemIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  // Derived values for the active tab
  const activePs = activeTabId ? (solveStates[activeTabId] ?? null) : null;
  const activeDetection = useMemo(
    () => (activeTabId ? detections.find((d) => d.id === activeTabId) ?? null : null),
    [detections, activeTabId],
  );
  const activeProblemText = activeDetection?.text || activeDetection?.label || "";


  // ─── Render ───────────────────────────────────────────────────────────────

  const solveButtonLabel =
    checkedProblemIds.size > 1
      ? `Solve Selected (${checkedProblemIds.size})`
      : "Solve";

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_#fff5da_0%,_#f7f4ef_35%,_#efe7dc_100%)]">
      <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-10 px-6 py-10">
        <header className="flex items-center justify-between">
          <button
            type="button"
            onClick={() => setStage("upload")}
            className="flex items-center gap-3 rounded-xl transition hover:opacity-75 active:scale-[0.97] focus-visible:outline-none"
          >
            <img
              src="/logos/icon.png"
              alt="MathAgora"
              className="h-10 w-10 rounded-2xl object-contain"
            />
            <div className="text-lg font-bold tracking-tight text-slate-900">
              MathAgora
            </div>
          </button>
        </header>

        {error ? (
          <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {error}
          </div>
        ) : null}

        {/* ── Upload ── */}
        {stage === "upload" ? (
          <section className="rounded-3xl border border-amber-100 bg-white/80 p-6 shadow-soft backdrop-blur">
            <ImageUploader
              image={uploadedImage}
              onFileSelected={handleFileSelected}
              datasetYear={datasetYear}
              datasetPage={datasetPage}
              onChangeDatasetYear={handleDatasetYearChange}
              onChangeDatasetPage={handleDatasetPageChange}
              datasetIndex={datasetIndex}
            />
            <div className="mt-6 flex flex-wrap items-center gap-3">
              <button
                className="rounded-full bg-slate-900 px-5 py-2 text-sm font-semibold text-white transition hover:bg-slate-800 active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-900/40 disabled:opacity-60"
                onClick={handleDetect}
                disabled={isDetecting}
              >
                {isDetecting ? "Detecting…" : "Detect Problems"}
              </button>
            </div>

            {/* ── Detect progress indicator ── */}
            {isDetecting ? (
              <div ref={detectProgressRef} className="mt-5 flex flex-col gap-4 rounded-2xl border border-amber-100 bg-white p-6 shadow-sm">
                {/* Header */}
                <div className="flex items-center gap-3">
                  <span className="h-5 w-5 shrink-0 animate-spin rounded-full border-[2.5px] border-amber-400 border-t-transparent" />
                  <span className="text-sm font-semibold text-slate-800">Analysing workbook…</span>
                  <span className="ml-auto text-xs text-slate-400">
                    {detectStageIdx + 1} / {DETECT_STAGES.length}
                  </span>
                </div>

                {/* Progress bar */}
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
                  <div
                    className="h-full rounded-full bg-amber-400 transition-all duration-700 ease-out"
                    style={{ width: `${((detectStageIdx + 1) / DETECT_STAGES.length) * 100}%` }}
                  />
                </div>

                {/* Stage list */}
                <ol className="flex flex-col gap-2.5">
                  {DETECT_STAGES.map((s, idx) => {
                    const isDone = idx < detectStageIdx;
                    const isActive = idx === detectStageIdx;
                    return (
                      <li key={idx} className="flex items-center gap-3">
                        {isDone ? (
                          <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-[10px] font-bold text-emerald-600">
                            ✓
                          </span>
                        ) : isActive ? (
                          <span className="h-5 w-5 shrink-0 animate-spin rounded-full border-[2px] border-amber-400 border-t-transparent" />
                        ) : (
                          <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-slate-100">
                            <span className="h-1.5 w-1.5 rounded-full bg-slate-300" />
                          </span>
                        )}
                        <div className="min-w-0 flex-1">
                          <span className={`text-sm ${isDone ? "text-slate-400 line-through" : isActive ? "font-medium text-slate-800" : "text-slate-400"}`}>
                            {s.label}
                          </span>
                        </div>
                      </li>
                    );
                  })}
                </ol>
              </div>
            ) : null}
          </section>
        ) : null}

        {/* ── Detect ── */}
        {stage === "detect" ? (
          <section className="rounded-3xl border border-amber-100 bg-white/80 p-6 shadow-soft backdrop-blur">
            <DetectorView
              imageUrl={uploadedImage?.dataUrl || MOCK_IMAGE_DATA_URL}
              detections={detections}
              documentType={documentType}
              selectedId={selectedProblemId}
              checkedIds={checkedProblemIds}
              onSelect={setSelectedProblemId}
              onToggleCheck={handleToggleCheck}
              imageSize={imageSize}
              solveModality={solveModality}
              onChangeSolveModality={setSolveModality}
            />
            <div className="mt-6 flex flex-wrap items-center gap-3">
              <button
                className="rounded-full bg-slate-900 px-5 py-2 text-sm font-semibold text-white transition hover:bg-slate-800 active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-900/40 disabled:opacity-60"
                onClick={handleSolveSelected}
                disabled={isSolving}
              >
                {isSolving ? "Solving..." : solveButtonLabel}
              </button>
              <button
                className="rounded-full border border-slate-300 px-5 py-2 text-sm font-semibold text-slate-700 transition hover:border-slate-500 hover:bg-slate-50 active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300/60"
                onClick={() => setStage("upload")}
              >
                Back to Upload
              </button>
            </div>
          </section>
        ) : null}

        {/* ── Solve ── */}
        {stage === "solve" && Object.keys(solveStates).length > 0 ? (
          <section className="rounded-3xl border border-amber-100 bg-white/80 p-6 shadow-soft backdrop-blur">
            <div className="flex flex-col gap-6">
              {/* Header */}
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h2 className="text-xl font-semibold text-slate-900">Solve & Compare</h2>
                  <p className="text-sm text-slate-600">
                    Step-by-step solutions from each model. When all models finish, the flow map is generated automatically.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    className="rounded-full border border-slate-300 px-4 py-1.5 text-sm font-semibold text-slate-700 transition hover:border-slate-500 hover:bg-slate-50"
                    onClick={handleStopAll}
                  >
                    Stop All
                  </button>
                  <button
                    className="rounded-full border border-slate-300 px-4 py-1.5 text-sm font-semibold text-slate-700 transition hover:border-slate-500 hover:bg-slate-50"
                    onClick={() => setStage("detect")}
                  >
                    Back to Detect
                  </button>
                </div>
              </div>

              {/* Tab bar */}
              <div className="flex gap-2 overflow-x-auto pb-1">
                {Object.keys(solveStates).map((pid) => {
                  const ps = solveStates[pid];
                  const status = getProblemStatus(ps);
                  const det = detections.find((d) => d.id === pid);
                  const label = truncateTabLabel(det?.label || pid);
                  const isActive = pid === activeTabId;
                  return (
                    <button
                      key={pid}
                      type="button"
                      onClick={() => setActiveTabId(pid)}
                      className={`flex shrink-0 items-center gap-1.5 rounded-full border px-4 py-1.5 text-sm font-medium transition ${
                        isActive
                          ? "border-amber-400 bg-amber-50 text-slate-900 shadow-sm"
                          : "border-slate-200 bg-white text-slate-500 hover:border-amber-300 hover:bg-amber-50/50 hover:text-slate-700"
                      }`}
                    >
                      <span className="max-w-[160px] truncate">{label}</span>
                      {status === "streaming" && (
                        <span className="h-2.5 w-2.5 shrink-0 animate-spin rounded-full border-2 border-slate-400 border-t-transparent" />
                      )}
                      {status === "done" && (
                        <span className="shrink-0 text-emerald-500">✓</span>
                      )}
                      {status === "error" && (
                        <span className="shrink-0 text-rose-500">✕</span>
                      )}
                    </button>
                  );
                })}
              </div>

              {/* Active tab content */}
              {activePs && activeTabId ? (
                <>
                  <SelectedProblemCard
                    imageUrl={uploadedImage?.dataUrl || MOCK_IMAGE_DATA_URL}
                    detection={activeDetection}
                    problemText={activeProblemText}
                    modality={activePs.inputModality}
                    imageSize={imageSize}
                    cropImageDataUrl={activePs.cropDataUrl || null}
                  />

                  <ModelOutputView
                    models={activePs.modelMetas}
                    streamStates={activePs.streamStates}
                    onStopModel={(modelId) => handleStopModel(activeTabId, modelId as ModelId)}
                    onStopAll={() => handleStopProblem(activeTabId)}
                  />

                  {activePs.flowmapData ? (
                    <div className="mt-2 flex flex-wrap items-center gap-3">
                      <button
                        className={`rounded-full border border-slate-300 px-5 py-2 text-sm font-semibold text-slate-700 transition hover:border-slate-500 hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300/60 ${activePs.isGeneratingFlowmap ? "cursor-not-allowed opacity-60" : ""}`}
                        onClick={() => handleGenerateFlowmap(activeTabId)}
                        disabled={activePs.isGeneratingFlowmap}
                      >
                        {activePs.isGeneratingFlowmap ? "Regenerating..." : "Regenerate flow map"}
                      </button>
                    </div>
                  ) : activePs.isGeneratingFlowmap ? (
                    <div className="mt-4 flex items-center gap-4 rounded-2xl border border-slate-200 bg-slate-50 px-6 py-5">
                      <span className="h-7 w-7 shrink-0 animate-spin rounded-full border-[3px] border-slate-300 border-t-slate-600" />
                      <div>
                        <p className="text-base font-semibold text-slate-800">Generating flow map…</p>
                        <p className="mt-0.5 text-sm text-slate-500">
                          Grouping solution steps across all models — this takes about 10–20 seconds.
                        </p>
                      </div>
                    </div>
                  ) : null}

                  {activePs.flowmapData ? (
                    <div ref={flowmapSectionRef} className="mt-4 flex flex-col gap-3">
                      <h3 className="text-lg font-semibold text-slate-900">Flow Map</h3>
                      <p className="text-sm text-slate-600">
                        Solution steps grouped by stage across models, highlighting shared strategies and divergence points.
                      </p>
                      <FlowMapView
                        data={activePs.flowmapData}
                        modelInfos={(activePs.structuredResults ?? []).map((r, idx) => ({
                          name: r.modelName,
                          color: MODEL_COLORS[idx % MODEL_COLORS.length],
                          modelId: activePs.modelMetas[idx]?.modelId,
                        }))}
                        finalAnswers={Object.fromEntries(
                          (activePs.structuredResults ?? []).map((r) => [r.modelName, r.finalAnswer]),
                        )}
                      />
                    </div>
                  ) : null}

                  {(activePs.aggregationData || activePs.isGeneratingAggregation) ? (
                    <div ref={aggregationSectionRef} className="mt-4 flex justify-center">
                      <div className="w-full max-w-2xl">
                        <AggregationView
                          data={activePs.aggregationData}
                          isGenerating={activePs.isGeneratingAggregation}
                          onRegenerate={() => handleGenerateAggregation(activeTabId)}
                          modelAnswers={(activePs.structuredResults ?? []).map((r) => {
                            const meta = activePs.modelMetas.find((m) => m.displayName === r.modelName);
                            return {
                              name: r.modelName,
                              modelId: meta?.modelId ?? "",
                              answer: r.finalAnswer?.trim() ?? "",
                            };
                          })}
                        />
                      </div>
                    </div>
                  ) : null}
                </>
              ) : null}
            </div>
          </section>
        ) : null}
      </div>
    </div>
  );
}
