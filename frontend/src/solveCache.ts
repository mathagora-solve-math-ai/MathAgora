/**
 * Persist solve results + flow map per problem so they survive navigation/refresh.
 */
import type { FlowMapData } from "./components/FlowMapView";
import type { ModelResult } from "./mockData";
import type { SolveModality } from "./llmStreamClient";
import type { UploadedImage } from "./mockData";

const PREFIX = "qanda_solve_cache_";

export type CachedSolve = {
  modality: SolveModality;
  structuredResults: ModelResult[];
  flowmapData: FlowMapData;
};

export function getSolveCacheKey(
  problemId: string,
  modality: SolveModality,
  uploadedImage: UploadedImage | null,
): string {
  const base = `${problemId}_${modality}`;
  if (uploadedImage?.datasetMeta) {
    const { year, page } = uploadedImage.datasetMeta;
    return `${base}_${year}_${page}`;
  }
  return `${base}_${uploadedImage?.updatedAt ?? "upload"}`;
}

export function getSolveCache(key: string): CachedSolve | null {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    if (!raw) return null;
    const data = JSON.parse(raw) as CachedSolve;
    if (!data?.structuredResults?.length || !data?.flowmapData?.groups) return null;
    return data;
  } catch {
    return null;
  }
}

export function setSolveCache(key: string, data: CachedSolve): void {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(data));
  } catch {
    // quota or disabled localStorage
  }
}
