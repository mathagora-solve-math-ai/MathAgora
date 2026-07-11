import type { Detection } from "./mockData.ts";

export type DatasetGroupId = "csat" | "sat";

export type DatasetPageEntry = {
  json?: string;
  image?: string;
  cropTimestamp?: string;
  pageKey?: string;
  label?: string;
};

export type DatasetYearEntry = {
  label?: string;
  pages: Record<string, DatasetPageEntry>;
};

export type DatasetGroupEntry = {
  label?: string;
  years: Record<string, DatasetYearEntry>;
};

export type DatasetIndex = {
  groups?: Partial<Record<DatasetGroupId, DatasetGroupEntry>>;
  years?: Record<string, DatasetYearEntry>;
};

export type DatasetMeta = {
  group: DatasetGroupId;
  year: string;
  set: string;
  page: string;
  pageKey: string;
  label: string;
};

export type DatasetPage = {
  imageUrl: string;
  detections: Detection[];
  meta: DatasetMeta;
  imageWidth?: number;
  imageHeight?: number;
};

type RawQuestion = {
  qid: string;
  merged_text?: string;
  postcorrection_text?: string;
  crop?: string;
  bbox: [number, number, number, number];
};

type RawPage = {
  page_image: string;
  questions: RawQuestion[];
  width?: number;
  height?: number;
};

const DATASET_GROUP_ORDER: DatasetGroupId[] = ["csat", "sat"];

const inferTrackByPage = (page: number): "common" | "probability" | "calculus" | "geometry" | null => {
  if (page >= 1 && page <= 8) return "common";
  if (page >= 9 && page <= 12) return "probability";
  if (page >= 13 && page <= 16) return "calculus";
  if (page >= 17 && page <= 20) return "geometry";
  return null;
};

const compareNatural = (a: string, b: string) =>
  a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });

const sortYears = (group: DatasetGroupId, years: string[]) =>
  [...years].sort((a, b) => (group === "csat" ? compareNatural(b, a) : compareNatural(a, b)));

const sortPages = (pages: string[]) => [...pages].sort(compareNatural);

const withBase = (path: string): string => {
  if (/^https?:\/\//.test(path) || path.startsWith("data:")) return path;
  return `${dataBase}${path.startsWith("/") ? path : `/${path}`}`;
};

const legacySatPageKey = (year: string, page: string): string => {
  if (page.includes("_")) return page;
  if (year && year !== "sat") return `${year}_${page}`;
  return page;
};

const splitLegacySatPage = (year?: string, page?: string) => {
  if (year === "sat" && page?.includes("_")) {
    const [test, ...rest] = page.split("_");
    return { year: test, page: rest.join("_") };
  }
  return { year, page };
};

const getGroupedYearEntry = (
  index: DatasetIndex | null | undefined,
  group: DatasetGroupId,
  year: string,
) => index?.groups?.[group]?.years?.[year];

const getLegacySatYearEntry = (
  index: DatasetIndex | null | undefined,
  year: string,
): DatasetYearEntry | undefined => {
  const sat = index?.years?.sat;
  if (!sat) return undefined;
  if (year === "sat") return sat;
  const prefix = `${year}_`;
  const pages = Object.entries(sat.pages).reduce<Record<string, DatasetPageEntry>>((acc, [pageKey, entry]) => {
    if (!pageKey.startsWith(prefix)) return acc;
    acc[pageKey.slice(prefix.length)] = { ...entry, pageKey };
    return acc;
  }, {});
  return Object.keys(pages).length ? { label: `Practice Test ${year}`, pages } : undefined;
};

const getYearEntry = (
  index: DatasetIndex | null | undefined,
  group: DatasetGroupId,
  year: string,
): DatasetYearEntry | undefined => {
  const grouped = getGroupedYearEntry(index, group, year);
  if (grouped) return grouped;
  if (group === "csat") return index?.years?.[year];
  return getLegacySatYearEntry(index, year);
};

const getPageEntry = (
  index: DatasetIndex | null | undefined,
  group: DatasetGroupId,
  year: string,
  page: string,
): DatasetPageEntry | undefined => {
  const grouped = getGroupedYearEntry(index, group, year)?.pages?.[page];
  if (grouped) return grouped;
  if (group === "csat") return index?.years?.[year]?.pages?.[page];
  return index?.years?.sat?.pages?.[legacySatPageKey(year, page)];
};

export function getDatasetGroupIds(
  index: DatasetIndex | null | undefined,
  activeGroup?: DatasetGroupId,
): DatasetGroupId[] {
  const ids = DATASET_GROUP_ORDER.filter((group) => {
    if (index?.groups?.[group]) return true;
    if (group === "csat") return Object.keys(index?.years ?? {}).some((year) => year !== "sat");
    return Boolean(index?.years?.sat);
  });
  if (!ids.length && activeGroup) return [activeGroup];
  return ids.length ? ids : ["csat"];
}

export function getDatasetGroupLabel(
  index: DatasetIndex | null | undefined,
  group: DatasetGroupId,
): string {
  return index?.groups?.[group]?.label ?? group.toUpperCase();
}

export function getDatasetYears(
  index: DatasetIndex | null | undefined,
  group: DatasetGroupId,
): string[] {
  const groupedYears = index?.groups?.[group]?.years;
  if (groupedYears) return sortYears(group, Object.keys(groupedYears));
  if (group === "csat") {
    return sortYears(group, Object.keys(index?.years ?? {}).filter((year) => year !== "sat"));
  }
  const legacySatPages = index?.years?.sat?.pages ?? {};
  const tests = new Set<string>();
  Object.keys(legacySatPages).forEach((pageKey) => {
    const [test] = pageKey.split("_");
    if (test) tests.add(test);
  });
  return sortYears(group, [...tests]);
}

export function getDatasetYearLabel(
  index: DatasetIndex | null | undefined,
  group: DatasetGroupId,
  year: string,
): string {
  const entry = getYearEntry(index, group, year);
  if (entry?.label) return entry.label;
  return group === "sat" ? `Practice Test ${year}` : year;
}

export function getDatasetPages(
  index: DatasetIndex | null | undefined,
  group: DatasetGroupId,
  year: string,
): string[] {
  return sortPages(Object.keys(getYearEntry(index, group, year)?.pages ?? {}));
}

export function getDatasetPageLabel(
  index: DatasetIndex | null | undefined,
  group: DatasetGroupId,
  year: string,
  page: string,
): string {
  return getPageEntry(index, group, year, page)?.label ?? `page_${page}`;
}

export function resolveDatasetSelection(
  index: DatasetIndex | null | undefined,
  preferredGroup?: DatasetGroupId,
  preferredYear?: string,
  preferredPage?: string,
): { group: DatasetGroupId; year: string; page: string } {
  const groups = getDatasetGroupIds(index, preferredGroup);
  const group = preferredGroup && groups.includes(preferredGroup) ? preferredGroup : groups[0] ?? "csat";
  const split = group === "sat" ? splitLegacySatPage(preferredYear, preferredPage) : { year: preferredYear, page: preferredPage };
  const years = getDatasetYears(index, group);
  const year = split.year && years.includes(split.year)
    ? split.year
    : years[0] ?? (group === "sat" ? "8" : "2025");
  const pages = getDatasetPages(index, group, year);
  const page = split.page && pages.includes(split.page) ? split.page : pages[0] ?? "001";
  return { group, year, page };
}

const buildProbId = (
  group: DatasetGroupId,
  year: string,
  page: string,
  qid: string,
  pageKey: string,
): string => {
  if (group === "sat") {
    return `sat_${pageKey}_${qid}`;
  }
  if (year === "2026") {
    const pageStr = String(page).padStart(3, "0");
    return `${year}_math_odd_page_${pageStr}_${qid}`;
  }
  const qNum = Number(qid);
  if (!Number.isFinite(qNum) || qNum <= 0 || !Number.isInteger(qNum)) {
    return qid;
  }
  if (qNum <= 22) {
    return `${year}_odd_common_${qNum}`;
  }
  const pageNum = Number(page);
  const track = inferTrackByPage(pageNum);
  if (!track) {
    return qid;
  }
  return `${year}_odd_${track}_${qNum}`;
};

const truncate = (text: string, max = 72) =>
  text.length <= max ? text : `${text.slice(0, max).trim()}...`;

/** Base path for static data (respects Vite baseUrl so /data works when app is under a subpath). */
const baseRaw =
  typeof import.meta !== "undefined" ? (import.meta.env?.BASE_URL ?? "/") : "/";
const dataBase = (typeof baseRaw === "string" ? baseRaw : "/").replace(/\/$/, "");

export async function loadDatasetIndex(): Promise<DatasetIndex> {
  const url = `${dataBase}/data/datasets.json`;
  let res: Response;
  try {
    res = await fetch(url);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new Error(
      msg.includes("fetch") || msg.includes("NetworkError")
        ? "Sample index could not be loaded. Run the app with a local server (e.g. npm run dev) and ensure public/data exists."
        : msg,
    );
  }
  if (!res.ok) throw new Error("Failed to load datasets index");
  return (await res.json()) as DatasetIndex;
}

export async function loadDatasetPage(params?: {
  group?: DatasetGroupId;
  year?: string;
  page?: string;
  index?: DatasetIndex;
}): Promise<DatasetPage> {
  const inferredGroup: DatasetGroupId = params?.group ?? (params?.year === "sat" ? "sat" : "csat");
  const selection = params?.index
    ? resolveDatasetSelection(params.index, inferredGroup, params?.year, params?.page)
    : {
        group: inferredGroup,
        year: params?.year ?? (inferredGroup === "sat" ? "8" : "2024"),
        page: params?.page ?? "001",
      };
  const { group, year, page } = selection;
  const index = params?.index;
  const pageEntry = getPageEntry(index, group, year, page);
  const pageKey = pageEntry?.pageKey ?? (group === "sat" ? legacySatPageKey(year, page) : page);
  const rawJsonPath = pageEntry?.json;
  const jsonPath = rawJsonPath != null
    ? withBase(rawJsonPath)
    : group === "sat"
      ? withBase(`/data/outputs_parsing/sat_math_odd/page_${pageKey}/page_${pageKey}.json`)
      : withBase(`/data/outputs_parsing/${year}_math_odd/page_${page}/page_${page}.json`);
  let res: Response;
  try {
    res = await fetch(jsonPath);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new Error(
      msg.includes("fetch") || msg.includes("NetworkError")
        ? "Sample data could not be loaded. Run the app with a local server (e.g. npm run dev) and ensure /data files are available."
        : msg,
    );
  }
  if (!res.ok) {
    throw new Error("Failed to load dataset page JSON");
  }

  const json = (await res.json()) as RawPage;
  const cropTimestamp = pageEntry?.cropTimestamp;

  const detections: Detection[] = json.questions.map((q) => {
    const [x1, y1, x2, y2] = q.bbox;
    const text = q.postcorrection_text || q.merged_text || "";
    const label = `${truncate(text.replace(/\s+/g, " "))}`;
    const cropPath = group === "sat"
      ? `/data/outputs_parsing/sat_math_odd/page_${pageKey}/questions/${q.qid}/page_${pageKey}__${q.qid}__${cropTimestamp}.png`
      : `/data/outputs_parsing/${year}_math_odd/page_${page}/questions/${q.qid}/page_${page}__${q.qid}__${cropTimestamp}.png`;
    const cropUrl = cropTimestamp ? withBase(cropPath) : q.crop ? withBase(q.crop) : undefined;
    return {
      id: buildProbId(group, year, page, q.qid, pageKey),
      x: x1,
      y: y1,
      w: x2 - x1,
      h: y2 - y1,
      label,
      text,
      cropUrl,
    };
  });

  let imageWidth = json.width;
  let imageHeight = json.height;
  if (imageWidth == null || imageHeight == null) {
    const maxX = Math.max(...detections.map((d) => d.x + d.w), 0);
    const maxY = Math.max(...detections.map((d) => d.y + d.h), 0);
    imageWidth = imageWidth ?? (maxX > 0 ? maxX : 1);
    imageHeight = imageHeight ?? (maxY > 0 ? maxY : 1);
  }

  const imageUrl = pageEntry?.image
    ? withBase(pageEntry.image)
    : group === "sat"
      ? withBase(`/data/sat_math_odd/page_${pageKey}.png`)
      : withBase(`/data/${year}_math_odd/page_${page}.png`);
  const label = group === "sat"
    ? `SAT Practice Test ${year} page_${page}`
    : `CSAT ${year} page_${page}`;

  return {
    imageUrl,
    detections,
    meta: {
      group,
      year,
      set: group === "sat" ? "sat_math_odd" : "math_odd",
      page,
      pageKey,
      label,
    },
    imageWidth,
    imageHeight,
  };
}
