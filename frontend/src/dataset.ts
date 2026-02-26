import type { Detection } from "./mockData.ts";

export type DatasetIndex = {
  years: Record<
    string,
    {
      pages: Record<
        string,
        {
          json?: string;
          cropTimestamp?: string;
        }
      >;
    }
  >;
};

export type DatasetPage = {
  imageUrl: string;
  detections: Detection[];
  meta: {
    year: string;
    set: string;
    page: string;
  };
  imageWidth?: number;
  imageHeight?: number;
};

type RawQuestion = {
  qid: string;
  merged_text: string;
  bbox: [number, number, number, number];
};

type RawPage = {
  page_image: string;
  questions: RawQuestion[];
  width?: number;
  height?: number;
};

const inferTrackByPage = (page: number): "common" | "probability" | "calculus" | "geometry" | null => {
  if (page >= 1 && page <= 8) return "common";
  if (page >= 9 && page <= 12) return "probability";
  if (page >= 13 && page <= 16) return "calculus";
  if (page >= 17 && page <= 20) return "geometry";
  return null;
};

const buildProbId = (year: string, page: string, qid: string): string => {
  if (year === "sat") {
    return `sat_${page}_${qid}`;
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
  year?: string;
  page?: string;
  index?: DatasetIndex;
}): Promise<DatasetPage> {
  const year = params?.year ?? "2024";
  const page = params?.page ?? "001";
  const index = params?.index;
  const jsonPath =
    index?.years?.[year]?.pages?.[page]?.json ??
    `${dataBase}/data/outputs_parsing/${year}_math_odd/page_${page}/page_${page}.json`;
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
  const cropTimestamp = index?.years?.[year]?.pages?.[page]?.cropTimestamp;

  const detections: Detection[] = json.questions.map((q) => {
    const [x1, y1, x2, y2] = q.bbox;
    const label = `${truncate(q.merged_text.replace(/\s+/g, " "))}`;
    const cropUrl = cropTimestamp
      ? `${dataBase}/data/outputs_parsing/${year}_math_odd/page_${page}/questions/${q.qid}/page_${page}__${q.qid}__${cropTimestamp}.png`
      : undefined;
    return {
      id: buildProbId(year, page, q.qid),
      x: x1,
      y: y1,
      w: x2 - x1,
      h: y2 - y1,
      label,
      text: q.merged_text,
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

  return {
    imageUrl: `${dataBase}/data/${year}_math_odd/page_${page}.png`,
    detections,
    meta: {
      year,
      set: "math_odd",
      page,
    },
    imageWidth,
    imageHeight,
  };
}
