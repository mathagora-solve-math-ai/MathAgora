import type { Detection } from "./mockData";

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

export async function loadDatasetIndex(): Promise<DatasetIndex> {
  const res = await fetch("/data/datasets.json");
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
    `/data/outputs_parsing/${year}_math_odd/page_${page}/page_${page}.json`;
  const res = await fetch(jsonPath);
  if (!res.ok) {
    throw new Error("Failed to load dataset page JSON");
  }

  const json = (await res.json()) as RawPage;
  const cropTimestamp = index?.years?.[year]?.pages?.[page]?.cropTimestamp;

  const detections: Detection[] = json.questions.map((q) => {
    const [x1, y1, x2, y2] = q.bbox;
    const label = `${truncate(q.merged_text.replace(/\s+/g, " "))}`;
    const cropUrl = cropTimestamp
      ? `/data/outputs_parsing/${year}_math_odd/page_${page}/questions/${q.qid}/page_${page}__${q.qid}__${cropTimestamp}.png`
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

  return {
    imageUrl: `/data/${year}_math_odd/page_${page}.png`,
    detections,
    meta: {
      year,
      set: "math_odd",
      page,
    },
    imageWidth: json.width,
    imageHeight: json.height,
  };
}
