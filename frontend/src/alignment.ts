import type { ModelResult, Step } from "./mockData.ts";

export type AlignedStep = Step & {
  alignmentGroupId?: string;
  similarityHint?: number;
};

export type AlignedModelResult = Omit<ModelResult, "steps"> & {
  steps: AlignedStep[];
};

export type AlignOptions = {
  threshold?: number;
  stopwords?: string[];
  match?: {
    mode?: "mutual_best" | "mutual_topk";
    topK?: number;
    indexPenaltyLambda?: number;
  };
  embedding?: {
    enabled: boolean;
    getEmbedding: (text: string) => Promise<number[]> | number[];
  };
};

type AlignmentOutput = {
  results: AlignedModelResult[];
  groups: {
    id: string;
    label: string;
    members: { modelId: string; stepId: string; score?: number }[];
  }[];
};

type StepNode = {
  modelId: string;
  stepId: string;
  title: string;
  body: string;
};

type SparseVector = Map<string, number>;

type StepFeature = {
  key: string;
  modelId: string;
  stepId: string;
  stepIndex: number;
  title: string;
  body: string;
  signature: string;
  titleTokens: string[];
  bodyTokens: string[];
  mathTokens: string[];
  charTitleTokens: string[];
  titleVector: SparseVector;
  bodyVector: SparseVector;
  mathVector: SparseVector;
  charTitleVector: SparseVector;
};

const DEFAULT_STOPWORDS = new Set([
  "the",
  "and",
  "of",
  "to",
  "in",
  "on",
  "for",
  "a",
  "an",
  "with",
  "is",
  "are",
  "을",
  "를",
  "이",
  "가",
  "은",
  "는",
  "에",
  "의",
  "으로",
  "로",
  "과",
  "와",
  "에서",
  "하다",
  "한다",
  "및",
]);

const MATH_KEYWORDS = [
  "sin",
  "cos",
  "tan",
  "log",
  "ln",
  "sqrt",
  "abs",
  "exp",
  "lim",
  "sum",
  "prod",
  "integral",
  "derivative",
  "미분",
  "적분",
  "로그",
  "사인",
  "코사인",
  "탄젠트",
  "루트",
  "절댓값",
  "지수",
  "합",
  "곱",
  "극한",
];

const RELATION_SYMBOLS = new Set(["=", "≠", "≤", "≥", "<", ">"]);
const OPERATOR_SYMBOLS = new Set(["+", "-", "*", "/", "^", "%", "·"]);
const STRUCTURE_SYMBOLS = new Set(["(", ")", "[", "]", "{", "}"]);

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));

const buildStopwords = (extra?: string[]) => {
  if (!extra?.length) return DEFAULT_STOPWORDS;
  return new Set([...DEFAULT_STOPWORDS, ...extra]);
};

const normalizeText = (text: string) =>
  text
    .toLowerCase()
    .replace(/\\[a-zA-Z]+/g, " ")
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();

const tokenizeText = (text: string, stopwords: Set<string>) =>
  normalizeText(text)
    .split(" ")
    .filter((token) => token && !stopwords.has(token));

const tokenizeCharNgrams = (text: string, n = 2) => {
  const compact = normalizeText(text).replace(/\s+/g, "");
  if (!compact) return [];
  if (compact.length < n) return [compact];
  const grams: string[] = [];
  for (let i = 0; i <= compact.length - n; i += 1) {
    grams.push(compact.slice(i, i + n));
  }
  return grams;
};

const pushRegexMatches = (
  text: string,
  pattern: RegExp,
  mapper: (match: string) => string,
  out: string[],
) => {
  let match: RegExpExecArray | null = null;
  while ((match = pattern.exec(text)) !== null) {
    out.push(mapper(match[0]));
  }
};

const extractMathTokens = (text: string) => {
  const out: string[] = [];
  const source = text;
  const lower = text.toLowerCase();

  // Keep math structures explicitly so symbolic equivalence is less brittle.
  pushRegexMatches(lower, /\\frac\{[^}]+\}\{[^}]+\}/g, () => "latex:frac", out);
  pushRegexMatches(lower, /\\sqrt\{[^}]+\}/g, () => "latex:sqrt", out);
  pushRegexMatches(lower, /\\[a-zA-Z]+/g, (m) => `latex:${m.slice(1)}`, out);
  pushRegexMatches(lower, /[a-z]\^\{?[-]?[a-z0-9]+\}?/g, () => "pattern:power", out);
  pushRegexMatches(lower, /[a-z]_\{?[a-z0-9]+\}?/g, () => "pattern:subscript", out);

  pushRegexMatches(source, /\b\d+(?:\.\d+)?\/\d+(?:\.\d+)?\b/g, () => "num:fraction", out);
  pushRegexMatches(source, /\b\d+(?:\.\d+)?\b/g, (m) => `num:${m}`, out);
  pushRegexMatches(lower, /sqrt\s*\([^)]*\)/g, () => "expr:sqrt", out);
  pushRegexMatches(source, /\|[^|]+\|/g, () => "expr:abs", out);

  for (const char of source) {
    if (RELATION_SYMBOLS.has(char)) out.push(`rel:${char}`);
    else if (OPERATOR_SYMBOLS.has(char)) out.push(`op:${char}`);
    else if (STRUCTURE_SYMBOLS.has(char)) out.push(`struct:${char}`);
  }

  pushRegexMatches(source, /\b[a-zA-Z](?:_[a-zA-Z0-9]+|\^\{?[a-zA-Z0-9-]+\}?)*\b/g, (m) => `var:${m.toLowerCase()}`, out);
  pushRegexMatches(source, /[α-ωΑ-ΩπθλμσΔΣΩ]/g, (m) => `greek:${m.toLowerCase()}`, out);

  MATH_KEYWORDS.forEach((kw) => {
    const re = /[a-z]/i.test(kw)
      ? new RegExp(`\\b${kw}\\b`, "gi")
      : new RegExp(kw, "g");
    let count = 0;
    while (re.exec(lower) !== null) count += 1;
    for (let i = 0; i < count; i += 1) {
      out.push(`kw:${kw}`);
    }
  });

  return out;
};

const buildCountMap = (tokens: string[]) => {
  const counts = new Map<string, number>();
  tokens.forEach((token) => {
    counts.set(token, (counts.get(token) ?? 0) + 1);
  });
  return counts;
};

const normalizeVector = (vector: SparseVector) => {
  let sqSum = 0;
  vector.forEach((value) => {
    sqSum += value * value;
  });
  if (!sqSum) return vector;
  const norm = Math.sqrt(sqSum);
  const normalized = new Map<string, number>();
  vector.forEach((value, key) => {
    normalized.set(key, value / norm);
  });
  return normalized;
};

const buildTfIdfVectors = (docs: string[][]) => {
  const df = new Map<string, number>();
  docs.forEach((tokens) => {
    const seen = new Set(tokens);
    seen.forEach((token) => {
      df.set(token, (df.get(token) ?? 0) + 1);
    });
  });

  const docCount = docs.length;
  return docs.map((tokens) => {
    const tf = buildCountMap(tokens);
    const total = tokens.length || 1;
    const vector = new Map<string, number>();
    tf.forEach((count, token) => {
      const idf = Math.log((docCount + 1) / ((df.get(token) ?? 0) + 1)) + 1;
      vector.set(token, (count / total) * idf);
    });
    return normalizeVector(vector);
  });
};

const buildCountVectors = (docs: string[][]) =>
  docs.map((tokens) => {
    const counts = buildCountMap(tokens);
    return normalizeVector(counts);
  });

const cosineSimilarity = (a: SparseVector, b: SparseVector) => {
  if (!a.size || !b.size) return 0;
  const [small, large] = a.size <= b.size ? [a, b] : [b, a];
  let dot = 0;
  small.forEach((value, key) => {
    dot += value * (large.get(key) ?? 0);
  });
  return clamp01(dot);
};

const cosineArray = (a: number[], b: number[]) => {
  if (!a.length || !b.length || a.length !== b.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (!normA || !normB) return 0;
  return clamp01(dot / (Math.sqrt(normA) * Math.sqrt(normB)));
};

const stepSignature = (step: Step) => {
  const baseTitle = step.title || step.body.split(" ").slice(0, 10).join(" ");
  return `${baseTitle}\n${baseTitle}\n${step.body}`;
};

const buildStepFeatures = (results: ModelResult[], stopwords: Set<string>) => {
  const features: StepFeature[] = [];
  const nodes: StepNode[] = [];
  const nodeIndex = new Map<string, number>();
  const modelStepKeys = new Map<string, string[]>();

  results.forEach((result) => {
    const keys: string[] = [];
    result.steps.forEach((step, stepIndex) => {
      const key = `${result.modelId}:${step.id}`;
      const signature = stepSignature(step);
      const feature: StepFeature = {
        key,
        modelId: result.modelId,
        stepId: step.id,
        stepIndex,
        title: step.title,
        body: step.body,
        signature,
        titleTokens: tokenizeText(step.title || step.body, stopwords),
        bodyTokens: tokenizeText(step.body, stopwords),
        mathTokens: extractMathTokens(signature),
        charTitleTokens: tokenizeCharNgrams(step.title || step.body, 2),
        titleVector: new Map(),
        bodyVector: new Map(),
        mathVector: new Map(),
        charTitleVector: new Map(),
      };

      nodeIndex.set(key, nodes.length);
      nodes.push({
        modelId: result.modelId,
        stepId: step.id,
        title: step.title,
        body: step.body,
      });
      features.push(feature);
      keys.push(key);
    });
    modelStepKeys.set(result.modelId, keys);
  });

  const titleVectors = buildTfIdfVectors(features.map((f) => f.titleTokens));
  const bodyVectors = buildTfIdfVectors(features.map((f) => f.bodyTokens));
  const mathVectors = buildTfIdfVectors(features.map((f) => f.mathTokens));
  const charVectors = buildCountVectors(features.map((f) => f.charTitleTokens));

  features.forEach((feature, idx) => {
    feature.titleVector = titleVectors[idx];
    feature.bodyVector = bodyVectors[idx];
    feature.mathVector = mathVectors[idx];
    feature.charTitleVector = charVectors[idx];
  });

  return { features, nodes, nodeIndex, modelStepKeys };
};

const deriveLabel = (steps: StepNode[], stopwords: Set<string>) => {
  const freq = new Map<string, number>();
  steps.forEach((step) => {
    const titleTokens = tokenizeText(step.title, stopwords);
    const fallbackBody = titleTokens.length
      ? []
      : tokenizeText(step.body, stopwords).slice(0, 6);
    [...titleTokens, ...fallbackBody].forEach((token) => {
      freq.set(token, (freq.get(token) ?? 0) + 1);
    });
  });

  const sorted = [...freq.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 2)
    .map(([token]) => token);
  return sorted.length ? sorted.join(" ") : "Group";
};

const unionFind = (size: number) => {
  const parent = Array.from({ length: size }, (_, i) => i);
  const find = (x: number): number =>
    parent[x] === x ? x : (parent[x] = find(parent[x]));
  const union = (a: number, b: number) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[rb] = ra;
  };
  return { find, union };
};

const topKIndices = (values: number[], k: number) =>
  values
    .map((value, idx) => ({ value, idx }))
    .sort((a, b) => b.value - a.value)
    .slice(0, Math.max(1, k))
    .map((item) => item.idx);

const collectEdges = (
  scores: number[][],
  threshold: number,
  mode: "mutual_best" | "mutual_topk",
  topK: number,
) => {
  const edges: Array<{ i: number; j: number; score: number }> = [];
  if (!scores.length || !scores[0]?.length) return edges;

  const rowBest = scores.map((row) => topKIndices(row, 1)[0]);
  const colBest = Array.from({ length: scores[0].length }, (_, colIdx) => {
    const column = scores.map((row) => row[colIdx]);
    return topKIndices(column, 1)[0];
  });

  if (mode === "mutual_best") {
    rowBest.forEach((bestJ, i) => {
      const score = scores[i][bestJ];
      if (score >= threshold && colBest[bestJ] === i) {
        edges.push({ i, j: bestJ, score });
      }
    });
    return edges;
  }

  const rowTopK = scores.map((row) => new Set(topKIndices(row, topK)));
  const colTopK = Array.from({ length: scores[0].length }, (_, colIdx) => {
    const column = scores.map((row) => row[colIdx]);
    return new Set(topKIndices(column, topK));
  });

  rowTopK.forEach((candJs, i) => {
    candJs.forEach((j) => {
      const score = scores[i][j];
      if (score >= threshold && colTopK[j].has(i)) {
        edges.push({ i, j, score });
      }
    });
  });

  return edges;
};

const resolveEmbeddingsSync = (
  features: StepFeature[],
  embedding: NonNullable<AlignOptions["embedding"]>,
) => {
  const map = new Map<string, number[]>();
  for (const feature of features) {
    const raw = embedding.getEmbedding(feature.signature);
    if (
      raw &&
      typeof raw === "object" &&
      "then" in raw &&
      typeof (raw as Promise<number[]>).then === "function"
    ) {
      throw new Error(
        "Async embedding provider detected. Use alignStepsAcrossModelsAsync.",
      );
    }
    map.set(feature.key, raw as number[]);
  }
  return map;
};

const resolveEmbeddingsAsync = async (
  features: StepFeature[],
  embedding: NonNullable<AlignOptions["embedding"]>,
) => {
  const map = new Map<string, number[]>();
  for (const feature of features) {
    const emb = await Promise.resolve(embedding.getEmbedding(feature.signature));
    map.set(feature.key, emb);
  }
  return map;
};

const computeScore = (
  a: StepFeature,
  b: StepFeature,
  embeddingMap?: Map<string, number[]>,
) => {
  const tfidfTitleCos = cosineSimilarity(a.titleVector, b.titleVector);
  const tfidfBodyCos = cosineSimilarity(a.bodyVector, b.bodyVector);
  const tfidfMathCos = cosineSimilarity(a.mathVector, b.mathVector);
  const charTitleCos = cosineSimilarity(a.charTitleVector, b.charTitleVector);

  const components = [
    { value: tfidfTitleCos, weight: 0.3 },
    { value: tfidfBodyCos, weight: 0.3 },
    { value: tfidfMathCos, weight: 0.2 },
    { value: charTitleCos, weight: 0.1 },
  ];

  if (embeddingMap) {
    const embA = embeddingMap.get(a.key);
    const embB = embeddingMap.get(b.key);
    if (embA && embB) {
      components.push({ value: cosineArray(embA, embB), weight: 0.1 });
    }
  }

  const weightSum = components.reduce((sum, c) => sum + c.weight, 0) || 1;
  const score =
    components.reduce((sum, c) => sum + c.value * c.weight, 0) / weightSum;
  return clamp01(score);
};

const runAlignment = (
  results: ModelResult[],
  opts?: AlignOptions,
  embeddingMap?: Map<string, number[]>,
): AlignmentOutput => {
  const stopwords = buildStopwords(opts?.stopwords);
  const threshold = opts?.threshold ?? 0.35;
  const mode = opts?.match?.mode ?? "mutual_best";
  const topK = opts?.match?.topK ?? 2;
  const penaltyLambda = opts?.match?.indexPenaltyLambda ?? 0.05;

  const { features, nodes, nodeIndex, modelStepKeys } = buildStepFeatures(
    results,
    stopwords,
  );
  const featureByKey = new Map(features.map((feature) => [feature.key, feature]));

  const uf = unionFind(nodes.length);
  const pairScores = new Map<string, number>();

  for (let i = 0; i < results.length; i += 1) {
    for (let j = i + 1; j < results.length; j += 1) {
      const modelA = results[i];
      const modelB = results[j];
      const keysA = modelStepKeys.get(modelA.modelId) ?? [];
      const keysB = modelStepKeys.get(modelB.modelId) ?? [];
      const maxLen = Math.max(keysA.length, keysB.length, 1);

      const scores: number[][] = keysA.map((keyA, idxA) =>
        keysB.map((keyB, idxB) => {
          const featureA = featureByKey.get(keyA);
          const featureB = featureByKey.get(keyB);
          if (!featureA || !featureB) return 0;
          const rawScore = computeScore(featureA, featureB, embeddingMap);
          const penalty = (penaltyLambda * Math.abs(idxA - idxB)) / maxLen;
          return clamp01(rawScore - penalty);
        }),
      );

      const edges = collectEdges(scores, threshold, mode, topK);
      edges.forEach(({ i: idxA, j: idxB, score }) => {
        const keyA = keysA[idxA];
        const keyB = keysB[idxB];
        const nodeA = nodeIndex.get(keyA);
        const nodeB = nodeIndex.get(keyB);
        if (nodeA === undefined || nodeB === undefined) return;

        uf.union(nodeA, nodeB);
        pairScores.set(`${nodeA}-${nodeB}`, score);
        pairScores.set(`${nodeB}-${nodeA}`, score);
      });
    }
  }

  const clusters = new Map<number, number[]>();
  nodes.forEach((_, index) => {
    const root = uf.find(index);
    const list = clusters.get(root) ?? [];
    list.push(index);
    clusters.set(root, list);
  });

  const groups = [...clusters.values()]
    .map((indices) => {
      const perModel = new Map<string, { idx: number; score: number }>();
      indices.forEach((idx) => {
        const node = nodes[idx];
        let best = 0;
        indices.forEach((other) => {
          if (idx === other) return;
          const score = pairScores.get(`${idx}-${other}`) ?? 0;
          best = Math.max(best, score);
        });

        const existing = perModel.get(node.modelId);
        if (!existing || best > existing.score) {
          perModel.set(node.modelId, { idx, score: best });
        }
      });

      const members = [...perModel.values()].map(({ idx, score }) => ({
        modelId: nodes[idx].modelId,
        stepId: nodes[idx].stepId,
        score,
        idx,
      }));

      return {
        indices: members.map((m) => m.idx),
        members: members.map((m) => ({
          modelId: m.modelId,
          stepId: m.stepId,
          score: m.score,
        })),
      };
    })
    .filter((group) => group.members.length >= 2);

  const labeledGroups = groups
    .map((group, idx) => ({
      id: `G${idx + 1}`,
      label: deriveLabel(group.indices.map((idx2) => nodes[idx2]), stopwords),
      members: group.members,
      size: group.members.length,
    }))
    .sort((a, b) => b.size - a.size || a.label.localeCompare(b.label))
    .map((group, idx) => ({
      id: `G${idx + 1}`,
      label: group.label,
      members: group.members,
    }));

  const alignmentMap = new Map<string, { id: string; score: number }>();
  labeledGroups.forEach((group) => {
    group.members.forEach((member) => {
      alignmentMap.set(`${member.modelId}:${member.stepId}`, {
        id: group.id,
        score: member.score ?? 0,
      });
    });
  });

  const alignedResults: AlignedModelResult[] = results.map((result) => ({
    ...result,
    steps: result.steps.map((step) => {
      const meta = alignmentMap.get(`${result.modelId}:${step.id}`);
      return {
        ...step,
        alignmentGroupId: meta?.id,
        similarityHint: meta?.score,
      };
    }),
  }));

  return { results: alignedResults, groups: labeledGroups };
};

export function alignStepsAcrossModels(
  results: ModelResult[],
  opts?: AlignOptions,
): AlignmentOutput {
  if (opts?.embedding?.enabled) {
    const stopwords = buildStopwords(opts.stopwords);
    const { features } = buildStepFeatures(results, stopwords);
    const embeddingMap = resolveEmbeddingsSync(features, opts.embedding);
    return runAlignment(results, opts, embeddingMap);
  }
  return runAlignment(results, opts);
}

export async function alignStepsAcrossModelsAsync(
  results: ModelResult[],
  opts?: AlignOptions,
): Promise<AlignmentOutput> {
  if (opts?.embedding?.enabled) {
    const stopwords = buildStopwords(opts.stopwords);
    const { features } = buildStepFeatures(results, stopwords);
    const embeddingMap = await resolveEmbeddingsAsync(features, opts.embedding);
    return runAlignment(results, opts, embeddingMap);
  }
  return runAlignment(results, opts);
}

// Minimal manual smoke-test helper (no test framework required).
export function runAlignmentSmokeTest() {
  const mockResults: ModelResult[] = [
    {
      modelId: "A",
      modelName: "A",
      version: "v1",
      latencyMs: 1,
      temperature: 0,
      strategy: "",
      finalAnswer: "",
      steps: [
        {
          id: "1",
          title: "식 정리",
          body: "\\(x f(x)-f(x)=(x-1)f(x)\\)로 정리한다.",
        },
      ],
    },
    {
      modelId: "B",
      modelName: "B",
      version: "v1",
      latencyMs: 1,
      temperature: 0,
      strategy: "",
      finalAnswer: "",
      steps: [
        {
          id: "1",
          title: "방정식 변형",
          body: "\\(x f(x)-f(x)\\)를 묶어 \\(x-1)f(x)\\)를 얻는다.",
        },
      ],
    },
  ];

  return alignStepsAcrossModels(mockResults, { threshold: 0.2 });
}
