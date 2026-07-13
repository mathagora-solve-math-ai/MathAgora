export type ParsedSolution = {
  strategy: string;
  steps: { title: string; body: string }[];
  finalAnswer: string;
};

export type FinalAnswerContext = {
  documentType?: "csat" | "sat" | null;
  problemId?: string | null;
  problemText?: string | null;
};

const SAT_INDEX_TO_LETTER: Record<string, string> = {
  "1": "A",
  "2": "B",
  "3": "C",
  "4": "D",
};

const isSatContext = (context?: FinalAnswerContext): boolean => {
  if (context?.documentType === "sat") return true;
  return Boolean(context?.problemId?.toLowerCase().startsWith("sat_"));
};

const stripAnswerDecorations = (answer: string): string => {
  let value = answer.trim();
  value = value.replace(/^["'`]+|["'`]+$/g, "").trim();
  const boxed = value.match(/^\\boxed\{(.+)\}$/);
  if (boxed) value = boxed[1].trim();
  return value;
};

const getSatOptionLabels = (problemText: string): Set<string> => {
  const labels = new Set<string>();
  const optionRegex = /(?:^|[\s|])([A-D])\s*[).:]\s*/g;
  for (const match of problemText.matchAll(optionRegex)) {
    labels.add(match[1].toUpperCase());
  }
  return labels;
};

const hasSatMultipleChoiceOptions = (problemText: string): boolean => {
  const labels = getSatOptionLabels(problemText);
  return ["A", "B", "C", "D"].every((label) => labels.has(label));
};

const hasSatChoiceMarkers = (problemText: string): boolean => {
  const labels = getSatOptionLabels(problemText);
  return labels.size >= 2;
};

const isLikelySatMultipleChoice = (problemText: string): boolean => {
  if (hasSatMultipleChoiceOptions(problemText) || hasSatChoiceMarkers(problemText)) {
    return true;
  }
  return /\bwhich\s+(?:of\s+the\s+following|equation|expression|function|graph|table|statement)\b/i.test(
    problemText,
  );
};

const normalizeOptionValue = (value: string): string =>
  value
    .replace(/\\frac\{([^{}]+)\}\{([^{}]+)\}/g, "$1/$2")
    .replace(/\\\(|\\\)/g, "")
    .replace(/[{}\\]/g, "")
    .replace(/\s+/g, "")
    .replace(/[.,;:]+$/g, "")
    .toLowerCase();

const getSatOptionValues = (problemText: string): Map<string, string> => {
  const values = new Map<string, string>();
  const optionRegex = /(?:^|[\n\r|])\s*([A-D])\s*[).:]\s*([\s\S]*?)(?=(?:[\n\r|]\s*[A-D]\s*[).:])|$)/g;
  for (const match of problemText.matchAll(optionRegex)) {
    const label = match[1].toUpperCase();
    const value = normalizeOptionValue(match[2] ?? "");
    if (value) values.set(label, value);
  }
  return values;
};

export function normalizeFinalAnswer(
  answer: string,
  context?: FinalAnswerContext,
): string {
  const raw = stripAnswerDecorations(answer);
  if (!raw || !isSatContext(context)) return raw;

  const problemText = context?.problemText ?? "";
  const hasOptions = isLikelySatMultipleChoice(problemText);
  const choiceLetter = raw.match(/^(?:option|choice|answer)?\s*([A-D])\s*[).:]?$/i);
  if (choiceLetter) return choiceLetter[1].toUpperCase();

  const choiceIndex = raw.match(/^(?:(?:option|choice|answer)\s*)?([1-4])\s*[).:]?$/i);
  const explicitlyChoice = /option|choice|answer/i.test(raw);
  if (choiceIndex && (hasOptions || explicitlyChoice)) {
    return SAT_INDEX_TO_LETTER[choiceIndex[1]] ?? raw;
  }

  if (hasOptions) {
    const normalizedRaw = normalizeOptionValue(raw);
    for (const [label, value] of getSatOptionValues(problemText)) {
      if (value && normalizedRaw === value) return label;
    }
  }

  return raw;
}

export function parseStructuredSolution(
  text: string,
  context?: FinalAnswerContext,
): ParsedSolution {
  const empty: ParsedSolution = { strategy: "", steps: [], finalAnswer: "" };
  if (!text.trim()) return empty;

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return empty;
  }

  if (!parsed || typeof parsed !== "object") return empty;
  const obj = parsed as Record<string, unknown>;

  const rawSteps = Array.isArray(obj.steps) ? obj.steps : [];
  const steps = rawSteps
    .filter((s): s is Record<string, unknown> => !!s && typeof s === "object")
    .map((s) => ({
      title: typeof s.title === "string" ? s.title : "",
      body: typeof s.content === "string" ? s.content : "",
    }));

  const finalAnswer =
    obj.final_answer !== undefined && obj.final_answer !== null
      ? normalizeFinalAnswer(String(obj.final_answer), context)
      : "";

  return { strategy: "", steps, finalAnswer };
}
