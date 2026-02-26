export type ParsedSolution = {
  strategy: string;
  steps: { title: string; body: string }[];
  finalAnswer: string;
};

export function parseStructuredSolution(text: string): ParsedSolution {
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
      ? String(obj.final_answer)
      : "";

  return { strategy: "", steps, finalAnswer };
}
