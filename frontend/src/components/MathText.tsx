import katex from "katex";

type Token =
  | { type: "text"; value: string }
  | { type: "math"; value: string; display: boolean };

const splitMathTokens = (text: string): Token[] => {
  const tokens: Token[] = [];
  // Order matters: $$ before $, \[ before \( — so longer delimiters are matched first.
  const pattern = /\\\[([\s\S]+?)\\\]|\$\$([\s\S]+?)\$\$|\\\(([\s\S]+?)\\\)|\$([\s\S]+?)\$/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null = null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      tokens.push({ type: "text", value: text.slice(lastIndex, match.index) });
    }
    // Groups 1 and 2 are display math (\[...\] and $$...$$), groups 3 and 4 are inline.
    const display = match[1] != null || match[2] != null;
    const mathValue = match[1] ?? match[2] ?? match[3] ?? match[4] ?? "";
    tokens.push({ type: "math", value: mathValue, display });
    lastIndex = pattern.lastIndex;
  }

  if (lastIndex < text.length) {
    tokens.push({ type: "text", value: text.slice(lastIndex) });
  }

  return tokens;
};

const escapeHtml = (value: string) =>
  value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

const renderMath = (math: string, display: boolean): string => {
  try {
    return katex.renderToString(math.trim(), {
      throwOnError: false,
      strict: "ignore",
      output: "html",
      displayMode: display,
    });
  } catch {
    return escapeHtml(math);
  }
};

export const MathText = ({ text }: { text: string }) => {
  const parts = splitMathTokens(text);

  return (
    <span className="math-text">
      {parts.map((part, idx) =>
        part.type === "math" ? (
          <span
            key={idx}
            className={part.display ? "math-display" : "math-inline"}
            dangerouslySetInnerHTML={{ __html: renderMath(part.value, part.display) }}
          />
        ) : (
          <span key={idx}>{part.value}</span>
        ),
      )}
    </span>
  );
};
