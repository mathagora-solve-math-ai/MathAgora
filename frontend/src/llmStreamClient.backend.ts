import type { LLMStreamClient } from "./llmStreamClient";
import { getApiBase } from "./apiBase";

const SOLVE_API_BASE = getApiBase("VITE_SOLVE_API_URL", "VITE_DETECT_API_URL");

export const backendStreamClient: LLMStreamClient = {
  async startSolveStream(req, opts) {
    const { signal, onChunk } = opts;
    const base = SOLVE_API_BASE ? SOLVE_API_BASE.replace(/\/$/, "") : "";
    const res = await fetch(`${base}/api/solve/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
      signal,
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(err || `Stream failed: ${res.status}`);
    }
    if (!res.body) {
      throw new Error("Stream failed: empty response body");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.trim()) continue;
        const chunk = JSON.parse(line);
        onChunk(chunk);
      }
    }

    const lastLine = buffer.trim();
    if (lastLine) {
      const chunk = JSON.parse(lastLine);
      onChunk(chunk);
    }
  },
};
