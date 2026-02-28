import type { LLMStreamClient } from "./llmStreamClient";
import { backendStreamClient } from "./llmStreamClient.backend";

export const createLLMStreamClient = (): LLMStreamClient => {
  return backendStreamClient;
};
