type ApiEnv = Record<string, string | boolean | undefined>;

const normalizeBase = (value: string | undefined): string =>
  (value ?? "").trim().replace(/\/+$/, "");

const isLegacyAclBase = (value: string): boolean =>
  value.startsWith("/csat_") && value.endsWith("2026demo");

export function getApiBase(...keys: string[]): string {
  const env = import.meta.env as ApiEnv;
  const configured = normalizeBase(
    keys
      .map((key) => env[key])
      .find((value): value is string => typeof value === "string" && value.trim().length > 0),
  );
  const viteBase = normalizeBase(typeof env.BASE_URL === "string" ? env.BASE_URL : "/");

  if (configured && !(isLegacyAclBase(configured) && viteBase === "/mathagora")) {
    return configured;
  }
  if (env.DEV) return "http://localhost:8000";
  return viteBase && viteBase !== "/" ? viteBase : "";
}
