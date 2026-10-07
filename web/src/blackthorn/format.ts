/** Small pure formatting helpers (kept out of component files). */
export function formatDuration(totalSeconds: number): string {
  const s = Math.max(0, Math.round(totalSeconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rest = s % 60;
  return m < 60 ? `${m}m ${rest.toString().padStart(2, "0")}s` : `${Math.floor(m / 60)}h ${(m % 60).toString().padStart(2, "0")}m`;
}

const PRIMARY_KEYS = ["command", "query", "url", "path", "note"];

/** The one argument worth showing in a collapsed tool row. */
export function primaryArg(args: Record<string, unknown>): string {
  for (const k of PRIMARY_KEYS) if (typeof args[k] === "string" && args[k]) return String(args[k]);
  const first = Object.values(args).find((v) => typeof v === "string" && v);
  return first ? String(first) : "";
}
