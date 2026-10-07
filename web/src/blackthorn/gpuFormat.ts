import type { GpuSnapshot } from "./types";

export function gpuLabel(gpu: GpuSnapshot | null, unreachable: boolean): string {
  if (unreachable && !gpu) return "GPU ?";
  switch (gpu?.state) {
    case "ready": return "GPU ready";
    case "starting": {
      const f = gpu.progress?.mode === "determinate" ? gpu.progress.fraction : undefined;
      return f !== undefined ? `Starting ${Math.round(f * 100)}%` : "Starting…";
    }
    case "stopping": return "Stopping…";
    case "error": return "GPU error";
    case "off": return "GPU off";
    default: return "GPU …";
  }
}

export const gpuDotClass = (gpu: GpuSnapshot | null, unreachable: boolean) =>
  `bt-dot bt-dot-gpu-${unreachable && !gpu ? "unknown" : gpu?.state ?? "unknown"}`;
