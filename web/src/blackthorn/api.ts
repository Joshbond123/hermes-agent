/**
 * REST + streaming client for the Blackthorn API (blackthorn/api.py).
 *
 * `runTurn` is the heart of live chat: it streams one assistant turn over SSE and, when the
 * connection drops mid-answer, transparently re-attaches to the *same* turn on the server
 * (which keeps running independently of the browser) asking only for events after the last
 * sequence number it saw — so nothing is duplicated and nothing is lost.
 */
import { api as stockApi, HERMES_BASE_PATH } from "@/lib/api";
import { SseParser } from "./sse";
import type {
  ApiMessage,
  GpuSnapshot,
  SessionSummary,
  StreamEvent,
  VersionInfo,
} from "./types";

const SESSION_HEADER = "X-Hermes-Session-Token";

function authHeaders(): Record<string, string> {
  const token = typeof window === "undefined" ? undefined : window.__HERMES_SESSION_TOKEN__;
  return token ? { [SESSION_HEADER]: token } : {};
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function errorFrom(res: Response): Promise<ApiError> {
  let detail = res.statusText || `HTTP ${res.status}`;
  try {
    const body = await res.json();
    const d = body?.detail ?? body;
    detail = typeof d === "string" ? d : JSON.stringify(d);
  } catch {
    /* non-JSON error body */
  }
  return new ApiError(res.status, detail);
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const headers: Record<string, string> = { ...authHeaders() };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(HERMES_BASE_PATH + path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
    credentials: "same-origin",
  });
  if (!res.ok) throw await errorFrom(res);
  return (await res.json()) as T;
}

export interface SessionPatch {
  title?: string;
  pinned?: boolean;
  archived?: boolean;
}

export interface StartBody {
  session_id?: string | null;
  message?: string;
  regenerate?: boolean;
  thinking?: boolean;
  attachments?: Array<{ path: string; name?: string }>;
}

export const btApi = {
  listSessions: (opts: { archived?: boolean; q?: string; limit?: number } = {}, signal?: AbortSignal) => {
    const p = new URLSearchParams();
    if (opts.archived) p.set("archived", "true");
    if (opts.q) p.set("q", opts.q);
    if (opts.limit) p.set("limit", String(opts.limit));
    const qs = p.toString();
    return request<{ sessions: SessionSummary[] }>("GET", `/api/studio/sessions${qs ? `?${qs}` : ""}`, undefined, signal);
  },
  getSession: (id: string, signal?: AbortSignal) =>
    request<{
      session: SessionSummary;
      active_turn: { turn_id: string; assistant_message_id: number | null; last_seq: number } | null;
    }>("GET", `/api/studio/sessions/${encodeURIComponent(id)}`, undefined, signal),
  messages: (id: string, signal?: AbortSignal) =>
    request<{ messages: ApiMessage[] }>("GET", `/api/studio/sessions/${encodeURIComponent(id)}/messages`, undefined, signal),
  patchSession: (id: string, patch: SessionPatch) =>
    request<{ session: SessionSummary }>("PATCH", `/api/studio/sessions/${encodeURIComponent(id)}`, patch),
  deleteSession: (id: string) =>
    request<{ ok: boolean }>("DELETE", `/api/studio/sessions/${encodeURIComponent(id)}`),
  cancelTurn: (turnId: string) =>
    request<{ ok: boolean; cancelled: boolean }>("POST", `/api/studio/turns/${encodeURIComponent(turnId)}/cancel`),

  getSystemPrompt: () =>
    request<{ prompt: string; default: string; max_chars: number }>("GET", "/api/system-prompt"),
  putSystemPrompt: (prompt: string) =>
    request<{ ok: boolean; prompt: string }>("PUT", "/api/system-prompt", { prompt }),

  gpuStatus: (signal?: AbortSignal) => request<GpuSnapshot>("GET", "/api/kaggle-gpu/status", undefined, signal),
  gpuTurnOn: () => request<GpuSnapshot & { ok: boolean; note?: string }>("POST", "/api/kaggle-gpu/turn-on"),
  gpuTurnOff: () => request<GpuSnapshot & { ok: boolean; note?: string }>("POST", "/api/kaggle-gpu/turn-off"),
  gpuRestart: () => request<GpuSnapshot & { ok: boolean; note?: string }>("POST", "/api/kaggle-gpu/restart"),
  gpuVerify: () => request<{ result: GpuSnapshot["last_inference"]; state: string }>("POST", "/api/kaggle-gpu/verify"),
  gpuAutoOff: (minutes: number) =>
    request<{ ok: boolean; gpu: GpuSnapshot }>("POST", "/api/kaggle-gpu/auto-off", { minutes }),
  gpuLogs: (limit = 120) => request<{ lines: string[]; note?: string }>("GET", `/api/kaggle-gpu/logs?limit=${limit}`),

  version: () => request<VersionInfo>("GET", "/api/blackthorn/version"),

  /** Upload to the managed files area and return the path the chat should reference. */
  uploadAttachment: async (file: File): Promise<{ path: string; name: string }> => {
    const safe = file.name.replace(/[^A-Za-z0-9._-]+/g, "_").slice(-80) || "file";
    const target = `uploads/${Date.now().toString(36)}-${safe}`;
    const res = await stockApi.uploadFile(target, file, true);
    return { path: res.path || target, name: file.name };
  },
};

// --------------------------------------------------------------------------- //
// streaming
// --------------------------------------------------------------------------- //
export interface RunTurnOptions {
  /** Start a new turn... */
  start?: StartBody;
  /** ...or attach to one that is already running / recently finished. */
  attach?: { turnId: string; after: number };
  signal: AbortSignal;
  /** Called with each batch of *new* events (already de-duplicated and in order). */
  onEvents: (events: StreamEvent[]) => void;
  onReconnecting?: (attempt: number) => void;
  maxReconnects?: number;
  /** Test hook: replace fetch. */
  fetchImpl?: typeof fetch;
  backoffMs?: number[];
}

export interface RunTurnResult {
  done: boolean;
  aborted: boolean;
  turnId: string | null;
  lastSeq: number;
  error: Error | null;
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal.aborted) return resolve();
    const t = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(t);
      resolve();
    };
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

export async function runTurn(opts: RunTurnOptions): Promise<RunTurnResult> {
  const doFetch = opts.fetchImpl ?? fetch;
  const backoff = opts.backoffMs ?? [400, 1000, 2000, 4000, 8000, 8000];
  const maxReconnects = opts.maxReconnects ?? backoff.length;
  let turnId: string | null = opts.attach?.turnId ?? null;
  let lastSeq = opts.attach?.after ?? 0;
  let done = false;
  let reconnects = 0;
  let first = !opts.attach;
  let lastError: Error | null = null;

  while (!done && !opts.signal.aborted) {
    try {
      let res: Response;
      if (first) {
        res = await doFetch(HERMES_BASE_PATH + "/api/studio/agent/stream", {
          method: "POST",
          headers: { ...authHeaders(), "Content-Type": "application/json", Accept: "text/event-stream" },
          body: JSON.stringify(opts.start ?? {}),
          signal: opts.signal,
          credentials: "same-origin",
        });
      } else {
        res = await doFetch(
          `${HERMES_BASE_PATH}/api/studio/turns/${encodeURIComponent(turnId ?? "")}/stream?after=${lastSeq}`,
          { headers: { ...authHeaders(), Accept: "text/event-stream" }, signal: opts.signal, credentials: "same-origin" },
        );
      }
      if (!res.ok) {
        const err = await errorFrom(res);
        // 4xx = our request is wrong (or the turn is gone): retrying cannot help.
        if (res.status < 500) return { done: false, aborted: false, turnId, lastSeq, error: err };
        throw err;
      }
      first = false;
      if (!res.body) throw new Error("the browser did not expose a response stream");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      const parser = new SseParser();
      let progressed = false;
      for (;;) {
        const { value, done: eof } = await reader.read();
        if (eof) break;
        const batch: StreamEvent[] = [];
        for (const frame of parser.push(decoder.decode(value, { stream: true }))) {
          let data: Record<string, any>;
          try {
            data = JSON.parse(frame.data);
          } catch {
            continue; // malformed event: skip it, keep the stream alive
          }
          const seq = frame.id ?? lastSeq + 1;
          if (seq <= lastSeq) continue; // already delivered before a reconnect
          lastSeq = seq;
          if (frame.event === "turn" && typeof data.turn_id === "string") turnId = data.turn_id;
          if (frame.event === "done") done = true;
          batch.push({ seq, type: frame.event, data });
        }
        if (batch.length) {
          progressed = true;
          opts.onEvents(batch);
        }
        if (done) break;
      }
      // Only an attempt that actually delivered something earns a fresh reconnect budget;
      // a server that keeps returning empty streams must not be retried forever.
      if (progressed) reconnects = 0;
      lastError = null;
      if (done) break;
      // The stream ended without `done`: the connection dropped. The turn itself is still alive.
    } catch (err) {
      if (opts.signal.aborted) break;
      lastError = err instanceof Error ? err : new Error(String(err));
      if (lastError instanceof ApiError && lastError.status < 500) {
        return { done: false, aborted: false, turnId, lastSeq, error: lastError };
      }
    }
    if (done || opts.signal.aborted) break;
    if (!turnId) {
      return {
        done: false,
        aborted: false,
        turnId,
        lastSeq,
        error: lastError ?? new Error("The connection was lost before the request was accepted."),
      };
    }
    if (reconnects >= maxReconnects) {
      return { done: false, aborted: false, turnId, lastSeq, error: lastError ?? new Error("The connection kept dropping.") };
    }
    reconnects += 1;
    opts.onReconnecting?.(reconnects);
    await sleep(backoff[Math.min(reconnects - 1, backoff.length - 1)], opts.signal);
  }
  return { done, aborted: opts.signal.aborted && !done, turnId, lastSeq, error: done ? null : lastError };
}
