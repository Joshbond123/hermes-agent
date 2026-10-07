/** Shared types for the Blackthorn chat. They mirror the wire format of blackthorn/api.py. */

export type Role = "user" | "assistant";
export type ToolStatus = "running" | "ok" | "error" | "cancelled";

export interface TextPart {
  t: "text";
  text: string;
}

export interface ToolPart {
  t: "tool";
  id: string;
  name: string;
  label: string;
  status: ToolStatus;
  args: Record<string, unknown>;
  output?: string;
  durationMs?: number;
  exitCode?: number | null;
  truncated?: boolean;
}

export type Part = TextPart | ToolPart;

export type FinishReason =
  | "stop"
  | "length"
  | "cancelled"
  | "error"
  | "incomplete"
  | "timeout"
  | "interrupted"
  | "running"
  | null;

export interface ErrorInfo {
  code: string;
  message: string;
  retryable?: boolean;
  action?: string | null;
}

export type MessageStatus = "complete" | "streaming" | "stopped" | "error";

export interface UiMessage {
  /** Stable React key (client generated for live messages, `m<id>` for stored ones). */
  key: string;
  /** Database id once known. */
  id?: number | null;
  role: Role;
  parts: Part[];
  status: MessageStatus;
  finishReason?: FinishReason;
  error?: ErrorInfo | null;
  thinkingMs?: number;
  durationMs?: number;
  completionTokens?: number;
  attachments?: string[];
  notices?: string[];
}

export interface SessionSummary {
  id: string;
  title: string;
  pinned: boolean;
  archived: boolean;
  created_at?: number | null;
  updated_at?: number | null;
  message_count: number;
  model: string;
}

/** A message as returned by GET /api/studio/sessions/{id}/messages */
export interface ApiMessage {
  id: number;
  role: Role;
  content: string;
  created_at?: number | null;
  finish_reason?: FinishReason;
  parts?: Array<Record<string, unknown>> | null;
  usage?: { completion_tokens?: number } | null;
  duration_ms?: number | null;
  thinking_ms?: number | null;
  error?: ErrorInfo | null;
  attachments?: string[] | null;
}

/** Decoded SSE payload. The wire format is JSON of server-chosen shape per event type; the reducer
 *  validates every field it reads. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type EventData = Record<string, any>;

export interface StreamEvent {
  seq: number;
  type: string;
  data: EventData;
}

export type GpuStateName = "unknown" | "off" | "starting" | "ready" | "stopping" | "error";

export interface GpuProgress {
  mode: "none" | "indeterminate" | "determinate";
  done?: number;
  total?: number;
  unit?: string;
  fraction?: number;
}

export interface GpuSnapshot {
  state: GpuStateName;
  message: string;
  stage?: string;
  stage_label?: string;
  progress?: GpuProgress;
  stage_elapsed_s?: number;
  silent_for_s?: number;
  elapsed_s?: number;
  stalled?: boolean;
  model?: string;
  model_loaded?: boolean | null;
  gpu?: string;
  tunnel?: boolean;
  error?: { code: string; message: string } | null;
  quota?: {
    used_hours: number;
    total_hours: number;
    remaining_hours: number;
    used_pct: number;
    refresh_time?: string;
  } | null;
  auto_off?: {
    minutes: number;
    enabled: boolean;
    idle_s: number;
    remaining_s: number | null;
    choices: number[];
  };
  last_inference?: {
    ok: boolean | null;
    latency_ms?: number;
    reply?: string;
    error?: string | null;
    at?: number;
  } | null;
  busy?: boolean;
  transitioning?: boolean;
  age_s?: number;
}

export interface VersionInfo {
  commit: string;
  branch: string;
  service: string;
  web_build?: { commit?: string; built_at?: string; web_source_hash?: string };
  started_at: number;
  uptime_s: number;
}
