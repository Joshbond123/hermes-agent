export type Status = 'streaming' | 'stop' | 'length' | 'cancelled' | 'error' | 'interrupted' | ''

export interface TextPart { type: 'text'; text: string }
export interface Source { title: string; url: string; domain: string; snippet?: string }
export interface ToolPart {
  type: 'tool'
  id: string
  name: string
  status: 'running' | 'ok' | 'error' | 'cancelled'
  /** what the tool was asked to do (redacted): command, query, path, url */
  args: string
  /** what it returned (redacted, short) */
  summary: string
  step?: number
  duration_ms?: number
  output?: string
  sources?: Source[]
  /** short human summary of what a web search found (not just URLs) */
  answer?: string
  error?: string
  exit_code?: number
  /** epoch ms when the run started this tool (live elapsed display) */
  startedAt?: number
}
export type Part = TextPart | ToolPart

export interface RunError { code: string; message: string; retryable?: boolean }
export interface Notice { level: 'info' | 'warn' | 'error'; text: string }
export interface MessageMeta {
  model?: string
  duration_ms?: number
  first_token_ms?: number | null
  steps?: number
  tool_calls?: number
  error?: RunError
  usage?: { prompt: number; completion: number }
}
export interface AttachmentInfo { name: string; chars: number; workspace_path?: string | null }

export interface Message {
  id: string
  seq?: number
  role: 'user' | 'assistant'
  parts: Part[]
  content: string
  status: Status
  created_at?: number
  meta?: MessageMeta
  attachments?: AttachmentInfo[]
  notices?: Notice[]
  thinkingSince?: number | null
  error?: RunError
}

export interface Session {
  id: string
  title: string
  pinned: boolean
  archived: boolean
  created_at?: number
  updated_at?: number
  message_count: number
}

export interface GpuQuota { used_hours: number; total_hours: number; remaining_hours: number; used_pct: number; refresh_time?: string }
export interface GpuStatus {
  online: boolean
  active: boolean
  booting: boolean
  status: string
  display_status?: string
  engine_state?: string
  model?: string
  model_loaded?: boolean | null
  gpu_info?: string
  worker_status?: string
  progress_step?: string
  progress_stage?: number
  progress_total_stages?: number
  progress_kind?: 'stage' | 'bytes'
  progress_bytes_done?: number
  progress_bytes_total?: number
  progress_label?: string
  progress_stalled?: boolean
  elapsed_seconds?: number
  quota?: GpuQuota
  quotas?: { model?: GpuQuota & { account?: string }; computer?: GpuQuota & { account?: string } }
  computer?: { status?: string; online?: boolean; booting?: boolean; display_status?: string; gpu_info?: string }
  computer_ready?: boolean
  error?: string
  can_turn_on: boolean
  can_turn_off: boolean
  busy?: boolean
  server_time?: number
}

export interface VersionInfo {
  version: string
  commit: string
  ui?: { hash?: string }
  integrity?: { drift?: string[]; checked?: number }
  overlay?: { active: boolean; disabled: boolean; stamp: string | null }
}

// ---- server-sent events ------------------------------------------------------------------------------------
interface Base { seq: number }
export type ServerEvent =
  | (Base & { type: 'run.start'; run_id: string; session_id: string; assistant_id: string; user_id?: string | null; model: string; session: Session; created: boolean })
  | (Base & { type: 'thinking'; state: 'start' | 'end'; ms?: number })
  | (Base & { type: 'text.delta'; text: string })
  | (Base & { type: 'tool.start'; id: string; name: string; summary: string; step?: number })
  | (Base & { type: 'tool.end'; id: string; name: string; status: 'ok' | 'error'; summary: string; duration_ms: number; error?: string | null; output?: string | null; sources?: Source[] | null; answer?: string | null; exit_code?: number | null; truncated?: boolean })
  | (Base & { type: 'notice'; level: 'info' | 'warn' | 'error'; text: string })
  | (Base & ({ type: 'error' } & RunError))
  | (Base & { type: 'run.end'; status: Status; finish_reason?: string | null; duration_ms: number; first_token_ms?: number | null; message_id: string; session_id: string; steps: number; tool_calls: number; usage?: { prompt: number; completion: number }; saved?: boolean })
