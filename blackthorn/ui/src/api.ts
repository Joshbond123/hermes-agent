import type { ModelCatalogue } from './models'
import type { CodespaceStatus, CodespaceUsage, Message, Session, VersionInfo } from './types'

export class ApiError extends Error {
  status: number
  code: string
  constructor(status: number, code: string, message: string) {
    super(message)
    this.status = status
    this.code = code
  }
}

async function toError(resp: Response): Promise<ApiError> {
  let code = 'http_error'
  let message = `Request failed (HTTP ${resp.status})`
  try {
    const data = await resp.json()
    const detail = data?.detail
    if (detail && typeof detail === 'object') {
      code = detail.code || code
      message = detail.message || message
    } else if (typeof detail === 'string') {
      message = detail
    } else if (Array.isArray(detail) && detail[0]?.msg) {
      code = 'invalid'
      message = detail[0].msg
    }
  } catch {
    /* non-JSON error body */
  }
  return new ApiError(resp.status, code, message)
}

export async function request<T>(path: string, init: RequestInit & { timeoutMs?: number } = {}): Promise<T> {
  const { timeoutMs = 20000, ...rest } = init
  const headers = new Headers(rest.headers)
  if (rest.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  let resp: Response
  try {
    resp = await fetch(path, { ...rest, headers, signal: rest.signal ?? AbortSignal.timeout(timeoutMs) })
  } catch (err) {
    const timedOut = err instanceof DOMException && (err.name === 'TimeoutError' || err.name === 'AbortError')
    throw new ApiError(0, timedOut ? 'timeout' : 'network', timedOut ? 'The request timed out. Check your connection and try again.' : 'Could not reach the server. Check your connection and try again.')
  }
  if (!resp.ok) throw await toError(resp)
  return (await resp.json()) as T
}

const json = (body: unknown): RequestInit => ({ body: JSON.stringify(body) })

export const api = {
  auth: {
    status: () => request<{ authenticated: boolean; configured: boolean }>('/api/auth/status'),
    login: (password: string) => request<{ ok: boolean }>('/api/auth/login', { method: 'POST', ...json({ password }) }),
    reset: (current_password: string, new_password: string) => request<{ ok: boolean }>('/api/auth/reset', { method: 'POST', ...json({ current_password, new_password }) }),
    logout: () => request<{ ok: boolean }>('/api/auth/logout', { method: 'POST' }),
  },
  sessions: {
    list: (archived = false, q = '') =>
      request<{ sessions: Session[] }>(`/api/chat/sessions?archived=${archived ? 1 : 0}${q ? `&q=${encodeURIComponent(q)}` : ''}`).then((r) => r.sessions),
    get: (id: string) =>
      request<{ session: Session; messages: Message[]; live_run: { run_id: string; last_seq: number } | null }>(`/api/chat/sessions/${encodeURIComponent(id)}`),
    patch: (id: string, body: { title?: string; pinned?: boolean; archived?: boolean; model?: string }) =>
      request<{ session: Session }>(`/api/chat/sessions/${encodeURIComponent(id)}`, { method: 'PATCH', ...json(body) }).then((r) => r.session),
    remove: (id: string) => request<{ ok: boolean }>(`/api/chat/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  },
  codespaces: {
    status: (refresh = false) => request<CodespaceStatus>(`/api/codespaces/status${refresh ? '?refresh=true' : ''}`, { timeoutMs: 30000 }),
    usage: () => request<CodespaceUsage>('/api/codespaces/usage', { timeoutMs: 30000 }),
    start: () => request<{ ok: boolean }>('/api/codespaces/start', { method: 'POST', timeoutMs: 240000 }),
    stop: () => request<{ ok: boolean; stopped?: boolean; reason?: string; detail?: string }>('/api/codespaces/stop', { method: 'POST', timeoutMs: 180000 }),
    restart: () => request<{ ok: boolean }>('/api/codespaces/restart', { method: 'POST', timeoutMs: 420000 }),
  },
  models: {
    list: (sessionId?: string | null) =>
      request<ModelCatalogue>(`/api/models${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ''}`),
    /** Switch the model of an existing conversation. The conversation itself is untouched. */
    choose: (sessionId: string, model: string) =>
      request<{ session: Session; model: string; applies_to: string; active_run?: { id: string; model: string; note: string } }>(
        `/api/chat/sessions/${encodeURIComponent(sessionId)}`, { method: 'PATCH', ...json({ model }) }),
  },
  gpu: {
    status: () => request<import('./gpu').GpuStatus>('/api/gpu/status'),
    usage: () => request<import('./gpu').GpuUsage>('/api/gpu/usage'),
    start: () => request<{ state: string }>('/api/gpu/start', { method: 'POST', ...json({}) }),
    /** The server may refuse a stop while work is running — that refusal is the guarantee. */
    stop: (force = false) => request<{ state: string }>(`/api/gpu/stop?force=${force ? 'true' : 'false'}`, { method: 'POST', ...json({}) }),
    install: () => request<{ ok: boolean; kernel: string }>('/api/gpu/install', { method: 'POST', ...json({}) }),
    setSettings: (body: { idle_minutes?: number; weekly_hours?: number }) =>
      request<{ settings: unknown }>('/api/gpu/settings', { method: 'PATCH', ...json(body) }),
  },
  prompt: {
    get: () => request<{ prompt: string }>('/api/system-prompt').then((r) => r.prompt),
    put: (prompt: string) => request<{ ok: boolean }>('/api/system-prompt', { method: 'PUT', ...json({ prompt }) }),
  },
  version: () => request<VersionInfo>('/api/version'),
}
