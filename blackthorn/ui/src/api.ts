import type { GpuStatus, Message, Session, VersionInfo } from './types'

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
  sessions: {
    list: (archived = false, q = '') =>
      request<{ sessions: Session[] }>(`/api/chat/sessions?archived=${archived ? 1 : 0}${q ? `&q=${encodeURIComponent(q)}` : ''}`).then((r) => r.sessions),
    get: (id: string) =>
      request<{ session: Session; messages: Message[]; live_run: { run_id: string; last_seq: number } | null }>(`/api/chat/sessions/${encodeURIComponent(id)}`),
    patch: (id: string, body: { title?: string; pinned?: boolean; archived?: boolean }) =>
      request<{ session: Session }>(`/api/chat/sessions/${encodeURIComponent(id)}`, { method: 'PATCH', ...json(body) }).then((r) => r.session),
    remove: (id: string) => request<{ ok: boolean }>(`/api/chat/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  },
  gpu: {
    status: (refresh = false) => request<GpuStatus>(`/api/kaggle-gpu/status${refresh ? '?refresh=true' : ''}`, { timeoutMs: 30000 }),
    turnOn: () => request<GpuStatus>('/api/kaggle-gpu/turn-on', { method: 'POST', timeoutMs: 60000 }),
    turnOff: () => request<GpuStatus>('/api/kaggle-gpu/turn-off', { method: 'POST', ...json({ confirm: true }), timeoutMs: 180000 }),
    activity: () => request<{ auto_off_minutes: number; choices: number[] }>('/api/kaggle-gpu/activity'),
    setAutoOff: (minutes: number) => request<{ auto_off_minutes: number }>('/api/kaggle-gpu/auto-off', { method: 'POST', ...json({ minutes }) }),
    logs: () => request<{ lines: string[]; error?: string }>('/api/kaggle-gpu/logs?limit=80', { timeoutMs: 25000 }),
  },
  prompt: {
    get: () => request<{ prompt: string }>('/api/system-prompt').then((r) => r.prompt),
    put: (prompt: string) => request<{ ok: boolean }>('/api/system-prompt', { method: 'PUT', ...json({ prompt }) }),
  },
  version: () => request<VersionInfo>('/api/version'),
}
