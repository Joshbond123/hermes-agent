/** Server-sent-event plumbing: a pure incremental parser, a reader, and a run driver that resumes dropped streams. */
import { ApiError, request } from './api'
import type { ServerEvent } from './types'

/** Split a text buffer into complete SSE events; returns the unconsumed tail. Pure and chunk-boundary safe. */
export function parseSse(buffer: string): { events: ServerEvent[]; rest: string } {
  const text = buffer.replace(/\r\n/g, '\n')
  const blocks = text.split('\n\n')
  const rest = blocks.pop() ?? ''
  const events: ServerEvent[] = []
  for (const block of blocks) {
    const data: string[] = []
    for (const line of block.split('\n')) {
      if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''))
    }
    if (!data.length) continue // comment / retry / heartbeat
    try {
      events.push(JSON.parse(data.join('\n')) as ServerEvent)
    } catch {
      /* ignore a malformed block rather than killing the stream */
    }
  }
  return { events, rest }
}

export async function readSse(resp: Response, onEvent: (ev: ServerEvent) => void): Promise<void> {
  if (!resp.body) throw new Error('streaming is not supported by this browser')
  const reader = resp.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parsed = parseSse(buffer)
    buffer = parsed.rest
    for (const ev of parsed.events) onEvent(ev)
  }
  buffer += decoder.decode()
  for (const ev of parseSse(buffer + '\n\n').events) onEvent(ev)
}

export type Outcome = 'done' | 'aborted' | 'gone' | 'lost'

export interface RunHandle {
  /** Resolves when the run ended, was aborted, or could not be followed any more. */
  done: Promise<Outcome>
  /** Ask the server to stop generating. The final run.end event still arrives on the open stream. */
  stop(): Promise<void>
  /** Stop *following* the run (e.g. the user opened another chat). The run keeps generating server-side and is saved. */
  detach(): void
  runId(): string | null
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

/**
 * Drive one run: open the stream (POST for a new run, GET to re-attach), forward events exactly once
 * (deduplicated by seq), and when the connection drops before run.end resume from the last seq seen.
 */
export function followRun(open: (signal: AbortSignal, after: number) => Promise<Response>, onEvent: (ev: ServerEvent) => void, opts: { startAfter?: number; runId?: string } = {}): RunHandle {
  const abort = new AbortController()
  let lastSeq = opts.startAfter ?? 0
  let runId: string | null = opts.runId ?? null
  let ended = false

  const done = (async (): Promise<Outcome> => {
    let resp = await open(abort.signal, lastSeq)
    for (;;) {
      try {
        await readSse(resp, (ev) => {
          if (ev.seq <= lastSeq) return
          lastSeq = ev.seq
          if (ev.type === 'run.start') runId = ev.run_id
          if (ev.type === 'run.end') ended = true
          onEvent(ev)
        })
      } catch {
        if (abort.signal.aborted) return 'aborted'
      }
      if (ended) return 'done'
      if (abort.signal.aborted) return 'aborted'
      if (!runId) return 'lost'
      // dropped before run.end: re-attach and receive exactly the events we missed
      let attached = false
      for (let attempt = 0; attempt < 6 && !attached; attempt++) {
        await sleep(Math.min(4000, 400 * 2 ** attempt))
        if (abort.signal.aborted) return 'aborted'
        try {
          const r = await fetch(`/api/chat/runs/${encodeURIComponent(runId)}/events?after=${lastSeq}`, { signal: abort.signal })
          if (r.status === 404) return 'gone'
          if (r.ok) {
            resp = r
            attached = true
          }
        } catch {
          if (abort.signal.aborted) return 'aborted'
        }
      }
      if (!attached) return 'lost'
    }
  })()

  return {
    done,
    runId: () => runId,
    detach: () => abort.abort(),
    async stop() {
      const id = runId
      if (id) {
        try {
          await request(`/api/chat/runs/${encodeURIComponent(id)}/cancel`, { method: 'POST', timeoutMs: 12000 })
        } catch (err) {
          if (!(err instanceof ApiError && err.status === 404)) abort.abort()
        }
      } else {
        abort.abort()
      }
      // safety net: if the stream does not close on its own, stop following it
      setTimeout(() => {
        if (!ended) abort.abort()
      }, 6000)
    },
  }
}

export function startRun(body: Record<string, unknown>, onEvent: (ev: ServerEvent) => void): RunHandle {
  return followRun(
    async (signal) => {
      const resp = await fetch('/api/chat/stream', { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' }, body: JSON.stringify(body), signal })
      if (!resp.ok) {
        let code = 'http_error'
        let message = `The server returned HTTP ${resp.status}.`
        try {
          const data = await resp.json()
          code = data?.detail?.code || code
          message = data?.detail?.message || message
        } catch {
          /* ignore */
        }
        throw new ApiError(resp.status, code, message)
      }
      return resp
    },
    onEvent,
  )
}

export function attachRun(runId: string, onEvent: (ev: ServerEvent) => void): RunHandle {
  return followRun(
    async (signal, after) => {
      const resp = await fetch(`/api/chat/runs/${encodeURIComponent(runId)}/events?after=${after}`, { signal })
      if (!resp.ok) throw new ApiError(resp.status, resp.status === 404 ? 'run_gone' : 'http_error', 'Could not re-attach to the running response.')
      return resp
    },
    onEvent,
    { runId },
  )
}
