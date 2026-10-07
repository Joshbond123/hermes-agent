import { afterEach, describe, expect, it, vi } from 'vitest'
import { followRun, parseSse } from './sse'
import type { ServerEvent } from './types'

const ev = (seq: number, extra: Record<string, unknown> = {}) => ({ seq, type: 'text.delta', text: `t${seq}`, ...extra }) as unknown as ServerEvent
const frame = (e: ServerEvent) => `id: ${e.seq}\ndata: ${JSON.stringify(e)}\n\n`

describe('parseSse', () => {
  it('parses complete events and keeps an incomplete tail', () => {
    const text = frame(ev(1)) + frame(ev(2)) + 'data: {"seq":3,"ty'
    const { events, rest } = parseSse(text)
    expect(events.map((e) => e.seq)).toEqual([1, 2])
    expect(rest).toBe('data: {"seq":3,"ty')
  })
  it('is safe at every chunk boundary (no loss, no duplicates)', () => {
    const full = [1, 2, 3, 4].map((n) => frame(ev(n))).join('')
    for (let cut = 0; cut <= full.length; cut++) {
      let buf = ''
      const seen: number[] = []
      for (const piece of [full.slice(0, cut), full.slice(cut)]) {
        buf += piece
        const parsed = parseSse(buf)
        buf = parsed.rest
        parsed.events.forEach((e) => seen.push(e.seq))
      }
      expect(seen).toEqual([1, 2, 3, 4])
    }
  })
  it('ignores heartbeats, retry hints and malformed JSON; handles CRLF and unicode', () => {
    const text = ': ping\n\nretry: 3000\n\ndata: not json\n\n' + frame(ev(5, { text: 'héllo 👋' })).replace(/\n/g, '\r\n')
    const { events } = parseSse(text + '\r\n\r\n')
    expect(events).toHaveLength(1)
    expect((events[0] as { text: string }).text).toBe('héllo 👋')
  })
})

function sseResponse(frames: string[], opts: { breakAfter?: boolean } = {}): Response {
  const enc = new TextEncoder()
  let i = 0
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i < frames.length) controller.enqueue(enc.encode(frames[i++]))
      else if (opts.breakAfter) controller.error(new TypeError('network error'))
      else controller.close()
    },
  })
  return new Response(stream, { status: 200, headers: { 'content-type': 'text/event-stream' } })
}

describe('followRun resume', () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers() })

  it('delivers each event exactly once and resumes from the last seq after a dropped connection', async () => {
    const start = { seq: 1, type: 'run.start', run_id: 'run_x', session_id: 's', assistant_id: 'a', model: 'm', created: true, session: {} } as unknown as ServerEvent
    const end = { seq: 4, type: 'run.end', status: 'stop', duration_ms: 1, message_id: 'a', session_id: 's', steps: 1, tool_calls: 0 } as unknown as ServerEvent
    const urls: string[] = []
    // the resumed stream (wrongly) replays seq 2 as well: it must be de-duplicated
    vi.stubGlobal('fetch', vi.fn(async (url: string) => { urls.push(url); return sseResponse([frame(ev(2)), frame(ev(3)), frame(end)]) }))
    const got: number[] = []
    const handle = followRun(async () => sseResponse([frame(start), frame(ev(2))], { breakAfter: true }), (e) => got.push(e.seq))
    expect(await handle.done).toBe('done')
    expect(got).toEqual([1, 2, 3, 4])
    expect(urls).toEqual(['/api/chat/runs/run_x/events?after=2'])
  })

  it('reports a clean "gone" when the server no longer has the run', async () => {
    const start = { seq: 1, type: 'run.start', run_id: 'run_y', session_id: 's', assistant_id: 'a', model: 'm', created: true, session: {} } as unknown as ServerEvent
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 404 })))
    const handle = followRun(async () => sseResponse([frame(start)], { breakAfter: true }), () => undefined)
    expect(await handle.done).toBe('gone')
  })

  it('detach() stops following without calling the cancel endpoint', async () => {
    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const start = { seq: 1, type: 'run.start', run_id: 'run_z', session_id: 's', assistant_id: 'a', model: 'm', created: true, session: {} } as unknown as ServerEvent
    // like a real fetch body: it errors with AbortError when the request signal aborts
    const open = async (signal: AbortSignal) =>
      new Response(new ReadableStream({ start(c) { c.enqueue(new TextEncoder().encode(frame(start))); signal.addEventListener('abort', () => c.error(new DOMException('Aborted', 'AbortError'))) } }), { status: 200 })
    const handle = followRun(open, () => undefined)
    await new Promise((r) => setTimeout(r, 20))
    handle.detach()
    expect(await handle.done).toBe('aborted')
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
