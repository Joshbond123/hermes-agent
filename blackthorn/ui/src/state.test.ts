import { describe, expect, it } from 'vitest'
import { initialState, reducer, sortSessions, type AppState } from './state'
import type { ServerEvent, Session } from './types'

const sess = (id: string, extra: Partial<Session> = {}): Session => ({ id, title: id, pinned: false, archived: false, updated_at: 1, message_count: 2, ...extra })
const E = (e: Record<string, unknown>, seq = 1) => ({ seq, ...e }) as unknown as ServerEvent
const run = (s: AppState, e: ServerEvent) => reducer(s, { type: 'run/event', event: e })

function started(): AppState {
  let s = reducer(initialState(), { type: 'send/start', userId: 'tmp-u', assistantId: 'tmpa-a', text: 'hello' })
  s = run(s, E({ type: 'run.start', run_id: 'run_1', session_id: 'S1', assistant_id: 'A1', user_id: 'U1', model: 'm', created: true, session: sess('S1', { title: 'hello' }) }))
  return s
}

describe('streaming reducer', () => {
  it('swaps temporary ids for server ids and registers the session', () => {
    const s = started()
    expect(s.chat.sessionId).toBe('S1')
    expect(s.chat.messages.map((m) => m.id)).toEqual(['U1', 'A1'])
    expect(s.chat.run).toMatchObject({ runId: 'run_1', assistantId: 'A1', phase: 'waiting' })
    expect(s.sessions[0]).toMatchObject({ id: 'S1', title: 'hello' })
  })

  it('builds ONE continuous assistant message with ordered text and tool parts', () => {
    let s = started()
    s = run(s, E({ type: 'text.delta', text: 'Let me ' }, 2))
    s = run(s, E({ type: 'text.delta', text: 'check.' }, 3))
    s = run(s, E({ type: 'tool.start', id: 't1', name: 'run_command', summary: '$ ls', step: 1 }, 4))
    s = run(s, E({ type: 'tool.end', id: 't1', name: 'run_command', status: 'ok', summary: 'exit code 0', duration_ms: 120, output: 'a.txt' }, 5))
    s = run(s, E({ type: 'text.delta', text: 'Done.' }, 6))
    s = run(s, E({ type: 'run.end', status: 'stop', duration_ms: 900, message_id: 'A1', session_id: 'S1', steps: 2, tool_calls: 1 }, 7))
    const assistant = s.chat.messages.filter((m) => m.role === 'assistant')
    expect(assistant).toHaveLength(1)
    expect(assistant[0].parts.map((p) => p.type)).toEqual(['text', 'tool', 'text'])
    expect(assistant[0].parts[0]).toEqual({ type: 'text', text: 'Let me check.' })
    expect(assistant[0].parts[1]).toMatchObject({ args: '$ ls', summary: 'exit code 0', status: 'ok', duration_ms: 120 })
    expect(assistant[0].content).toBe('Let me check.\n\nDone.')
    expect(assistant[0].status).toBe('stop')
    expect(s.chat.run).toBeNull()
  })

  it('never shows reasoning: thinking events only toggle an indicator', () => {
    let s = started()
    s = run(s, E({ type: 'thinking', state: 'start' }, 2))
    expect(s.chat.messages[1].thinkingSince).toBeTypeOf('number')
    expect(s.chat.run?.phase).toBe('thinking')
    s = run(s, E({ type: 'text.delta', text: 'Hi' }, 3))
    expect(s.chat.messages[1].thinkingSince).toBeNull()
    expect(s.chat.messages[1].parts).toEqual([{ type: 'text', text: 'Hi' }])
  })

  it('marks running tools as cancelled/error when the run ends, and keeps the error', () => {
    let s = started()
    s = run(s, E({ type: 'tool.start', id: 't1', name: 'run_command', summary: '$ sleep 99' }, 2))
    s = run(s, E({ type: 'run.end', status: 'cancelled', duration_ms: 5, message_id: 'A1', session_id: 'S1', steps: 1, tool_calls: 1 }, 3))
    expect(s.chat.messages[1].status).toBe('cancelled')
    expect(s.chat.messages[1].parts[0]).toMatchObject({ status: 'cancelled' })
    let e = started()
    e = run(e, E({ type: 'error', code: 'gpu_dropped', message: 'dropped', retryable: false }, 2))
    e = run(e, E({ type: 'run.end', status: 'error', duration_ms: 5, message_id: 'A1', session_id: 'S1', steps: 1, tool_calls: 0 }, 3))
    expect(e.chat.messages[1]).toMatchObject({ status: 'error', error: { code: 'gpu_dropped' } })
  })

  it('send/failed removes both optimistic messages (draft is restored by the composer)', () => {
    let s = reducer(initialState(), { type: 'send/start', userId: 'tmp-u', assistantId: 'tmpa-a', text: 'x' })
    s = reducer(s, { type: 'send/failed', userId: 'tmp-u', assistantId: 'tmpa-a' })
    expect(s.chat.messages).toEqual([])
    expect(s.chat.run).toBeNull()
  })

  it('regenerate replaces the last assistant message', () => {
    let s = started()
    s = run(s, E({ type: 'text.delta', text: 'first' }, 2))
    s = run(s, E({ type: 'run.end', status: 'stop', duration_ms: 1, message_id: 'A1', session_id: 'S1', steps: 1, tool_calls: 0 }, 3))
    s = reducer(s, { type: 'regenerate/start', assistantId: 'tmpa-b' })
    expect(s.chat.messages.map((m) => m.role)).toEqual(['user', 'assistant'])
    expect(s.chat.messages[1]).toMatchObject({ id: 'tmpa-b', parts: [], status: 'streaming' })
  })

  it('re-attaching replays the whole run into a fresh placeholder (refresh mid-stream)', () => {
    let s = reducer(initialState(), { type: 'chat/loaded', session: sess('S1'), messages: [
      { id: 'U1', role: 'user', parts: [{ type: 'text', text: 'q' }], content: 'q', status: '' },
      { id: 'A1', role: 'assistant', parts: [{ type: 'text', text: 'partial saved' }], content: 'partial saved', status: 'streaming' },
    ] })
    s = reducer(s, { type: 'attach/start', assistantId: 'tmpa-x', runId: 'run_1' })
    expect(s.chat.messages).toHaveLength(2)
    s = run(s, E({ type: 'run.start', run_id: 'run_1', session_id: 'S1', assistant_id: 'A1', user_id: 'U1', model: 'm', created: false, session: sess('S1') }, 1))
    s = run(s, E({ type: 'text.delta', text: 'full ' }, 2))
    s = run(s, E({ type: 'text.delta', text: 'answer' }, 3))
    s = run(s, E({ type: 'run.end', status: 'stop', duration_ms: 1, message_id: 'A1', session_id: 'S1', steps: 1, tool_calls: 0 }, 4))
    expect(s.chat.messages).toHaveLength(2)
    expect(s.chat.messages[1]).toMatchObject({ id: 'A1', content: 'full answer', status: 'stop' })
  })

  it('run/lost keeps the partial text and marks the message interrupted', () => {
    let s = started()
    s = run(s, E({ type: 'text.delta', text: 'partial' }, 2))
    s = reducer(s, { type: 'run/lost', message: 'connection lost' })
    expect(s.chat.messages[1]).toMatchObject({ status: 'interrupted', error: { code: 'connection_lost' } })
    expect(s.chat.messages[1].parts).toEqual([{ type: 'text', text: 'partial' }])
    expect(s.chat.run).toBeNull()
  })
})

describe('history list reducer', () => {
  it('pinned first, then most recent', () => {
    const list = sortSessions([sess('a', { updated_at: 5 }), sess('b', { updated_at: 9 }), sess('c', { updated_at: 1, pinned: true })])
    expect(list.map((s) => s.id)).toEqual(['c', 'b', 'a'])
  })
  it('archive moves a session between lists; pin reorders; delete clears the open chat', () => {
    let s = reducer(initialState(), { type: 'sessions/loaded', sessions: [sess('a', { updated_at: 2 }), sess('b', { updated_at: 1 })] })
    s = reducer(s, { type: 'archived/loaded', sessions: [] })
    s = reducer(s, { type: 'session/patched', session: sess('b', { pinned: true, updated_at: 1 }) })
    expect(s.sessions.map((x) => x.id)).toEqual(['b', 'a'])
    s = reducer(s, { type: 'session/patched', session: sess('a', { archived: true }) })
    expect(s.sessions.map((x) => x.id)).toEqual(['b'])
    expect(s.archived?.map((x) => x.id)).toEqual(['a'])
    s = reducer(s, { type: 'session/patched', session: sess('a', { archived: false }) })
    expect(s.archived).toEqual([])
    s = reducer(s, { type: 'chat/loaded', session: sess('a'), messages: [] })
    s = reducer(s, { type: 'session/removed', id: 'a' })
    expect(s.chat.sessionId).toBeNull()
    expect(s.sessions.map((x) => x.id)).toEqual(['b'])
  })
})
