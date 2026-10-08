import type { AttachmentInfo, Message, Part, ServerEvent, Session, ToolPart } from './types'

export type Phase = 'connecting' | 'waiting' | 'thinking' | 'streaming' | 'tool'

export interface ActiveRun {
  runId: string | null
  assistantId: string
  phase: Phase
  stopping: boolean
}

export interface ChatState {
  sessionId: string | null
  title: string
  messages: Message[]
  loading: boolean
  loadError: string | null
  run: ActiveRun | null
}

export interface AppState {
  sessions: Session[]
  archived: Session[] | null
  sessionsLoaded: boolean
  sessionsError: string | null
  chat: ChatState
}

export const emptyChat = (): ChatState => ({ sessionId: null, title: '', messages: [], loading: false, loadError: null, run: null })

export const initialState = (): AppState => ({ sessions: [], archived: null, sessionsLoaded: false, sessionsError: null, chat: emptyChat() })

export type Action =
  | { type: 'sessions/loaded'; sessions: Session[] }
  | { type: 'sessions/error'; message: string }
  | { type: 'archived/loaded'; sessions: Session[] }
  | { type: 'session/patched'; session: Session }
  | { type: 'session/removed'; id: string }
  | { type: 'chat/new' }
  | { type: 'chat/loading'; id: string }
  | { type: 'chat/loadFailed'; message: string }
  | { type: 'chat/loaded'; session: Session; messages: Message[] }
  | { type: 'send/start'; userId: string; assistantId: string; text: string; attachments?: AttachmentInfo[] }
  | { type: 'send/failed'; userId: string; assistantId: string }
  | { type: 'regenerate/start'; assistantId: string }
  | { type: 'attach/start'; assistantId: string; runId: string }
  | { type: 'run/event'; event: ServerEvent }
  | { type: 'run/stopping' }
  | { type: 'run/lost'; message: string }

const nowSec = () => Date.now() / 1000

export function sortSessions(list: Session[]): Session[] {
  return [...list].sort((a, b) => Number(b.pinned) - Number(a.pinned) || (b.updated_at ?? 0) - (a.updated_at ?? 0))
}

function upsert(list: Session[], session: Session): Session[] {
  const rest = list.filter((s) => s.id !== session.id)
  return sortSessions([session, ...rest])
}

export function placeholderAssistant(id: string): Message {
  return { id, role: 'assistant', parts: [], content: '', status: 'streaming', created_at: nowSec() }
}

/** Apply one server event to the assistant message it belongs to. Pure. */
export function applyToMessage(msg: Message, ev: ServerEvent): Message {
  switch (ev.type) {
    case 'thinking':
      return { ...msg, thinkingSince: ev.state === 'start' ? Date.now() : null }
    case 'text.delta': {
      const parts = msg.parts.slice()
      const last = parts[parts.length - 1]
      if (last && last.type === 'text') parts[parts.length - 1] = { type: 'text', text: last.text + ev.text }
      else parts.push({ type: 'text', text: ev.text })
      return { ...msg, parts, thinkingSince: null }
    }
    case 'tool.start': {
      if (msg.parts.some((p) => p.type === 'tool' && p.id === ev.id)) return msg
      const part: ToolPart = { type: 'tool', id: ev.id, name: ev.name, status: 'running', args: ev.summary, summary: '', step: ev.step, startedAt: Date.now() }
      return { ...msg, parts: [...msg.parts, part], thinkingSince: null }
    }
    case 'tool.end': {
      const parts: Part[] = msg.parts.map((p) =>
        p.type === 'tool' && p.id === ev.id
          ? { ...p, status: ev.status, summary: ev.summary, duration_ms: ev.duration_ms, error: ev.error ?? undefined, output: ev.output ?? undefined, sources: ev.sources ?? undefined, answer: ev.answer ?? undefined, exit_code: ev.exit_code ?? undefined }
          : p,
      )
      return { ...msg, parts }
    }
    case 'notice':
      return { ...msg, notices: [...(msg.notices ?? []), { level: ev.level, text: ev.text }] }
    case 'error':
      return { ...msg, error: { code: ev.code, message: ev.message, retryable: ev.retryable } }
    case 'run.end': {
      const parts = msg.parts.map((p) => (p.type === 'tool' && p.status === 'running' ? ({ ...p, status: ev.status === 'cancelled' ? 'cancelled' : 'error' } as ToolPart) : p))
      const content = parts.filter((p): p is { type: 'text'; text: string } => p.type === 'text').map((p) => p.text.trim()).filter(Boolean).join('\n\n')
      return { ...msg, parts, content, status: ev.status, thinkingSince: null, id: ev.message_id || msg.id,
        meta: { ...(msg.meta ?? {}), duration_ms: ev.duration_ms, first_token_ms: ev.first_token_ms, steps: ev.steps, tool_calls: ev.tool_calls, usage: ev.usage, error: msg.error } }
    }
    default:
      return msg
  }
}

function phaseFor(ev: ServerEvent, current: ActiveRun): ActiveRun {
  switch (ev.type) {
    case 'run.start':
      return { ...current, runId: ev.run_id, assistantId: ev.assistant_id, phase: 'waiting' }
    case 'thinking':
      return { ...current, phase: ev.state === 'start' ? 'thinking' : current.phase }
    case 'text.delta':
      return { ...current, phase: 'streaming' }
    case 'tool.start':
      return { ...current, phase: 'tool' }
    case 'tool.end':
      return { ...current, phase: 'waiting' }
    default:
      return current
  }
}

function reduceEvent(state: AppState, ev: ServerEvent): AppState {
  const chat = state.chat
  if (ev.type === 'run.start') {
    const ids = new Map<string, string>()
    const run = chat.run
    // swap the temporary client ids for the server's ids (user message then assistant message)
    const messages = chat.messages.map((m, i) => {
      if (run && m.id === run.assistantId) return { ...m, id: ev.assistant_id }
      if (ev.user_id && m.role === 'user' && i === chat.messages.length - 2 && m.id.startsWith('tmp-')) {
        ids.set(m.id, ev.user_id)
        return { ...m, id: ev.user_id }
      }
      return m
    })
    const session = ev.session
    return {
      ...state,
      sessions: upsert(state.sessions, { ...session, updated_at: nowSec() }),
      archived: state.archived ? state.archived.filter((s) => s.id !== session.id) : state.archived,
      chat: { ...chat, sessionId: ev.session_id, title: session.title, messages, run: run ? phaseFor(ev, run) : run },
    }
  }
  const run = chat.run
  if (!run) return state
  const messages = chat.messages.map((m) => (m.id === run.assistantId ? applyToMessage(m, ev) : m))
  if (ev.type === 'run.end') {
    const sessions = state.sessions.map((s) => (s.id === ev.session_id ? { ...s, updated_at: nowSec(), message_count: Math.max(s.message_count, 2) } : s))
    return { ...state, sessions: sortSessions(sessions), chat: { ...chat, messages, run: null } }
  }
  return { ...state, chat: { ...chat, messages, run: phaseFor(ev, run) } }
}

export function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'sessions/loaded':
      return { ...state, sessions: sortSessions(action.sessions), sessionsLoaded: true, sessionsError: null }
    case 'sessions/error':
      return { ...state, sessionsLoaded: true, sessionsError: action.message }
    case 'archived/loaded':
      return { ...state, archived: action.sessions }
    case 'session/patched': {
      const s = action.session
      const sessions = s.archived ? state.sessions.filter((x) => x.id !== s.id) : upsert(state.sessions, s)
      const archived = state.archived ? (s.archived ? upsert(state.archived, s) : state.archived.filter((x) => x.id !== s.id)) : state.archived
      const chat = state.chat.sessionId === s.id ? { ...state.chat, title: s.title } : state.chat
      return { ...state, sessions, archived, chat }
    }
    case 'session/removed':
      return {
        ...state,
        sessions: state.sessions.filter((s) => s.id !== action.id),
        archived: state.archived ? state.archived.filter((s) => s.id !== action.id) : state.archived,
        chat: state.chat.sessionId === action.id ? emptyChat() : state.chat,
      }
    case 'chat/new':
      return { ...state, chat: emptyChat() }
    case 'chat/loading':
      return { ...state, chat: { ...emptyChat(), sessionId: action.id, loading: true } }
    case 'chat/loadFailed':
      return { ...state, chat: { ...state.chat, loading: false, loadError: action.message } }
    case 'chat/loaded':
      return { ...state, chat: { sessionId: action.session.id, title: action.session.title, messages: action.messages, loading: false, loadError: null, run: null } }
    case 'send/start': {
      const user: Message = { id: action.userId, role: 'user', parts: [{ type: 'text', text: action.text }], content: action.text, status: '', created_at: nowSec(), attachments: action.attachments }
      return {
        ...state,
        chat: { ...state.chat, messages: [...state.chat.messages, user, placeholderAssistant(action.assistantId)], run: { runId: null, assistantId: action.assistantId, phase: 'connecting', stopping: false } },
      }
    }
    case 'send/failed':
      return { ...state, chat: { ...state.chat, messages: state.chat.messages.filter((m) => m.id !== action.userId && m.id !== action.assistantId), run: null } }
    case 'regenerate/start': {
      const messages = [...state.chat.messages]
      if (messages.length && messages[messages.length - 1].role === 'assistant') messages.pop()
      messages.push(placeholderAssistant(action.assistantId))
      return { ...state, chat: { ...state.chat, messages, run: { runId: null, assistantId: action.assistantId, phase: 'connecting', stopping: false } } }
    }
    case 'attach/start': {
      const messages = state.chat.messages.filter((m) => !(m.role === 'assistant' && m.status === 'streaming'))
      messages.push(placeholderAssistant(action.assistantId))
      return { ...state, chat: { ...state.chat, messages, run: { runId: action.runId, assistantId: action.assistantId, phase: 'waiting', stopping: false } } }
    }
    case 'run/event':
      return reduceEvent(state, action.event)
    case 'run/stopping':
      return state.chat.run ? { ...state, chat: { ...state.chat, run: { ...state.chat.run, stopping: true } } } : state
    case 'run/lost': {
      const run = state.chat.run
      if (!run) return state
      const messages = state.chat.messages.map((m) =>
        m.id === run.assistantId ? { ...m, status: 'interrupted' as const, thinkingSince: null, error: m.error ?? { code: 'connection_lost', message: action.message, retryable: true } } : m,
      )
      return { ...state, chat: { ...state.chat, messages, run: null } }
    }
    default:
      return state
  }
}

export const isBusy = (state: AppState) => state.chat.run !== null
