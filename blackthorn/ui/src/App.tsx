import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { api, ApiError } from './api'
import { Chat } from './components/Chat'
import { Composer, type AttachedFile } from './components/Composer'
import { Confirm } from './components/Dialog'
import { Header } from './components/Header'
import { Sidebar } from './components/Sidebar'
import { SettingsDialog } from './components/Settings'
import { Toasts, type Toast } from './components/Toasts'
import { summarize, useGpu } from './gpu'
import { attachRun, startRun, type Outcome, type RunHandle } from './sse'
import { initialState, reducer } from './state'
import type { ServerEvent, Session, VersionInfo } from './types'
import { formatElapsed, uid } from './util'

const NARROW = '(max-width: 899px)'
const routeId = () => /^\/c\/([^/]+)/.exec(window.location.pathname)?.[1] ?? null

function useNarrow() {
  const [narrow, setNarrow] = useState(() => window.matchMedia(NARROW).matches)
  useEffect(() => {
    const mq = window.matchMedia(NARROW)
    const on = () => setNarrow(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  return narrow
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, undefined, initialState)
  const stateRef = useRef(state)
  stateRef.current = state
  const narrow = useNarrow()
  const [sidebarPref, setSidebarPref] = useState(() => localStorage.getItem('bt.sidebar') !== '0')
  const [drawer, setDrawer] = useState(false)
  const sidebarOpen = narrow ? drawer : sidebarPref
  const [gpuOpen, setGpuOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [deleting, setDeleting] = useState<Session | null>(null)
  const [deleteBusy, setDeleteBusy] = useState(false)
  const [query, setQuery] = useState('')
  const [toasts, setToasts] = useState<Toast[]>([])
  const runRef = useRef<RunHandle | null>(null)
  const openToken = useRef(0)
  const [theme, setThemeState] = useState<'system' | 'light' | 'dark'>(() => (localStorage.getItem('bt.theme') as 'light' | 'dark' | 'system' | null) ?? 'dark')
  const [version, setVersion] = useState<VersionInfo | null>(null)
  useEffect(() => {
    if (theme === 'system') document.documentElement.removeAttribute('data-theme')
    else document.documentElement.dataset.theme = theme
  }, [theme])
  const setTheme = useCallback((next: 'system' | 'light' | 'dark') => {
    if (next === 'system') localStorage.removeItem('bt.theme')
    else localStorage.setItem('bt.theme', next)
    setThemeState(next)
  }, [])
  useEffect(() => { void api.version().then(setVersion).catch(() => undefined) }, [])

  const notify = useCallback((kind: 'info' | 'error', text: string) => {
    const id = Date.now() + Math.random()
    setToasts((t) => [...t.slice(-3), { id, kind, text }])
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), kind === 'error' ? 7000 : 3500)
  }, [])
  const gpu = useGpu(gpuOpen, notify)
  const gpuSummary = summarize(gpu)

  // ---------------------------------------------------------------- sessions list
  const loadSessions = useCallback(async (q = '') => {
    try {
      dispatch({ type: 'sessions/loaded', sessions: await api.sessions.list(false, q) })
    } catch (err) {
      dispatch({ type: 'sessions/error', message: err instanceof ApiError ? err.message : 'Could not load your chats.' })
    }
  }, [])
  useEffect(() => { void loadSessions() }, [loadSessions])
  useEffect(() => {
    const id = setTimeout(() => void loadSessions(query.trim()), query ? 280 : 0)
    return () => clearTimeout(id)
  }, [query, loadSessions])

  // ---------------------------------------------------------------- run plumbing
  const onEvent = useCallback((ev: ServerEvent) => {
    if (ev.type === 'run.start' && !stateRef.current.chat.sessionId) window.history.replaceState(null, '', `/c/${ev.session_id}`)
    dispatch({ type: 'run/event', event: ev })
  }, [])

  const reloadCurrent = useCallback(async () => {
    const id = stateRef.current.chat.sessionId
    if (!id) return
    try {
      const data = await api.sessions.get(id)
      if (stateRef.current.chat.sessionId === id && !stateRef.current.chat.run) dispatch({ type: 'chat/loaded', session: data.session, messages: data.messages })
    } catch { /* keep what is on screen */ }
  }, [])

  const afterRun = useCallback((outcome: Outcome, handle: RunHandle) => {
    const current = runRef.current === handle
    if (current) runRef.current = null
    if (!current && outcome !== 'done') return // a stale run (the user moved on) must not touch the active one
    if (outcome === 'done') { void loadSessions(query.trim()); return }
    if (outcome === 'aborted') {
      if (stateRef.current.chat.run) { dispatch({ type: 'run/lost', message: 'The connection was closed before the response finished.' }); void reloadCurrent() }
      return
    }
    dispatch({ type: 'run/lost', message: outcome === 'gone' ? 'The server no longer has this response in progress.' : 'The connection to the server was lost.' })
    setTimeout(() => void reloadCurrent(), 800)
  }, [loadSessions, query, reloadCurrent])

  const detach = useCallback(() => {
    const h = runRef.current
    runRef.current = null
    if (h) {
      void h.done.catch(() => undefined)
      h.detach() // stop *following* only; the run keeps generating and is saved server-side
    }
  }, [])

  const attach = useCallback((runId: string) => {
    dispatch({ type: 'attach/start', assistantId: uid('tmpa-'), runId })
    const handle = attachRun(runId, onEvent)
    runRef.current = handle
    handle.done.then((o) => afterRun(o, handle)).catch(() => { dispatch({ type: 'run/lost', message: 'Could not re-attach to the running response.' }); void reloadCurrent() })
  }, [onEvent, afterRun, reloadCurrent])

  // ---------------------------------------------------------------- navigation
  const goNew = useCallback((push = true) => {
    detach()
    openToken.current++
    dispatch({ type: 'chat/new' })
    if (push && window.location.pathname !== '/') window.history.pushState(null, '', '/')
    if (narrow) setDrawer(false)
  }, [detach, narrow])

  const openSession = useCallback(async (id: string, push = true) => {
    if (stateRef.current.chat.sessionId === id && !stateRef.current.chat.loadError && push) { if (narrow) setDrawer(false); return }
    detach()
    const token = ++openToken.current
    if (push) window.history.pushState(null, '', `/c/${id}`)
    if (narrow) setDrawer(false)
    dispatch({ type: 'chat/loading', id })
    try {
      const data = await api.sessions.get(id)
      if (token !== openToken.current) return
      dispatch({ type: 'chat/loaded', session: data.session, messages: data.messages })
      if (data.live_run) attach(data.live_run.run_id)
    } catch (err) {
      if (token !== openToken.current) return
      if (err instanceof ApiError && err.status === 404) {
        notify('error', 'That conversation no longer exists.')
        goNew(false)
        window.history.replaceState(null, '', '/')
      } else {
        dispatch({ type: 'chat/loadFailed', message: err instanceof ApiError ? err.message : 'Could not load this conversation.' })
      }
    }
  }, [attach, detach, goNew, narrow, notify])

  useEffect(() => {
    const id = routeId()
    if (id) void openSession(id, false)
    const onPop = () => { const next = routeId(); if (next) void openSession(next, false); else goNew(false) }
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const title = state.chat.title
    document.title = title ? `${title} · Blackthorn` : 'Blackthorn'
  }, [state.chat.title])

  // ---------------------------------------------------------------- sending
  const failedToStart = useCallback((err: unknown) => {
    const e = err instanceof ApiError ? err : new ApiError(0, 'network', 'Could not reach the server. Your message was not sent.')
    if (e.code.startsWith('gpu_')) void gpu.refresh()
    notify('error', e.message)
  }, [gpu, notify])

  const send = useCallback(async (text: string, files: AttachedFile[]): Promise<boolean> => {
    if (stateRef.current.chat.run) return false
    const userId = uid('tmp-')
    const assistantId = uid('tmpa-')
    dispatch({ type: 'send/start', userId, assistantId, text, attachments: files.map((f) => ({ name: f.name, chars: f.content.length })) })
    const handle = startRun({ session_id: stateRef.current.chat.sessionId ?? undefined, message: text, attachments: files }, onEvent)
    runRef.current = handle
    try {
      afterRun(await handle.done, handle)
      return true
    } catch (err) {
      runRef.current = null
      dispatch({ type: 'send/failed', userId, assistantId })
      failedToStart(err)
      return false
    }
  }, [afterRun, failedToStart, onEvent])

  const regenerate = useCallback(async () => {
    const id = stateRef.current.chat.sessionId
    if (!id || stateRef.current.chat.run) return
    dispatch({ type: 'regenerate/start', assistantId: uid('tmpa-') })
    const handle = startRun({ session_id: id, regenerate: true }, onEvent)
    runRef.current = handle
    try {
      afterRun(await handle.done, handle)
    } catch (err) {
      runRef.current = null
      dispatch({ type: 'run/lost', message: err instanceof ApiError ? err.message : 'Could not retry.' })
      failedToStart(err)
      void reloadCurrent()
    }
  }, [afterRun, failedToStart, onEvent, reloadCurrent])

  const stop = useCallback(async () => {
    dispatch({ type: 'run/stopping' })
    await runRef.current?.stop()
  }, [])

  // ---------------------------------------------------------------- history actions (all persisted server-side)
  const patch = useCallback(async (s: Session, change: { title?: string; pinned?: boolean; archived?: boolean }, optimistic: Partial<Session>) => {
    dispatch({ type: 'session/patched', session: { ...s, ...optimistic } })
    try {
      dispatch({ type: 'session/patched', session: await api.sessions.patch(s.id, change) })
    } catch (err) {
      dispatch({ type: 'session/patched', session: s })
      notify('error', err instanceof ApiError ? err.message : 'That change could not be saved.')
    }
  }, [notify])
  const find = (id: string) => stateRef.current.sessions.find((s) => s.id === id) ?? stateRef.current.archived?.find((s) => s.id === id)
  const onRename = useCallback(async (id: string, title: string) => { const s = find(id); if (s) await patch(s, { title }, { title }) }, [patch])
  const onPin = useCallback(async (id: string, pinned: boolean) => { const s = find(id); if (s) await patch(s, { pinned }, { pinned }) }, [patch])
  const onArchive = useCallback(async (id: string, archived: boolean) => {
    const s = find(id)
    if (!s) return
    await patch(s, { archived }, { archived, pinned: archived ? false : s.pinned })
    notify('info', archived ? 'Chat archived.' : 'Chat restored.')
  }, [patch, notify])
  const loadArchived = useCallback(async () => {
    try { dispatch({ type: 'archived/loaded', sessions: await api.sessions.list(true) }) } catch (err) { notify('error', err instanceof ApiError ? err.message : 'Could not load archived chats.') }
  }, [notify])
  const confirmDelete = useCallback(async () => {
    if (!deleting) return
    setDeleteBusy(true)
    try {
      await api.sessions.remove(deleting.id)
      const wasCurrent = stateRef.current.chat.sessionId === deleting.id
      dispatch({ type: 'session/removed', id: deleting.id })
      if (wasCurrent) { detach(); window.history.replaceState(null, '', '/') }
      notify('info', 'Chat deleted.')
      setDeleting(null)
    } catch (err) {
      notify('error', err instanceof ApiError ? err.message : 'Could not delete this chat.')
    } finally {
      setDeleteBusy(false)
    }
  }, [deleting, detach, notify])

  // ---------------------------------------------------------------- chrome
  const toggleSidebar = () => {
    if (narrow) setDrawer((v) => !v)
    else setSidebarPref((v) => { localStorage.setItem('bt.sidebar', v ? '0' : '1'); return !v })
  }
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && narrow && drawer) setDrawer(false) }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [narrow, drawer])

  const chat = state.chat
  const banner = useMemo(() => {
    if (gpu.status === null) return null
    const s = gpu.status
    if (s.online && s.model_loaded !== false) return null
    if (s.booting || gpu.busy === 'on') {
      const el = s.elapsed_seconds != null ? ` · ${formatElapsed(s.elapsed_seconds)}` : ''
      return <div className="banner warn" data-testid="gpu-banner" role="status"><span>{gpuSummary.label}{el}. You can send messages once it is ready.</span><button type="button" className="link" onClick={() => setGpuOpen(true)}>Details</button></div>
    }
    if (s.online) return <div className="banner warn" data-testid="gpu-banner" role="status"><span>Loading the model into GPU memory…</span></div>
    return (
      <div className={`banner ${s.error ? 'bad' : 'warn'}`} data-testid="gpu-banner" role="status">
        <span>{s.error ? `GPU problem: ${s.error}` : 'The GPU is off, so the assistant cannot answer yet.'}</span>
        <button type="button" className="link" onClick={() => void gpu.turnOn()} data-testid="banner-turn-on" disabled={!s.can_turn_on}>Turn on GPU</button>
      </div>
    )
  }, [gpu, gpuSummary.label])

  return (
    <div className={`app${sidebarOpen && !narrow ? ' with-sidebar' : ''}`} data-testid="app">
      <Sidebar open={sidebarOpen} narrow={narrow} onClose={() => setDrawer(false)} sessions={state.sessions} archived={state.archived}
        loaded={state.sessionsLoaded} error={state.sessionsError} currentId={chat.sessionId} query={query} onQuery={setQuery}
        onSelect={(id) => void openSession(id)} onNew={() => goNew()} onRename={onRename} onPin={onPin} onArchive={onArchive}
        onDelete={setDeleting} onLoadArchived={() => void loadArchived()} onRetry={() => void loadSessions(query.trim())}
        onSettings={() => setSettingsOpen(true)} version={version} />
      <div className="main">
        <Header title={chat.title} sidebarOpen={sidebarOpen} onToggleSidebar={toggleSidebar} onNew={() => goNew()} gpu={gpu}
          gpuOpen={gpuOpen} onToggleGpu={() => setGpuOpen((v) => !v)} onCloseGpu={() => setGpuOpen(false)} />
        <Chat chat={chat} onRegenerate={() => void regenerate()} onRetryLoad={() => chat.sessionId && void openSession(chat.sessionId, false)} />
        <Composer busy={chat.run !== null} stopping={chat.run?.stopping ?? false} onSend={send} onStop={() => void stop()}
          onError={(t) => notify('error', t)} focusKey={chat.sessionId ?? 'new'} banner={banner} />
      </div>
      {settingsOpen && <SettingsDialog theme={theme} onTheme={setTheme} onClose={() => setSettingsOpen(false)} onPromptSaved={() => notify('info', 'System prompt saved.')} />}
      {deleting && <Confirm title="Delete this chat?" danger confirmLabel="Delete" busy={deleteBusy}
        message={`“${deleting.title}” and all of its messages will be permanently deleted. This cannot be undone.`}
        onCancel={() => setDeleting(null)} onConfirm={() => void confirmDelete()} />}
      <Toasts toasts={toasts} onDismiss={(id) => setToasts((t) => t.filter((x) => x.id !== id))} />
    </div>
  )
}
