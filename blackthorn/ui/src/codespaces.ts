import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError } from './api'
import type { CodespaceStatus, CodespaceUsage } from './types'

export interface CodespacesView {
  status: CodespaceStatus | null
  usage: CodespaceUsage | null
  fetchedAt: number
  error: string | null
  busy: 'start' | 'stop' | 'restart' | null
  refresh: () => Promise<void>
  start: () => Promise<void>
  stop: () => Promise<void>
  restart: () => Promise<void>
}

/**
 * Live Codespaces state and month-to-date usage.
 *
 * The panel never shows a stale number: status and usage are fetched together, polled every 5s
 * while the panel is open (15s otherwise) and again whenever the tab regains focus, so "hours
 * remaining" and "the agent is using it" move in something close to real time.
 */
export function useCodespaces(panelOpen: boolean, notify: (kind: 'info' | 'error', text: string) => void): CodespacesView {
  const [status, setStatus] = useState<CodespaceStatus | null>(null)
  const [usage, setUsage] = useState<CodespaceUsage | null>(null)
  const [fetchedAt, setFetchedAt] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<'start' | 'stop' | 'restart' | null>(null)
  const inflight = useRef(false)
  const wasUsing = useRef<boolean | null>(null)

  const apply = useCallback((s: CodespaceStatus, u: CodespaceUsage | null) => {
    setStatus(s)
    if (u) setUsage(u)
    setFetchedAt(Date.now())
    setError(null)
    if (u && wasUsing.current === true && !s.agent_using) {
      notify('info', 'The assistant finished with the computer; it will stop when idle.')
    }
    wasUsing.current = Boolean(s.agent_using)
  }, [notify])

  const refresh = useCallback(async () => {
    if (inflight.current) return
    inflight.current = true
    try {
      const [s, u] = await Promise.all([api.codespaces.status(), api.codespaces.usage()])
      apply(s, u)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Codespaces status is unavailable.')
    } finally {
      inflight.current = false
    }
  }, [apply])

  useEffect(() => { void refresh() }, [refresh])
  useEffect(() => {
    const interval = panelOpen ? 5000 : 15000
    const id = setInterval(() => void refresh(), interval)
    return () => clearInterval(id)
  }, [refresh, panelOpen])
  useEffect(() => {
    const wake = () => { if (document.visibilityState === 'visible') void refresh() }
    document.addEventListener('visibilitychange', wake)
    window.addEventListener('focus', wake)
    window.addEventListener('online', wake)
    return () => {
      document.removeEventListener('visibilitychange', wake)
      window.removeEventListener('focus', wake)
      window.removeEventListener('online', wake)
    }
  }, [refresh])

  const act = useCallback(async (kind: 'start' | 'stop' | 'restart') => {
    setBusy(kind)
    try {
      const next = kind === 'start' ? await api.codespaces.start()
        : kind === 'stop' ? await api.codespaces.stop()
        : await api.codespaces.restart()
      if (kind === 'stop' && next && (next as { stopped?: boolean }).stopped === false) {
        notify('info', (next as { detail?: string }).detail || 'The computer is busy, so it stayed on.')
      } else {
        notify('info', kind === 'start' ? 'Starting the computer.' : kind === 'stop' ? 'The computer is stopping.' : 'Restarting the computer; its files are kept.')
      }
      void refresh()
    } catch (err) {
      notify('error', err instanceof ApiError ? err.message : 'The computer action failed.')
    } finally {
      setBusy(null)
      void refresh()
    }
  }, [notify, refresh])

  return {
    status, usage, fetchedAt, error, busy, refresh,
    start: () => act('start'), stop: () => act('stop'), restart: () => act('restart'),
  }
}

export type Tone = 'ok' | 'warn' | 'bad' | 'off' | 'idle'

const LABEL: Record<string, string> = {
  ON: 'Computer on',
  OFF: 'Computer off',
  STARTING: 'Computer starting',
  STOPPING: 'Computer stopping',
  ERROR: 'Computer error',
}

/** One-line summary for the header button. Remaining hours lead, because that is the number that runs out. */
export function summarize(c: CodespacesView): { label: string; tone: Tone; detail: string; hours: string } {
  const u = c.usage
  const s = c.status
  const hours = u ? `${u.hours_remaining.toFixed(1)}h left` : ''
  if (c.error && !s) return { label: 'Computer unavailable', tone: 'bad', detail: c.error, hours }
  if (!s) return { label: 'Checking computer', tone: 'idle', detail: 'Checking GitHub Codespaces', hours }
  const state = (c.busy === 'start' ? 'STARTING' : c.busy === 'stop' ? 'STOPPING' : s.state) || 'OFF'
  if (state === 'ERROR') return { label: LABEL.ERROR, tone: 'bad', detail: s.error || 'The computer reported an error', hours }
  if (state === 'STARTING' || state === 'STOPPING') {
    return { label: LABEL[state], tone: 'warn', detail: state === 'STARTING' ? 'Booting the codespace' : 'Shutting down safely', hours }
  }
  if (state === 'ON') {
    if (s.agent_using) return { label: 'Computer in use', tone: 'ok', detail: 'The assistant is working on the computer now', hours }
    return { label: 'Computer on · idle', tone: 'warn', detail: 'Waiting; it stops automatically when idle', hours }
  }
  return { label: LABEL.OFF, tone: 'off', detail: 'Off — nothing is being billed while it is off', hours }
}

/** Share of the monthly allowance already used, clamped for the bar. */
export function usageBar(u: CodespaceUsage | null): { used: number; tone: Tone } {
  if (!u) return { used: 0, tone: 'idle' }
  const used = Math.min(100, Math.max(0, u.percent_used))
  const tone: Tone = u.hours_remaining <= 1 ? 'bad' : u.hours_remaining <= Math.max(3, u.hours_total * 0.15) ? 'warn' : 'ok'
  return { used, tone }
}
