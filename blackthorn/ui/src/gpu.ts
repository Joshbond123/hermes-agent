import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'

/** What the server says about the Kaggle GPU. Nothing here is guessed in the browser. */
export type GpuStatus = {
  provider: string
  state: 'OFF' | 'STARTING' | 'LOADING' | 'READY' | 'STOPPING' | 'ERROR'
  running: boolean
  serving: boolean
  base_url: string
  phase: string
  loaded_seconds: number | null
  weights_source: string
  reconnected: boolean
  session_seconds: number
  last_activity: number | null
  last_activity_reason: string
  stop_requested: boolean
  error: string
  dataset: string
  kaggle_user: string
  busy: string
  kernels?: Record<string, string>
  settings: { idle_minutes: number; idle_choices: number[]; weekly_hours: number }
}

export type GpuUsage = {
  hours_total: number
  hours_used: number
  hours_remaining: number
  percent_used: number
  current_session_seconds: number
  sessions_this_week: number
  week_label: string
  usage_source: string
  hours_basis: string
}

export type GpuView = {
  status: GpuStatus | null
  usage: GpuUsage | null
  loading: boolean
  busy: boolean
  error: string
  refresh: () => void
  start: () => Promise<void>
  stop: () => Promise<void>
  setTimer: (minutes: number) => Promise<void>
  install: () => Promise<void>
}

/**
 * Live GPU state, polled from the server.
 *
 * The browser is only a window onto this: the idle timer and the off-order live on the server, so
 * closing the tab can neither keep a GPU alive nor lose one that is working. Polling stops while the
 * tab is hidden and refreshes immediately when it comes back.
 */
export function useGpu(): GpuView {
  const [status, setStatus] = useState<GpuStatus | null>(null)
  const [usage, setUsage] = useState<GpuUsage | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const alive = useRef(true)

  const load = useCallback(async () => {
    try {
      const [s, u] = await Promise.all([api.gpu.status(), api.gpu.usage()])
      if (!alive.current) return
      setStatus(s)
      setUsage(u)
      setError('')
    } catch (e) {
      if (!alive.current) return
      setError(e instanceof Error ? e.message : 'GPU status unavailable')
    } finally {
      if (alive.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    alive.current = true
    void load()
    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible') void load()
    }, 8000)
    const onVisible = () => { if (document.visibilityState === 'visible') void load() }
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      alive.current = false
      window.clearInterval(timer)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [load])

  const run = useCallback(async (fn: () => Promise<unknown>) => {
    setBusy(true)
    try {
      await fn()
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'That did not work')
    } finally {
      setBusy(false)
      void load()
    }
  }, [load])

  return {
    status, usage, loading, busy, error,
    refresh: () => void load(),
    start: () => run(() => api.gpu.start()),
    stop: () => run(() => api.gpu.stop()),
    setTimer: (minutes: number) => run(() => api.gpu.setSettings({ idle_minutes: minutes })),
    install: () => run(() => api.gpu.install()),
  }
}

/** ``m:ss`` / ``h:mm:ss`` for a duration in seconds. */
export function clock(seconds: number | null | undefined): string {
  const total = Math.max(0, Math.floor(seconds ?? 0))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  if (h) return `${h}h ${String(m).padStart(2, '0')}m`
  return `${m}:${String(s).padStart(2, '0')}`
}
