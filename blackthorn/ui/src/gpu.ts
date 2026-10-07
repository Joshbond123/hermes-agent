import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError } from './api'
import type { GpuStatus } from './types'

export interface GpuView {
  status: GpuStatus | null
  fetchedAt: number
  error: string | null
  busy: 'on' | 'off' | null
  refresh: () => Promise<void>
  turnOn: () => Promise<void>
  turnOff: () => Promise<void>
}

/** Live GPU state. Polls faster while booting or while the panel is open, and again whenever the tab regains focus. */
export function useGpu(panelOpen: boolean, notify: (kind: 'info' | 'error', text: string) => void): GpuView {
  const [status, setStatus] = useState<GpuStatus | null>(null)
  const [fetchedAt, setFetchedAt] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<'on' | 'off' | null>(null)
  const inflight = useRef(false)
  const wasOnline = useRef<boolean | null>(null)

  const apply = useCallback((s: GpuStatus) => {
    setStatus(s)
    setFetchedAt(Date.now())
    setError(null)
    if (wasOnline.current === false && s.online) notify('info', 'The GPU is ready.')
    wasOnline.current = s.online
  }, [notify])

  const refresh = useCallback(async () => {
    if (inflight.current) return
    inflight.current = true
    try {
      apply(await api.gpu.status())
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'GPU status is unavailable.')
    } finally {
      inflight.current = false
    }
  }, [apply])

  const booting = Boolean(status?.booting) || busy !== null || status?.status === 'STOPPING_KAGGLE_GPU'
  useEffect(() => {
    void refresh()
  }, [refresh])
  useEffect(() => {
    const id = setInterval(() => void refresh(), booting || panelOpen ? 4000 : 15000)
    return () => clearInterval(id)
  }, [refresh, booting, panelOpen])
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

  const act = useCallback(async (kind: 'on' | 'off') => {
    setBusy(kind)
    try {
      apply(await (kind === 'on' ? api.gpu.turnOn() : api.gpu.turnOff()))
      notify('info', kind === 'on' ? 'Starting the GPU. This can take a few minutes.' : 'The GPU was turned off.')
    } catch (err) {
      notify('error', err instanceof ApiError ? err.message : 'The GPU action failed.')
    } finally {
      setBusy(null)
      void refresh()
    }
  }, [apply, notify, refresh])

  return { status, fetchedAt, error, busy, refresh, turnOn: () => act('on'), turnOff: () => act('off') }
}

export type Tone = 'ok' | 'warn' | 'bad' | 'off' | 'idle'

export function summarize(g: GpuView): { label: string; tone: Tone; detail: string } {
  const s = g.status
  if (!s) return g.error ? { label: 'GPU unavailable', tone: 'bad', detail: g.error } : { label: 'GPU…', tone: 'idle', detail: 'Checking the GPU' }
  if (g.busy === 'off' || s.status === 'STOPPING_KAGGLE_GPU') return { label: 'GPU stopping', tone: 'warn', detail: 'Stopping the Kaggle session' }
  if (s.booting || g.busy === 'on') {
    const step = s.progress_stage && s.progress_total_stages ? ` ${s.progress_stage}/${s.progress_total_stages}` : ''
    return { label: `GPU starting${step}`, tone: 'warn', detail: s.progress_step || s.display_status || 'Starting' }
  }
  if (s.online && s.model_loaded === false) return { label: 'Loading model', tone: 'warn', detail: 'Loading weights into GPU memory' }
  if (s.online) return { label: 'GPU ready', tone: 'ok', detail: s.model || 'Ready' }
  if (s.error) return { label: 'GPU error', tone: 'bad', detail: s.error }
  return { label: 'GPU off', tone: 'off', detail: s.display_status || 'Offline' }
}
