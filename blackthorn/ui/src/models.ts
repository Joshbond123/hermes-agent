/**
 * The models a conversation can be answered by, and switching between them.
 *
 * A conversation owns its model (the server stores it on the session), so switching here is like
 * changing the model inside an existing ChatGPT conversation: same chat, same transcript, same plan.
 * A task that is already running keeps the model it started with — the server says so explicitly and
 * this hook surfaces that instead of pretending the swap happened mid-task.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'

export interface ModelOption {
  id: string
  label: string
  provider: string
  model: string
  available: boolean
  reason: string
  detail: string
  active: boolean
}

export interface ModelCatalogue {
  models: ModelOption[]
  active: string
  default: string
}

export interface ModelsView {
  models: ModelOption[]
  activeId: string
  active: ModelOption | null
  loading: boolean
  error: string | null
  switching: boolean
  /** Set after a switch that could not apply to a running task yet. */
  pendingNote: string | null
  refresh: () => Promise<void>
  choose: (id: string) => Promise<boolean>
}

export function useModels(sessionId: string | null, runActive: boolean): ModelsView {
  const [catalogue, setCatalogue] = useState<ModelCatalogue | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [switching, setSwitching] = useState(false)
  const [pendingNote, setPendingNote] = useState<string | null>(null)
  const sessionRef = useRef<string | null>(sessionId)
  sessionRef.current = sessionId
  const errorText = (e: unknown) => (e instanceof Error ? e.message : 'Could not load the models.')

  const refresh = useCallback(async () => {
    const id = sessionRef.current
    try {
      const next = await api.models.list(id)
      // A conversation without a stored model answers with the server default.
      setCatalogue(next)
      setError(null)
    } catch (e) {
      setError(errorText(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { setLoading(true); void refresh() }, [refresh, sessionId])

  const choose = useCallback(async (id: string) => {
    const session = sessionRef.current
    const picked = catalogue?.models.find((m) => m.id === id)
    if (picked && !picked.available) { setError(`${picked.label} is not available: ${picked.reason}`); return false }
    if (!session) {
      // New chat: the choice is applied to the first message, and remembered for that chat by the
      // server as soon as it exists.
      setCatalogue((c) => (c ? { ...c, active: id, models: c.models.map((m) => ({ ...m, active: m.id === id })) } : c))
      setPendingNote(null)
      return true
    }
    setSwitching(true)
    try {
      const out = await api.models.choose(session, id)
      setCatalogue((c) => (c ? { ...c, active: id, models: c.models.map((m) => ({ ...m, active: m.id === id })) } : c))
      setPendingNote(out.active_run
        ? `“${picked?.label ?? id}” is now this chat's model. The task that is running keeps ${out.active_run.model || 'the model it started with'} until it finishes.`
        : null)
      setError(null)
      return true
    } catch (e) {
      setError(errorText(e))
      return false
    } finally {
      setSwitching(false)
    }
  }, [catalogue])

  // A run finishing during the switch turns the note into history.
  useEffect(() => { if (!runActive && pendingNote) setPendingNote(null) }, [runActive, pendingNote])

  const activeId = catalogue?.active ?? ''
  return {
    models: catalogue?.models ?? [],
    activeId,
    active: catalogue?.models.find((m) => m.id === activeId) ?? null,
    loading, error, switching, pendingNote, refresh, choose,
  }
}
