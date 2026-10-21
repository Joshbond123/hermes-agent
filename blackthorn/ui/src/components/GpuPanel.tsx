import { useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../api'
import { summarize, type GpuView } from '../gpu'
import type { FleetSlot, VersionInfo } from '../types'
import { formatBytes, formatElapsed } from '../util'
import { Confirm } from './Dialog'
import { Popover } from './Menu'

function useTick(active: boolean) {
  const [, set] = useState(0)
  useEffect(() => {
    if (!active) return
    const id = setInterval(() => set((n) => n + 1), 1000)
    return () => clearInterval(id)
  }, [active])
}

/** Elapsed since this boot: prefer server elapsed, but never go backwards; local 0 when boot just started. */
function useBootElapsed(gpu: GpuView): number | null {
  const booting = Boolean(gpu.status?.booting)
  const serverElapsed = gpu.status?.elapsed_seconds
  const localStart = useRef<number | null>(null)
  useTick(booting)

  useEffect(() => {
    if (booting) {
      if (localStart.current == null) localStart.current = Date.now()
    } else {
      localStart.current = null
    }
  }, [booting])

  if (!booting) return null
  const local = localStart.current != null ? (Date.now() - localStart.current) / 1000 : 0
  if (serverElapsed == null || serverElapsed < 0) return Math.max(0, Math.floor(local))
  // Prefer the larger of local and server so a late status refresh cannot jump the clock backward.
  return Math.max(0, Math.floor(Math.max(local, serverElapsed)))
}

function SlotRow({ label, slot, active }: { label: string; slot?: FleetSlot | null; active?: boolean }) {
  if (!slot) {
    return (
      <div className="fleet-slot muted">
        <span className="fleet-slot-label">{label}</span>
        <span>—</span>
      </div>
    )
  }
  if (slot.error) {
    return (
      <div className="fleet-slot bad">
        <span className="fleet-slot-label">{label}</span>
        <span className="fleet-slot-account">{slot.account || 'unknown'}</span>
        <span className="muted small">{slot.error}</span>
      </div>
    )
  }
  const rem = slot.remaining_hours ?? 0
  const tone = rem <= 0.5 ? 'bad' : rem <= 2 ? 'warn' : 'ok'
  return (
    <div className={`fleet-slot${active ? ' active' : ''}`}>
      <span className="fleet-slot-label">{label}{active ? ' · active' : ''}</span>
      <span className="fleet-slot-account">{slot.account}</span>
      <span className={`fleet-slot-quota tone-${tone}`}>{rem.toFixed(1)}h left</span>
      <div className="fleet-quota-bar" aria-hidden>
        <i style={{ width: `${Math.min(100, Math.max(0, 100 - (slot.used_pct || 0)))}%` }} />
      </div>
      <span className="muted small">
        {(slot.used_hours ?? 0).toFixed(1)} / {(slot.total_hours ?? 30).toFixed(0)}h used this week
      </span>
    </div>
  )
}

export function GpuButton({ gpu, open, onToggle }: { gpu: GpuView; open: boolean; onToggle: () => void }) {
  const sum = summarize(gpu)
  const elapsed = useBootElapsed(gpu)
  return (
    <button
      type="button"
      className={`gpu-btn tone-${sum.tone}${open ? ' open' : ''}`}
      onClick={onToggle}
      aria-haspopup="dialog"
      aria-expanded={open}
      data-popover-trigger
      data-testid="gpu-button"
      data-state={sum.tone}
      title={sum.detail}
    >
      <span className="dot" aria-hidden />
      <span className="gpu-label">{sum.label}</span>
      {elapsed != null && <span className="gpu-elapsed">{formatElapsed(elapsed)}</span>}
    </button>
  )
}

export function GpuPanel({ gpu, onClose }: { gpu: GpuView; onClose: () => void }) {
  const s = gpu.status
  const sum = summarize(gpu)
  const elapsed = useBootElapsed(gpu)
  const [confirmOff, setConfirmOff] = useState(false)
  const [showLogs, setShowLogs] = useState(false)
  const [logs, setLogs] = useState<{ lines?: string[]; error?: string } | null>(null)
  const [auto, setAuto] = useState<{ minutes: number; choices: number[] } | null>(null)
  const [autoErr, setAutoErr] = useState('')
  const [version, setVersion] = useState<VersionInfo | null>(null)

  useEffect(() => {
    void api.version().then(setVersion).catch(() => null)
    void api.gpu.activity().then((a: any) => {
      setAuto({ minutes: a.auto_off_minutes ?? 15, choices: a.choices?.length ? a.choices : [0, 5, 10, 15, 30, 60] })
    }).catch(() => null)
  }, [])

  useEffect(() => {
    if (!showLogs) return
    void api.gpu.logs().then(setLogs).catch((e) => setLogs({ error: e instanceof ApiError ? e.message : String(e) }))
  }, [showLogs])

  async function changeAuto(minutes: number) {
    setAutoErr('')
    try {
      const out: any = await api.gpu.setAutoOff(minutes)
      setAuto({ minutes: out.auto_off_minutes ?? minutes, choices: out.choices?.length ? out.choices : [0, 5, 10, 15, 30, 60] })
    } catch (e) {
      setAutoErr(e instanceof ApiError ? e.message : String(e))
    }
  }

  const modelSlots = s?.fleet_slots?.model
  const compSlots = s?.fleet_slots?.computer
  const modelQuota = s?.quotas?.model || s?.quota
  const compQuota = s?.quotas?.computer

  const pct =
    s?.progress_bytes_done != null && s?.progress_bytes_total
      ? Math.min(100, Math.round((100 * s.progress_bytes_done) / Math.max(1, s.progress_bytes_total)))
      : s?.progress_stage != null && s?.progress_total_stages
        ? Math.min(100, Math.round((100 * s.progress_stage) / Math.max(1, s.progress_total_stages)))
        : null

  return (
    <>
      <div className="gpu-panel-scrim" onClick={onClose} aria-hidden />
      <Popover onClose={onClose} className="gpu-panel gpu-panel-v2" testId="gpu-panel">
        <header className="gpu-panel-head">
          <div>
            <strong data-testid="gpu-status-text">{s?.display_status || sum.label}</strong>
            <div className="muted small">
              {s?.model_name || s?.model || 'Qwen3.8-27B-Uncensored'}
              {s?.transport ? ` · ${s.transport}` : ''}
            </div>
          </div>
          <span className={`gpu-pill tone-${sum.tone}`}>{sum.tone}</span>
        </header>

        {gpu.error && <div className="notice error small">{gpu.error}</div>}
        {s?.error && <div className="notice error small">{s.error}</div>}

        <section className="gpu-section">
          <div className="gpu-section-title">Model GPU</div>
          <div className="gpu-kv">
            <span>Status</span>
            <span>{s?.status || '—'}</span>
          </div>
          <div className="gpu-kv">
            <span>Account</span>
            <span>{modelQuota?.account || modelSlots?.primary?.account || '—'}</span>
          </div>
          {elapsed != null && (
            <div className="gpu-kv">
              <span>Boot time</span>
              <span className="gpu-elapsed-strong">{formatElapsed(elapsed)}</span>
            </div>
          )}
          {s?.progress_step && (
            <div className="muted small gpu-step">{s.progress_step}</div>
          )}
          {pct != null && (
            <div className="gpu-progress" aria-label={`${pct}%`}>
              <i style={{ width: `${pct}%` }} />
              <span>
                {s?.progress_bytes_done != null && s?.progress_bytes_total
                  ? `${formatBytes(s.progress_bytes_done)} / ${formatBytes(s.progress_bytes_total)}`
                  : s?.progress_stage != null
                    ? `Stage ${s.progress_stage}/${s.progress_total_stages}`
                    : `${pct}%`}
              </span>
            </div>
          )}
          <div className="fleet-grid">
            <SlotRow label="Primary" slot={modelSlots?.primary} active={modelSlots?.active === 'primary'} />
            <SlotRow label="Backup" slot={modelSlots?.backup} active={modelSlots?.active === 'backup'} />
          </div>
        </section>

        <section className="gpu-section">
          <div className="gpu-section-title">Computer GPU</div>
          <div className="gpu-kv">
            <span>Status</span>
            <span>
              {s?.computer?.display_status || s?.computer?.status || '—'}
              {s?.computer_ready ? ' · ready' : ''}
            </span>
          </div>
          <div className="gpu-kv">
            <span>Account</span>
            <span>{compQuota?.account || compSlots?.primary?.account || '—'}</span>
          </div>
          <div className="gpu-kv">
            <span>Transport</span>
            <span>{s?.computer?.transport || s?.computer?.host || '—'}</span>
          </div>
          <div className="fleet-grid">
            <SlotRow label="Primary" slot={compSlots?.primary} active={compSlots?.active === 'primary'} />
            <SlotRow label="Backup" slot={compSlots?.backup} active={compSlots?.active === 'backup'} />
          </div>
        </section>

        {s?.model_cache_dataset && (
          <div className="muted small">Model cache: {s.model_cache_dataset}</div>
        )}

        {auto && (
          <div className="auto-off-row">
            <label className="muted small" htmlFor="auto-off">Auto-off after idle</label>
            <select id="auto-off" value={auto.minutes} onChange={(e) => void changeAuto(Number(e.target.value))}>
              {auto.choices.map((m) => (
                <option key={m} value={m}>{m === 0 ? 'Never' : `${m} min`}</option>
              ))}
            </select>
            {s?.auto_off?.reason && <span className="muted small">{s.auto_off.reason}</span>}
            {autoErr && <span className="notice error small">{autoErr}</span>}
          </div>
        )}

        <div className="panel-actions">
          {s?.can_turn_on && (
            <button type="button" className="btn primary" disabled={gpu.busy !== null} data-testid="gpu-turn-on"
              onClick={() => void gpu.turnOn()}>
              {gpu.busy === 'on' ? 'Starting…' : 'Turn GPU on'}
            </button>
          )}
          {s?.can_turn_off && (
            <button type="button" className="btn danger" disabled={gpu.busy !== null} data-testid="gpu-turn-off"
              onClick={() => setConfirmOff(true)}>
              {gpu.busy === 'off' ? 'Stopping…' : 'Turn GPU off'}
            </button>
          )}
          <button type="button" className="btn ghost" onClick={() => void gpu.refresh()}>Refresh</button>
          <button type="button" className="btn ghost" onClick={() => setShowLogs((v) => !v)}>
            {showLogs ? 'Hide logs' : 'Logs'}
          </button>
        </div>

        {showLogs && (
          <pre className="gpu-logs" data-testid="gpu-logs">
            {logs?.error ? logs.error : (logs?.lines || []).join('\n') || 'No log lines yet.'}
          </pre>
        )}

        {version && (
          <div className="muted small version-foot">
            build {version.commit?.slice(0, 8) || version.version}
          </div>
        )}
      </Popover>

      {confirmOff && (
        <Confirm
          title="Turn off GPU?"
          message="This stops the model and computer Kaggle sessions and frees weekly GPU quota."
          confirmLabel="Turn off"
          danger
          onCancel={() => setConfirmOff(false)}
          onConfirm={() => { setConfirmOff(false); void gpu.turnOff() }}
        />
      )}
    </>
  )
}
