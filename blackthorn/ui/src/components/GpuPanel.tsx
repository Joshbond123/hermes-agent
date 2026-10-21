import { useEffect, useState } from 'react'
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

function SlotRow({ label, slot, active }: { label: string; slot?: FleetSlot | null; active?: boolean }) {
  if (!slot) return (
    <div className="fleet-slot muted">
      <span className="fleet-slot-label">{label}</span>
      <span>—</span>
    </div>
  )
  if (slot.error) return (
    <div className="fleet-slot bad">
      <span className="fleet-slot-label">{label}</span>
      <span className="fleet-slot-account">{slot.account || 'unknown'}</span>
      <span className="muted small">{slot.error}</span>
    </div>
  )
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
      <span className="muted small">{(slot.used_hours ?? 0).toFixed(1)} / {(slot.total_hours ?? 30).toFixed(0)}h used</span>
    </div>
  )
}

export function GpuButton({ gpu, open, onToggle }: { gpu: GpuView; open: boolean; onToggle: () => void }) {
  const sum = summarize(gpu)
  useTick(Boolean(gpu.status?.booting))
  const elapsed = gpu.status?.booting && gpu.status.elapsed_seconds != null
    ? gpu.status.elapsed_seconds + (Date.now() - gpu.fetchedAt) / 1000
    : null
  return (
    <button type="button" className={`gpu-btn tone-${sum.tone}${open ? ' open' : ''}`} onClick={onToggle}
      aria-haspopup="dialog" aria-expanded={open}
      data-popover-trigger data-testid="gpu-button" data-state={sum.tone} title={sum.detail}>
      <span className="dot" aria-hidden />
      <span className="gpu-label">{sum.label}</span>
      {elapsed != null && <span className="gpu-elapsed">{formatElapsed(elapsed)}</span>}
    </button>
  )
}

export function GpuPanel({ gpu, onClose }: { gpu: GpuView; onClose: () => void }) {
  const s = gpu.status
  const sum = summarize(gpu)
  const [confirmOff, setConfirmOff] = useState(false)
  const [showLogs, setShowLogs] = useState(false)
  const [logs, setLogs] = useState<{ lines: string[]; error?: string } | null>(null)
  const [auto, setAuto] = useState<{ minutes: number; choices: number[] } | null>(null)
  const [autoErr, setAutoErr] = useState<string | null>(null)
  const [version, setVersion] = useState<VersionInfo | null>(null)
  useTick(Boolean(s?.booting))

  useEffect(() => {
    void api.gpu.activity().then((a) => setAuto({ minutes: a.auto_off_minutes, choices: a.choices })).catch(() => undefined)
    void api.version().then(setVersion).catch(() => undefined)
  }, [])
  useEffect(() => {
    if (!showLogs) return
    let alive = true
    const load = () => api.gpu.logs().then((l) => alive && setLogs(l)).catch((e) => alive && setLogs({ lines: [], error: e instanceof ApiError ? e.message : 'Could not load logs' }))
    void load()
    const id = setInterval(load, 8000)
    return () => { alive = false; clearInterval(id) }
  }, [showLogs])

  const elapsed = s?.booting && s.elapsed_seconds != null
    ? s.elapsed_seconds + (Date.now() - gpu.fetchedAt) / 1000
    : null
  const bytes = s?.progress_kind === 'bytes' && s.progress_bytes_total ? s : null
  const stageFrac = s?.progress_stage && s.progress_total_stages ? s.progress_stage / s.progress_total_stages : 0
  const changeAuto = async (minutes: number) => {
    setAutoErr(null)
    try {
      const r = await api.gpu.setAutoOff(minutes)
      setAuto((a) => (a ? { ...a, minutes: r.auto_off_minutes } : a))
    } catch (e) {
      setAutoErr(e instanceof ApiError ? e.message : 'Could not save')
    }
  }

  const fleet = s?.fleet_slots
  const modelName = s?.model_name || s?.model || 'Qwen3.8-27B-Uncensored'
  const cacheDs = s?.model_cache_dataset

  return (
    <>
      <div className="gpu-panel-scrim" data-testid="gpu-panel-scrim" aria-hidden onClick={onClose} />
      <Popover onClose={onClose} className="gpu-panel gpu-panel-v2" testId="gpu-panel">
        <div className="panel-head">
          <span className={`dot tone-${sum.tone}`} aria-hidden />
          <div>
            <strong data-testid="gpu-status-text">{s?.display_status || sum.label}</strong>
            <div className="muted small">{sum.detail}</div>
          </div>
        </div>

        {gpu.error && (
          <div className="notice error" role="alert">
            {gpu.error}{' '}
            <button type="button" className="link" onClick={() => void gpu.refresh()}>Retry</button>
          </div>
        )}
        {s?.error && !s.online && (
          <div className="notice error" role="alert" data-testid="gpu-error">{s.error}</div>
        )}

        <div className="gpu-meta-grid">
          <div>
            <div className="muted small">Model</div>
            <div className="gpu-meta-value" data-testid="gpu-model-name">{modelName}</div>
          </div>
          <div>
            <div className="muted small">Cache dataset</div>
            <div className="gpu-meta-value mono small" data-testid="gpu-cache-dataset">
              {cacheDs || 'black797/qwen38-27b-uncensored-model-cache'}
            </div>
          </div>
        </div>

        {s?.booting && (
          <div className="progress-block" data-testid="gpu-progress">
            <div className="progress-line">
              <span>
                {bytes
                  ? `${s.progress_label ?? 'Transferring'} · ${formatBytes(bytes.progress_bytes_done ?? 0)} of ${formatBytes(bytes.progress_bytes_total ?? 0)}`
                  : s.progress_stage
                    ? `Step ${s.progress_stage} of ${s.progress_total_stages}`
                    : 'Starting'}
              </span>
              {elapsed != null && <span className="muted">{formatElapsed(elapsed)} elapsed</span>}
            </div>
            <div className="bar" role="progressbar" aria-valuemin={0} aria-valuemax={100}
              aria-valuenow={Math.round((bytes ? (bytes.progress_bytes_done ?? 0) / (bytes.progress_bytes_total ?? 1) : stageFrac) * 100)}>
              <i style={{ width: `${Math.round((bytes ? (bytes.progress_bytes_done ?? 0) / (bytes.progress_bytes_total ?? 1) : stageFrac) * 100)}%` }} />
            </div>
            <div className="muted small">{s.progress_step}</div>
          </div>
        )}

        <section className="fleet-section" data-testid="gpu-fleet-model">
          <h3>Model GPU fleet</h3>
          <p className="muted small">Primary hosts Qwen permanently (Kaggle dataset cache). Backup starts when primary has ≤30 min left.</p>
          <SlotRow label="Primary" slot={fleet?.model?.primary || s?.quotas?.model} active={(fleet?.model?.active || 'primary') === 'primary'} />
          <SlotRow label="Backup" slot={fleet?.model?.backup} active={fleet?.model?.active === 'backup'} />
        </section>

        <section className="fleet-section" data-testid="gpu-fleet-computer">
          <h3>Computer GPU fleet</h3>
          <p className="muted small">Agent terminal, downloads, and workspace. Same 30-minute failover rule.</p>
          <SlotRow label="Primary" slot={fleet?.computer?.primary || s?.quotas?.computer} active={(fleet?.computer?.active || 'primary') === 'primary'} />
          <SlotRow label="Backup" slot={fleet?.computer?.backup} active={fleet?.computer?.active === 'backup'} />
          {s?.computer && (
            <div className="computer-status muted small">
              Computer: {s.computer.display_status || s.computer.status || (s.computer.online ? 'online' : 'offline')}
              {s.computer_ready ? ' · ready' : ''}
            </div>
          )}
        </section>

        {auto && (
          <div className="auto-off-row">
            <label className="muted small" htmlFor="auto-off">Auto-off after idle</label>
            <select id="auto-off" value={auto.minutes} onChange={(e) => void changeAuto(Number(e.target.value))}>
              {auto.choices.map((m) => (
                <option key={m} value={m}>{m === 0 ? 'Never' : `${m} min`}</option>
              ))}
            </select>
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
