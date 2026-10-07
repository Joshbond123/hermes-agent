import { useEffect, useState } from 'react'
import { api, ApiError } from '../api'
import { summarize, type GpuView } from '../gpu'
import type { VersionInfo } from '../types'
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

export function GpuButton({ gpu, open, onToggle }: { gpu: GpuView; open: boolean; onToggle: () => void }) {
  const sum = summarize(gpu)
  useTick(Boolean(gpu.status?.booting))
  const elapsed = gpu.status?.booting && gpu.status.elapsed_seconds != null ? gpu.status.elapsed_seconds + (Date.now() - gpu.fetchedAt) / 1000 : null
  return (
    <button type="button" className={`gpu-btn tone-${sum.tone}${open ? ' open' : ''}`} onClick={onToggle} aria-haspopup="dialog" aria-expanded={open}
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

  const elapsed = s?.booting && s.elapsed_seconds != null ? s.elapsed_seconds + (Date.now() - gpu.fetchedAt) / 1000 : null
  const q = s?.quota
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

  return (
    <>
      <Popover onClose={onClose} className="gpu-panel" testId="gpu-panel">
        <div className="panel-head">
          <span className={`dot tone-${sum.tone}`} aria-hidden />
          <div>
            <strong data-testid="gpu-status-text">{s?.display_status || sum.label}</strong>
            <div className="muted small">{sum.detail}</div>
          </div>
        </div>
        {gpu.error && <div className="notice error" role="alert">{gpu.error} <button type="button" className="link" onClick={() => void gpu.refresh()}>Retry</button></div>}
        {s?.error && !s.online && <div className="notice error" role="alert" data-testid="gpu-error">{s.error}</div>}

        {s?.booting && (
          <div className="progress-block" data-testid="gpu-progress">
            <div className="progress-line">
              <span>{bytes ? `Downloading · ${formatBytes(bytes.progress_bytes_done ?? 0)} of ${formatBytes(bytes.progress_bytes_total ?? 0)}` : s.progress_stage ? `Step ${s.progress_stage} of ${s.progress_total_stages}` : 'Starting'}</span>
              {elapsed != null && <span className="muted">{formatElapsed(elapsed)} elapsed</span>}
            </div>
            <div className="bar" role="progressbar" aria-valuemin={0} aria-valuemax={100}
              aria-valuenow={Math.round((bytes ? (bytes.progress_bytes_done ?? 0) / (bytes.progress_bytes_total ?? 1) : stageFrac) * 100)}>
              <i style={{ width: `${Math.round((bytes ? (bytes.progress_bytes_done ?? 0) / (bytes.progress_bytes_total ?? 1) : stageFrac) * 100)}%` }} />
            </div>
            <div className="muted small">{s.progress_step}</div>
          </div>
        )}

        <dl className="kv">
          {s?.model && (<><dt>Model</dt><dd>{s.model}</dd></>)}
          {s?.gpu_info && (<><dt>GPUs</dt><dd>{s.gpu_info}</dd></>)}
          {s?.online && s.model_loaded != null && (<><dt>Model memory</dt><dd>{s.model_loaded ? 'Loaded' : 'Loading…'}</dd></>)}
        </dl>

        {q && (
          <div className="quota" data-testid="gpu-quota">
            <div className="progress-line"><span>Weekly GPU quota</span><span className="muted">{q.used_hours.toFixed(1)} h of {q.total_hours} h used</span></div>
            <div className="bar"><i style={{ width: `${Math.min(100, q.used_pct)}%` }} /></div>
            {q.refresh_time && <div className="muted small">Resets {new Date(q.refresh_time).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</div>}
          </div>
        )}

        <div className="panel-actions">
          {s?.can_turn_on && <button type="button" className="btn primary" onClick={() => void gpu.turnOn()} disabled={gpu.busy !== null} data-testid="gpu-turn-on">{gpu.busy === 'on' ? 'Starting…' : 'Turn on GPU'}</button>}
          {s?.can_turn_off && <button type="button" className="btn danger-outline" onClick={() => setConfirmOff(true)} disabled={gpu.busy !== null} data-testid="gpu-turn-off">Turn off GPU</button>}
          <button type="button" className="btn" onClick={() => void gpu.refresh()} data-testid="gpu-refresh">Refresh</button>
        </div>

        {auto && (
          <label className="field row">
            <span>Turn off after inactivity</span>
            <select value={auto.minutes} onChange={(e) => void changeAuto(Number(e.target.value))} data-testid="gpu-auto-off">
              {auto.choices.map((m) => <option key={m} value={m}>{m === 0 ? 'Never' : `${m} min`}</option>)}
            </select>
          </label>
        )}
        {autoErr && <div className="notice error">{autoErr}</div>}

        <button type="button" className="link small" onClick={() => setShowLogs((v) => !v)} aria-expanded={showLogs} data-testid="gpu-logs-toggle">{showLogs ? 'Hide' : 'Show'} server log</button>
        {showLogs && (
          <pre className="logs" data-testid="gpu-logs">{logs ? (logs.error ? logs.error : logs.lines.join('\n') || '(empty)') : 'Loading…'}</pre>
        )}
        {version && <div className="muted tiny" data-testid="version-line">Blackthorn v{version.version} · {version.commit.slice(0, 7)}{version.ui?.hash ? ` · ui ${version.ui.hash.slice(0, 7)}` : ''}</div>}
      </Popover>
      {confirmOff && (
        <Confirm title="Turn off the GPU?" danger confirmLabel="Turn off" busy={gpu.busy === 'off'}
          message="This stops the Kaggle session that runs the model. Any response being generated will fail, and turning the GPU back on takes several minutes."
          onCancel={() => setConfirmOff(false)} onConfirm={async () => { await gpu.turnOff(); setConfirmOff(false) }} />
      )}
    </>
  )
}
