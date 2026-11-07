import { useEffect, useRef } from 'react'
import type { GpuView } from '../gpu'
import { clock } from '../gpu'

/**
 * The header's GPU control.
 *
 * Switching the GPU on starts a real Kaggle session that serves Qwen; switching it off releases the
 * quota. The trigger always shows the true server state, and the panel shows *measured* hours — the
 * allowance is displayed with the note that says where it came from, because Kaggle publishes no
 * usage API and a made-up number would be worse than no number.
 */
export function GpuPanel({ gpu, open, onToggle, onClose }: {
  gpu: GpuView
  open: boolean
  onToggle: () => void
  onClose: () => void
}) {
  const wrap = useRef<HTMLDivElement>(null)
  const status = gpu.status

  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (wrap.current && !wrap.current.contains(e.target as Node)) onClose()
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('mousedown', onDoc); document.removeEventListener('keydown', onKey) }
  }, [open, onClose])

  const state = status?.state ?? 'OFF'
  const on = state === 'READY' || state === 'STARTING' || state === 'LOADING'
  const tone = state === 'READY' ? 'on' : state === 'ERROR' ? 'bad' : on ? 'wait' : ''
  const label = state === 'READY' ? 'GPU on'
    : state === 'STARTING' ? 'GPU starting'
    : state === 'LOADING' ? 'GPU loading'
    : state === 'STOPPING' ? 'GPU stopping'
    : state === 'ERROR' ? 'GPU error'
    : 'GPU off'

  const usage = gpu.usage
  const idle = status?.settings.idle_minutes ?? 15
  const choices = status?.settings.idle_choices ?? [5, 10, 15, 30, 45, 60]

  return (
    <div className="rel gpu-picker" ref={wrap}>
      <button
        type="button"
        className={`gpu-trigger ${tone}${open ? ' open' : ''}`}
        onClick={onToggle}
        aria-haspopup="dialog"
        aria-expanded={open}
        data-testid="gpu-button"
        title={`${label}${status?.busy ? ` · ${status.busy}` : ''}`}
      >
        <span className={`gpu-dot ${tone}`} aria-hidden="true" />
        <span className="gpu-name" data-testid="gpu-state">{label}</span>
      </button>

      {open && (
        <div className="gpu-menu" role="dialog" aria-label="Kaggle GPU" data-testid="gpu-menu">
          <div className="gpu-head">
            <span>Kaggle GPU</span>
            {status?.session_seconds ? <span className="mono" data-testid="gpu-session">{clock(status.session_seconds)}</span> : null}
          </div>

          <div className="gpu-row gpu-power">
            <button
              type="button"
              className={`gpu-btn on${on ? ' active' : ''}`}
              onClick={() => void gpu.start()}
              disabled={gpu.busy || state === 'STARTING' || state === 'LOADING' || state === 'READY'}
              data-testid="gpu-on"
            >
              Turn on
            </button>
            <button
              type="button"
              className={`gpu-btn off${!on ? ' active' : ''}`}
              onClick={() => void gpu.stop()}
              // The server refuses this while work is running; the button says so before it is pressed.
              disabled={gpu.busy || !on || Boolean(status?.busy)}
              data-testid="gpu-off"
            >
              Turn off
            </button>
          </div>

          {status?.phase && state !== 'OFF' && state !== 'READY' ? (
            <div className="gpu-line" data-testid="gpu-phase">{status.phase}</div>
          ) : null}
          {status?.busy ? <div className="gpu-line warn" data-testid="gpu-busy">{status.busy} — the GPU will not be switched off mid-task.</div> : null}
          {status?.state === 'READY' ? (
            <div className="gpu-line ok" data-testid="gpu-serving">
              Qwen is answering from {status.weights_source || 'the persistent cache'}
              {status.loaded_seconds ? ` · ready in ${Math.round(status.loaded_seconds)}s` : ''}
            </div>
          ) : null}
          {status?.error ? <div className="gpu-line error" data-testid="gpu-error">{status.error}</div> : null}
          {gpu.error ? <div className="gpu-line error" data-testid="gpu-client-error">{gpu.error}</div> : null}
          {status && !status.dataset && status.kernels?.install === 'error' ? (
            <button type="button" className="gpu-install" onClick={() => void gpu.install()} data-testid="gpu-install">
              Install the model on Kaggle (one download, no GPU quota)
            </button>
          ) : null}

          <label className="gpu-timer">
            <span>Switch off after</span>
            <select
              value={idle}
              onChange={(e) => void gpu.setTimer(Number(e.target.value))}
              data-testid="gpu-timer"
            >
              {choices.map((m) => <option key={m} value={m}>{m} min idle</option>)}
              {choices.includes(idle) ? null : <option value={idle}>{idle} min idle</option>}
            </select>
          </label>

          <div className="gpu-usage" data-testid="gpu-usage">
            <div className="gpu-usage-head">This week <span className="dim">since {usage?.week_label ?? '—'}</span></div>
            <div className="gpu-bar" aria-hidden="true">
              <span style={{ width: `${Math.min(100, usage?.percent_used ?? 0)}%` }} />
            </div>
            <dl>
              <div><dt>Available</dt><dd data-testid="gpu-hours-total">{(usage?.hours_total ?? 0).toFixed(1)} h</dd></div>
              <div><dt>Used</dt><dd data-testid="gpu-hours-used">{(usage?.hours_used ?? 0).toFixed(2)} h</dd></div>
              <div><dt>Remaining</dt><dd data-testid="gpu-hours-left">{(usage?.hours_remaining ?? 0).toFixed(2)} h</dd></div>
              <div><dt>Used</dt><dd data-testid="gpu-percent">{(usage?.percent_used ?? 0).toFixed(1)}%</dd></div>
              <div><dt>Session</dt><dd data-testid="gpu-session-usage">{clock(usage?.current_session_seconds)}</dd></div>
            </dl>
            <p className="gpu-basis" data-testid="gpu-basis">{usage?.hours_basis}</p>
          </div>
        </div>
      )}
    </div>
  )
}
