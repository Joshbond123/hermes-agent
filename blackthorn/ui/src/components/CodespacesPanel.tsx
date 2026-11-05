import { summarize, usageBar, type CodespacesView } from '../codespaces'
import type { CodespaceStatus, CodespaceUsage } from '../types'
import { Popover } from './Menu'

const STATE_TEXT: Record<string, string> = {
  ON: 'On', OFF: 'Off', STARTING: 'Starting', STOPPING: 'Stopping', ERROR: 'Error',
}

/** Header button: state dot + the number that actually runs out (hours remaining this month). */
export function CodespacesButton({ codespaces, open, onToggle }: { codespaces: CodespacesView; open: boolean; onToggle: () => void }) {
  const sum = summarize(codespaces)
  return (
    <button
      type="button"
      className={`cs-btn tone-${sum.tone}${open ? ' open' : ''}`}
      onClick={onToggle}
      aria-haspopup="dialog"
      aria-expanded={open}
      data-popover-trigger
      aria-label={`Codespaces: ${sum.label}${sum.hours ? `, ${sum.hours}` : ''}`}
      title={sum.detail}
      data-testid="codespaces-button"
    >
      <span className="cs-dot" aria-hidden />
      <span className="cs-label">{sum.label}</span>
      {sum.hours && <span className="cs-hours" data-testid="codespaces-hours-left">{sum.hours}</span>}
    </button>
  )
}

function Row({ label, value, strong }: { label: string; value: string; strong?: boolean }) {
  return (
    <div className="cs-kv">
      <span>{label}</span>
      <span className={strong ? 'cs-strong' : undefined}>{value}</span>
    </div>
  )
}

function UsageBody({ usage, status, error }: { usage: CodespaceUsage | null; status: CodespaceStatus | null; error: string | null }) {
  if (!usage) {
    return <p className="muted small">{error || 'Reading this month’s Codespaces usage…'}</p>
  }
  const bar = usageBar(usage)
  return (
    <>
      <div className="cs-remaining" data-testid="codespaces-remaining">
        <span className="cs-remaining-value">{usage.hours_remaining.toFixed(1)}</span>
        <span className="cs-remaining-unit">hours left</span>
      </div>
      <div className={`cs-bar tone-${bar.tone}`} aria-hidden><i style={{ width: `${bar.used}%` }} /></div>
      <div className="cs-bar-legend">
        <span>{usage.hours_used.toFixed(1)}h used</span>
        <span>{usage.hours_total.toFixed(1)}h total</span>
      </div>
      <div className="cs-grid">
        <div><div className="cs-kv-label">Used this month</div><div className="cs-kv-value">{usage.hours_used.toFixed(2)} h</div></div>
        <div><div className="cs-kv-label">Allowance</div><div className="cs-kv-value">{usage.hours_total.toFixed(1)} h</div></div>
        <div><div className="cs-kv-label">Used</div><div className="cs-kv-value">{usage.percent_used.toFixed(1)}%</div></div>
        <div><div className="cs-kv-label">Billing month</div><div className="cs-kv-value">{usage.month}</div></div>
      </div>
      <Row label="State" value={STATE_TEXT[usage.state] || usage.state} strong />
      <Row label="Agent using it" value={usage.agent_using ? 'Yes — a task is running' : 'No'} />
      <Row label="Machine" value={`${usage.machine} · ${usage.cores} cores`} />
      {usage.name && <Row label="Codespace" value={usage.name} />}
      {status && status.agent_connected === false && usage.state === 'ON' && (
        <p className="muted small">The codespace is running but its agent has not attached yet; commands will wait for it.</p>
      )}
      <p className="muted small">
        Hours are measured from the intervals this app observed the codespace running ({usage.hours_basis}).
        {usage.usage_source.includes('measured') ? '' : ` Source: ${usage.usage_source}.`}
      </p>
    </>
  )
}

/** The Codespaces Usage Controller: power, live link, and the month's hours in one place. */
export function CodespacesPanel({ codespaces, onClose }: { codespaces: CodespacesView; onClose: () => void }) {
  const { status, usage, error, busy } = codespaces
  const state = status?.state || usage?.state || 'OFF'
  const canStop = state === 'ON' || state === 'ERROR'
  return (
    <Popover className="cs-panel" onClose={onClose} testId="codespaces-panel">
      <div className="cs-panel-head">
        <div>
          <div className="cs-panel-title">Codespaces</div>
          <div className="muted small">{status?.workspace || '/workspace'} · the assistant’s computer</div>
        </div>
        <span className={`cs-pill tone-${state === 'ON' ? 'ok' : state === 'ERROR' ? 'bad' : state === 'OFF' ? 'off' : 'warn'}`}>
          {STATE_TEXT[state] || state}
        </span>
      </div>

      <UsageBody usage={usage} status={status} error={error} />

      <div className="cs-actions">
        {state === 'OFF' || state === 'ERROR' ? (
          <button type="button" className="btn" onClick={() => void codespaces.start()} disabled={busy !== null} data-testid="codespaces-start">
            {busy === 'start' ? 'Starting…' : 'Start computer'}
          </button>
        ) : (
          <button type="button" className="btn" onClick={() => void codespaces.stop()} disabled={busy !== null || !canStop} data-testid="codespaces-stop">
            {busy === 'stop' ? 'Stopping…' : 'Stop computer'}
          </button>
        )}
        <button type="button" className="btn ghost" onClick={() => void codespaces.restart()} disabled={busy !== null} data-testid="codespaces-restart">
          {busy === 'restart' ? 'Restarting…' : 'Restart'}
        </button>
      </div>
      <p className="muted small">
        A task starts the computer by itself and stops it when finished. Stopping never deletes files: the disk,
        installed packages and task progress survive, so unfinished work can be resumed.
      </p>
      {usage && usage.state === 'ON' && usage.agent_using && (
        <p className="cs-note">A task is using the computer, so it will not be shut down until the task finishes.</p>
      )}
      <div className="cs-foot muted small">
        Updated {new Date(codespaces.fetchedAt || Date.now()).toLocaleTimeString()} · refreshes every few seconds
      </div>
    </Popover>
  )
}
