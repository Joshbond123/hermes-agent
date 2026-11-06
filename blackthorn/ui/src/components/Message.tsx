import { memo, useEffect, useState } from 'react'
import type { Phase } from '../state'
import type { Message, Part, ToolPart } from '../types'
import { Markdown } from '../Markdown'
import { copyText, formatDuration, formatElapsed } from '../util'
import { ChevronIcon } from './Icons'

/**
 * Message rendering.
 *
 * Three visual layers, deliberately distinct, because "what the assistant said" and "what the agent
 * did" are different things:
 *
 *   prose        the answer itself — generous measure, soft entrance, a caret while it streams
 *   trajectory   the tool steps — a rail of compact cards, running steps shimmer, failures are red
 *   status       one sticky line saying what is happening right now and for how long
 */

const LABELS: Record<string, string> = {
  web_search: 'Web search', fetch_url: 'Open page', run_command: 'Terminal', read_file: 'Read file', write_file: 'Write file',
  list_files: 'List files', computer_info: 'System info', remember: 'Memory', skill_search: 'Search skills',
  skill_load: 'Load skill', skill_read: 'Read skill', tool_search: 'Search tools', cyber_run: 'Run tool',
}
const label = (name: string) => LABELS[name] ?? name.replace(/_/g, ' ')

/** Re-render every second while `active` so live elapsed timers tick. */
function useTick(active: boolean) {
  const [, setN] = useState(0)
  useEffect(() => {
    if (!active) return
    const id = setInterval(() => setN((n) => n + 1), 1000)
    return () => clearInterval(id)
  }, [active])
}

/** One step in the trajectory rail. */
function Tool({ t }: { t: ToolPart }) {
  const [more, setMore] = useState(false)
  const running = t.status === 'running'
  useTick(running)
  const liveMs = running && t.startedAt ? Date.now() - t.startedAt : null
  const hasDetail = Boolean(t.output || t.sources?.length || t.error || t.answer)
  return (
    <li className={`tool tool-${t.status}${running ? ' is-running' : ''}`} data-testid="tool-row" data-tool={t.name} data-status={t.status}>
      <span className="tool-marker" aria-hidden="true" />
      <div className="tool-body">
        <div className="tool-line">
          <span className="tool-name">{label(t.name)}</span>
          {t.args && <code className="tool-args" title={t.args}>{t.args}</code>}
          <span className="tool-result">
            {running
              ? (liveMs != null ? `running ${formatElapsed(liveMs / 1000)}` : 'running')
              : t.error ? t.error : t.summary}
          </span>
          {t.duration_ms != null && !running && <span className="tool-time">{formatDuration(t.duration_ms)}</span>}
          {hasDetail && (
            <button type="button" className="link tiny tool-more" aria-expanded={more} onClick={() => setMore((v) => !v)}>
              {more ? 'Hide' : 'Details'}
            </button>
          )}
        </div>
        {running && <span className="tool-shimmer" aria-hidden="true" />}
        {more && (
          <div className="tool-detail">
            {t.answer && (
              <div className="tool-answer" data-testid="tool-answer">
                <span className="tool-answer-label">Found</span> {t.answer}
              </div>
            )}
            {t.sources && t.sources.length > 0 && (
              <ul className="sources">{t.sources.map((s) => (
                <li key={s.url}>
                  <a href={s.url} target="_blank" rel="noopener noreferrer">{s.title}</a>
                  <span className="muted">{s.domain}</span>
                  {s.snippet && <div className="muted tiny">{s.snippet}</div>}
                </li>
              ))}</ul>
            )}
            {t.output && <pre className="tool-out">{t.output}</pre>}
          </div>
        )}
      </div>
    </li>
  )
}

/**
 * The trajectory: every step the agent took, in order, as one compact rail.
 *
 * Expanded while work is happening (a person should see progress without clicking), collapsing to a
 * single summary line once the turn is finished — the transcript stays readable, the record stays
 * complete.
 */
function Trajectory({ tools, live }: { tools: ToolPart[]; live: boolean }) {
  const running = tools.some((t) => t.status === 'running')
  const [open, setOpen] = useState(true)
  const [touched, setTouched] = useState(false)
  // Auto-collapse a finished trajectory, unless the reader opened it themselves.
  useEffect(() => { if (!live && !running && !touched) setOpen(false) }, [live, running, touched])
  useEffect(() => { if (running && !touched) setOpen(true) }, [running, touched])

  if (!tools.length) return null
  const failed = tools.filter((t) => t.status === 'error').length
  const total = tools.reduce((n, t) => n + (t.duration_ms ?? 0), 0)
  const names = Array.from(new Set(tools.map((t) => label(t.name))))
  return (
    <div className={`activity${open ? ' open' : ''}${running ? ' running' : ''}`} data-testid="activity-group" data-open={open}>
      <button
        type="button"
        className="activity-head"
        aria-expanded={open}
        onClick={() => { setTouched(true); setOpen((v) => !v) }}
        data-testid="activity-toggle"
      >
        <ChevronIcon className="chev" />
        <span className="activity-title">{running ? 'Working' : 'Worked'}</span>
        <span className="activity-steps">{tools.length} {tools.length === 1 ? 'step' : 'steps'}</span>
        <span className="activity-names">{names.slice(0, 3).join(', ')}{names.length > 3 ? ` +${names.length - 3}` : ''}</span>
        {failed > 0 && <span className="activity-bad">{failed} failed</span>}
        {total > 0 && !running && <span className="activity-time">{formatDuration(total)}</span>}
      </button>
      {open && <ul className="tools">{tools.map((t) => <Tool key={t.id} t={t} />)}</ul>}
    </div>
  )
}

/** The always-visible progress line for a live run: what is happening now, and for how long. */
function StatusLine({ phase, since, tools }: { phase: Phase | null; since: number; tools: ToolPart[] }) {
  useTick(true)
  const elapsed = Math.max(0, (Date.now() - since) / 1000)
  const running = tools.filter((t) => t.status === 'running')
  let text: string
  if (phase === 'connecting') text = 'Connecting…'
  else if (running.length > 0) {
    const names = running.map((t) => label(t.name)).join(', ')
    const oldest = running.reduce((n, t) => Math.max(n, t.startedAt ?? since * 1000), 0)
    text = `${names} · ${formatElapsed((Date.now() - oldest) / 1000)}`
  } else if (phase === 'thinking') text = `Thinking · ${formatElapsed(elapsed)}`
  else if (phase === 'streaming') text = `Writing · ${formatElapsed(elapsed)}`
  else text = `Working · ${formatElapsed(elapsed)}`
  return (
    <div className="status-line" data-testid="status-line" role="status" data-phase={phase ?? ''}>
      <span className="status-dots" aria-hidden="true"><i /><i /><i /></span>
      <span className="status-text">{text}</span>
    </div>
  )
}

function renderParts(parts: Part[], live: boolean) {
  const out: React.ReactNode[] = []
  let tools: ToolPart[] = []
  const flush = (key: string) => {
    if (tools.length) { out.push(<Trajectory key={`a-${key}`} tools={tools} live={live} />); tools = [] }
  }
  let lastText = -1
  parts.forEach((p, i) => { if (p.type === 'text' && p.text.trim()) lastText = i })
  parts.forEach((p, i) => {
    if (p.type === 'tool') { tools.push(p); return }
    flush(String(i))
    if (p.text.trim()) out.push(<Markdown key={`t-${i}`} text={p.text} streaming={live && i === lastText} />)
  })
  flush('end')
  return out
}

interface Props {
  m: Message
  live: boolean
  phase: Phase | null
  isLast: boolean
  busy: boolean
  onRegenerate: () => void
}

export const MessageView = memo(function MessageView({ m, live, phase, isLast, busy, onRegenerate }: Props) {
  const [copied, setCopied] = useState(false)
  const text = m.content || m.parts.filter((p): p is { type: 'text'; text: string } => p.type === 'text').map((p) => p.text).join('\n\n')
  const copy = async () => { if (await copyText(text)) { setCopied(true); setTimeout(() => setCopied(false), 1500) } }

  if (m.role === 'user') {
    return (
      <article className="msg user" data-testid="message" data-role="user">
        <div className="msg-body user-text">{text}</div>
        {m.attachments?.length ? <ul className="chips">{m.attachments.map((a) => <li key={a.name} className="chip" title={a.workspace_path ?? a.name}>{a.name}</li>)}</ul> : null}
        <div className="msg-actions"><button type="button" className="link tiny" onClick={copy} data-testid="copy-message">{copied ? 'Copied' : 'Copy'}</button></div>
      </article>
    )
  }

  const hasText = m.parts.some((p) => p.type === 'text' && p.text.trim())
  const err = m.error ?? m.meta?.error
  const retryable = m.status === 'error' || m.status === 'interrupted'
  const tools = m.parts.filter((p): p is ToolPart => p.type === 'tool')
  return (
    <article className="msg assistant" data-testid="message" data-role="assistant" data-status={m.status} data-live={live}>
      <div className="msg-mark" aria-hidden="true">
        <span className={`msg-avatar${live ? ' live' : ''}`} data-testid="thinking" />
      </div>
      <div className="msg-main">
        <div className="msg-body">
          {renderParts(m.parts, live)}
          {live && (
            <div className="live-status" data-testid="live-status">
              <StatusLine phase={phase} since={(m.created_at ?? Date.now() / 1000) * 1000} tools={tools} />
            </div>
          )}
        </div>
        {m.notices?.map((n, i) => <div key={i} className={`notice ${n.level}`} data-testid="notice">{n.text}</div>)}
        {m.status === 'cancelled' && <div className="status-note" data-testid="stopped-note">{hasText ? 'Stopped.' : 'Stopped before anything was written.'}</div>}
        {m.status === 'length' && <div className="notice warn">The answer reached the model's length limit and was cut off.</div>}
        {m.status === 'interrupted' && <div className="notice warn" data-testid="interrupted-note">This response was interrupted{err?.message && err.code === 'connection_lost' ? `: ${err.message}` : ''}. {hasText ? 'What was written so far is kept above.' : ''}</div>}
        {m.status === 'error' && <div className="notice error" role="alert" data-testid="error-note">{err?.message ?? 'The response failed.'}</div>}
        {!live && (
          <div className="msg-actions">
            {text && <button type="button" className="link tiny" onClick={copy} data-testid="copy-message">{copied ? 'Copied' : 'Copy'}</button>}
            {isLast && !busy && (
              <button type="button" className="link tiny" onClick={onRegenerate} data-testid="regenerate">{retryable ? 'Retry' : 'Regenerate'}</button>
            )}
            {m.meta?.duration_ms != null && m.status === 'stop' && (
              <span className="muted tiny meta">
                {m.meta.model ? `${m.meta.model.split('/').pop()} · ` : ''}
                {formatDuration(m.meta.duration_ms)}{m.meta.tool_calls ? ` · ${m.meta.tool_calls} tool call${m.meta.tool_calls > 1 ? 's' : ''}` : ''}
              </span>
            )}
          </div>
        )}
      </div>
    </article>
  )
})
