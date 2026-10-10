import { memo, useEffect, useState } from 'react'
import type { Phase } from '../state'
import type { Message, Part, ToolPart } from '../types'
import { Markdown } from '../Markdown'
import { copyText, formatDuration, formatElapsed } from '../util'
import { ChevronIcon } from './Icons'

const LABELS: Record<string, string> = {
  web_search: 'Web search', fetch_url: 'Open page', run_command: 'Terminal', read_file: 'Read file', write_file: 'Write file',
  list_files: 'List files', computer_info: 'System info', remember: 'Memory',
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

function Tool({ t }: { t: ToolPart }) {
  const [more, setMore] = useState(false)
  const running = t.status === 'running'
  useTick(running)
  const liveMs = running && t.startedAt ? Date.now() - t.startedAt : null
  const hasDetail = Boolean(t.output || t.sources?.length || t.error || t.answer)
  return (
    <li className={`tool tool-${t.status}`} data-testid="tool-row" data-tool={t.name} data-status={t.status}>
      <div className="tool-line">
        <span className="tool-dot" aria-label={t.status} />
        <span className="tool-name">{label(t.name)}</span>
        {t.args && <code className="tool-args" title={t.args}>{t.args}</code>}
        <span className="tool-result">{running ? (liveMs != null ? `running… ${formatElapsed(liveMs / 1000)}` : 'running…') : t.error ? t.error : t.summary}</span>
        {t.duration_ms != null && !running && <span className="tool-time">{formatDuration(t.duration_ms)}</span>}
        {hasDetail && <button type="button" className="link tiny" aria-expanded={more} onClick={() => setMore((v) => !v)}>{more ? 'Hide' : 'Details'}</button>}
      </div>
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
    </li>
  )
}

function Activity({ tools, live }: { tools: ToolPart[]; live: boolean }) {
  // Collapsed by default (Replit-style): show a one-line status; expand only on click.
  const running = tools.some((t) => t.status === 'running')
  const [open, setOpen] = useState(false)
  const failed = tools.filter((t) => t.status === 'error').length
  const total = tools.reduce((n, t) => n + (t.duration_ms ?? 0), 0)
  const names = Array.from(new Set(tools.map((t) => label(t.name)))).join(', ')
  return (
    <div className={`activity${open ? ' open' : ''}`} data-testid="activity-group" data-open={open}>
      <button type="button" className="activity-head" aria-expanded={open} onClick={() => setOpen((v) => !v)} data-testid="activity-toggle">
        <ChevronIcon className="chev" />
        <span className="activity-title">{running || live ? 'Working' : 'Activity'}</span>
        <span className="muted">{names || (live ? 'preparing…' : '')}</span>
        <span className="muted">· {tools.length} {tools.length === 1 ? 'step' : 'steps'}{failed ? `, ${failed} failed` : ''}{total && !running ? ` · ${formatDuration(total)}` : ''}{running ? ' · running' : ''}</span>
      </button>
      {open && <ul className="tools">{tools.map((t) => <Tool key={t.id} t={t} />)}</ul>}
    </div>
  )
}

/**
 * The always-visible progress line for a live run: what the agent is doing right now and
 * for how long. Serious agent UIs never leave a blank idle state while work is happening.
 */
function StatusLine({ phase, since, tools, steps }: { phase: Phase | null; since: number; tools: ToolPart[]; steps: number }) {
  useTick(true)
  const elapsed = Math.max(0, (Date.now() - since) / 1000)
  const running = tools.filter((t) => t.status === 'running')
  let text: string
  if (phase === 'connecting') text = 'Connecting to the agent…'
  else if (running.length > 0) {
    const names = running.map((t) => label(t.name)).join(', ')
    const oldest = running.reduce((n, t) => Math.max(n, t.startedAt ?? since * 1000), 0)
    text = `Working — ${names} · ${formatElapsed((Date.now() - oldest) / 1000)}`
  } else if (phase === 'thinking') text = `Thinking · ${formatElapsed(elapsed)}`
  else text = `Working · ${formatElapsed(elapsed)}`
  return (
    <div className="status-line" data-testid="status-line" role="status" data-phase={phase ?? ''}>
      <span className="status-pulse" aria-hidden="true" />
      <span className="status-text">{text}</span>
      {steps > 0 && <span className="muted tiny">step {steps}{tools.length ? ` · ${tools.length} tool call${tools.length > 1 ? 's' : ''}` : ''}</span>}
    </div>
  )
}

function renderParts(parts: Part[], live: boolean) {
  const out: React.ReactNode[] = []
  let tools: ToolPart[] = []
  const flush = (key: string) => { if (tools.length) { out.push(<Activity key={`a-${key}`} tools={tools} live={live} />); tools = [] } }
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
        <div className="msg-label">You</div>
        <div className="msg-body user-text">{text}</div>
        {m.attachments?.length ? <ul className="chips">{m.attachments.map((a) => <li key={a.name} className="chip" title={a.workspace_path ?? a.name}>{a.name}</li>)}</ul> : null}
        <div className="msg-actions"><button type="button" className="link tiny" onClick={copy} data-testid="copy-message">{copied ? 'Copied' : 'Copy'}</button></div>
      </article>
    )
  }

  const hasText = m.parts.some((p) => p.type === 'text' && p.text.trim())
  const err = m.error ?? m.meta?.error
  const retryable = m.status === 'error' || m.status === 'interrupted'
  return (
    <article className="msg assistant" data-testid="message" data-role="assistant" data-status={m.status} data-live={live}>
      <div className="msg-label">Blackthorn</div>
      <div className="msg-body">
        {renderParts(m.parts, live)}
        {live ? (
          <div className="activity live-status" data-testid="thinking" role="status">
            <div className="activity-head static">
              <span className="activity-title">{m.thinkingSince ? 'Working' : (phase === 'tool' ? 'Using tools' : 'Responding')}</span>
              <span className="muted">
                {m.thinkingSince
                  ? `${Math.max(0, Math.round((Date.now() - m.thinkingSince) / 1000))}s`
                  : null}
              </span>
            </div>
            <StatusLine
              phase={phase}
              since={(m.created_at ?? Date.now() / 1000) * 1000}
              tools={m.parts.filter((p): p is ToolPart => p.type === 'tool')}
              steps={m.parts.filter((p): p is ToolPart => p.type === 'tool').reduce((n, t) => Math.max(n, t.step ?? 0), 0)}
            />
          </div>
        ) : null}
        {m.notices?.map((n, i) => <div key={i} className={`notice ${n.level}`} data-testid="notice">{n.text}</div>)}
        {m.status === 'cancelled' && <div className="status-note" data-testid="stopped-note">{hasText ? 'Stopped.' : 'Stopped before anything was written.'}</div>}
        {m.status === 'length' && <div className="notice warn">The answer reached the model's length limit and was cut off.</div>}
        {m.status === 'interrupted' && <div className="notice warn" data-testid="interrupted-note">This response was interrupted{err?.message && err.code === 'connection_lost' ? `: ${err.message}` : ''}. {hasText ? 'What was written so far is kept above.' : ''}</div>}
        {m.status === 'error' && <div className="notice error" role="alert" data-testid="error-note">{err?.message ?? 'The response failed.'}</div>}
      </div>
      {!live && (
        <div className="msg-actions">
          {text && <button type="button" className="link tiny" onClick={copy} data-testid="copy-message">{copied ? 'Copied' : 'Copy'}</button>}
          {isLast && !busy && (
            <button type="button" className="link tiny" onClick={onRegenerate} data-testid="regenerate">{retryable ? 'Retry' : 'Regenerate'}</button>
          )}
          {m.meta?.duration_ms != null && m.status === 'stop' && <span className="muted tiny meta">{formatDuration(m.meta.duration_ms)}{m.meta.tool_calls ? ` · ${m.meta.tool_calls} tool call${m.meta.tool_calls > 1 ? 's' : ''}` : ''}</span>}
        </div>
      )}
    </article>
  )
})
