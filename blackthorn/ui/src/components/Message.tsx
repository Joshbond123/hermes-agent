import { memo, useEffect, useRef, useState } from 'react'
import type { Phase } from '../state'
import type { Message, Part, ToolPart } from '../types'
import { Markdown } from '../Markdown'
import { copyText, formatDuration } from '../util'
import { ChevronIcon } from './Icons'

const LABELS: Record<string, string> = {
  web_search: 'Web search', fetch_url: 'Open page', run_command: 'Terminal', read_file: 'Read file', write_file: 'Write file',
  list_files: 'List files', computer_info: 'System info', remember: 'Memory',
}
const label = (name: string) => LABELS[name] ?? name.replace(/_/g, ' ')

function Tool({ t }: { t: ToolPart }) {
  const [more, setMore] = useState(false)
  const hasDetail = Boolean(t.output || t.sources?.length || t.error)
  return (
    <li className={`tool tool-${t.status}`} data-testid="tool-row" data-tool={t.name} data-status={t.status}>
      <div className="tool-line">
        <span className="tool-dot" aria-label={t.status} />
        <span className="tool-name">{label(t.name)}</span>
        {t.args && <code className="tool-args" title={t.args}>{t.args}</code>}
        <span className="tool-result">{t.status === 'running' ? 'running…' : t.error ? t.error : t.summary}</span>
        {t.duration_ms != null && t.status !== 'running' && <span className="tool-time">{formatDuration(t.duration_ms)}</span>}
        {hasDetail && <button type="button" className="link tiny" aria-expanded={more} onClick={() => setMore((v) => !v)}>{more ? 'Hide' : 'Details'}</button>}
      </div>
      {more && (
        <div className="tool-detail">
          {t.sources && t.sources.length > 0 && (
            <ul className="sources">{t.sources.map((s) => <li key={s.url}><a href={s.url} target="_blank" rel="noopener noreferrer">{s.title}</a> <span className="muted">{s.domain}</span></li>)}</ul>
          )}
          {t.output && <pre className="tool-out">{t.output}</pre>}
        </div>
      )}
    </li>
  )
}

function Activity({ tools, live }: { tools: ToolPart[]; live: boolean }) {
  const running = tools.some((t) => t.status === 'running')
  const [open, setOpen] = useState(running || live)
  const touched = useRef(false)
  useEffect(() => { if (!touched.current) setOpen(running) }, [running])
  const failed = tools.filter((t) => t.status === 'error').length
  const total = tools.reduce((n, t) => n + (t.duration_ms ?? 0), 0)
  const names = Array.from(new Set(tools.map((t) => label(t.name)))).join(', ')
  return (
    <div className={`activity${open ? ' open' : ''}`} data-testid="activity-group" data-open={open}>
      <button type="button" className="activity-head" aria-expanded={open} onClick={() => { touched.current = true; setOpen((v) => !v) }} data-testid="activity-toggle">
        <ChevronIcon className="chev" />
        <span className="activity-title">{running ? 'Working' : 'Used tools'}</span>
        <span className="muted">{names}</span>
        <span className="muted">· {tools.length} {tools.length === 1 ? 'step' : 'steps'}{failed ? `, ${failed} failed` : ''}{total && !running ? ` · ${formatDuration(total)}` : ''}</span>
      </button>
      {open && <ul className="tools">{tools.map((t) => <Tool key={t.id} t={t} />)}</ul>}
    </div>
  )
}

function Thinking({ since }: { since: number }) {
  const [, tick] = useState(0)
  useEffect(() => { const id = setInterval(() => tick((n) => n + 1), 500); return () => clearInterval(id) }, [])
  return <div className="waiting" data-testid="thinking" role="status">Thinking <span className="muted">{Math.max(0, Math.round((Date.now() - since) / 1000))}s</span></div>
}

const Dots = () => <div className="waiting dots" data-testid="waiting" role="status" aria-label="Waiting for the model"><i /><i /><i /></div>

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
  const lastIsTool = m.parts.length > 0 && m.parts[m.parts.length - 1].type === 'tool'
  const err = m.error ?? m.meta?.error
  const retryable = m.status === 'error' || m.status === 'interrupted'
  return (
    <article className="msg assistant" data-testid="message" data-role="assistant" data-status={m.status} data-live={live}>
      <div className="msg-label">Blackthorn</div>
      <div className="msg-body">
        {renderParts(m.parts, live)}
        {live && m.thinkingSince ? <Thinking since={m.thinkingSince} /> : null}
        {live && !m.thinkingSince && (phase === 'connecting' || phase === 'waiting') && (!m.parts.length || lastIsTool) ? <Dots /> : null}
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
