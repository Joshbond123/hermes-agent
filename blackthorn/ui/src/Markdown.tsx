import { Lexer, type Token, type Tokens } from 'marked'
import { memo, useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { highlightCode, languageLabel } from './highlight'
import { copyText, decodeEntities, safeHref } from './util'

// ---------------------------------------------------------------------------------------------------- inline
function inline(tokens: Token[] | undefined, key: string, reveal = false): ReactNode[] {
  if (!tokens) return []
  return tokens.map((t, i) => {
    const k = `${key}.${i}`
    switch (t.type) {
      case 'text': {
        const tt = t as Tokens.Text
        if (tt.tokens?.length) return <span key={k}>{inline(tt.tokens, k, reveal)}</span>
        // A line appears once when it arrives; later token updates preserve earlier DOM
        // nodes and do not restart their fade. Non-streaming Markdown stays unchanged.
        if (reveal) return tt.text.split('\n').map((line, n) => (
          <span className="stream-line" key={`${k}.line${n}`}>{n ? '\n' : ''}{decodeEntities(line)}</span>
        ))
        return decodeEntities(tt.text)
      }
      case 'escape':
        return (t as Tokens.Escape).text
      case 'strong':
        return <strong key={k}>{inline((t as Tokens.Strong).tokens, k, reveal)}</strong>
      case 'em':
        return <em key={k}>{inline((t as Tokens.Em).tokens, k, reveal)}</em>
      case 'del':
        return <del key={k}>{inline((t as Tokens.Del).tokens, k, reveal)}</del>
      case 'codespan':
        return <code key={k} className="inline-code">{decodeEntities((t as Tokens.Codespan).text)}</code>
      case 'br':
        return <br key={k} />
      case 'link': {
        const lt = t as Tokens.Link
        const href = safeHref(lt.href)
        const body = inline(lt.tokens, k, reveal)
        return href ? <a key={k} href={href} target="_blank" rel="noopener noreferrer">{body}</a> : <span key={k}>{body}</span>
      }
      case 'image': {
        const it = t as Tokens.Image
        const href = safeHref(it.href)
        return href ? <a key={k} href={href} target="_blank" rel="noopener noreferrer">{it.text || href}</a> : <span key={k}>{it.text}</span>
      }
      case 'html':
        return (t as Tokens.HTML).text // never injected as markup: shown as literal text
      default:
        return (t as { raw?: string }).raw ?? ''
    }
  })
}

// ---------------------------------------------------------------------------------------------------- code
export function CodeBlock({ code, lang }: { code: string; lang?: string | null }) {
  const [copied, setCopied] = useState(false)
  const html = useMemo(() => highlightCode(code, lang), [code, lang])
  const label = languageLabel(lang)
  const onCopy = async () => {
    if (await copyText(code)) {
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    }
  }
  return (
    <div className="code" data-testid="code-block">
      <div className="code-head">
        <span className="code-lang" data-testid="code-lang">{label}</span>
        <button type="button" className="code-copy" data-testid="copy-code" onClick={onCopy} aria-label={`Copy ${label} code`}>
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre tabIndex={0}>
        {html != null ? <code className={`hljs language-${label}`} dangerouslySetInnerHTML={{ __html: html }} /> : <code className="hljs">{code}</code>}
      </pre>
    </div>
  )
}

// ---------------------------------------------------------------------------------------------------- blocks
function blocks(tokens: Token[] | undefined, key: string, reveal = false): ReactNode[] {
  if (!tokens) return []
  const out: ReactNode[] = []
  tokens.forEach((t, i) => {
    const k = `${key}.${i}`
    switch (t.type) {
      case 'space':
        break
      case 'paragraph':
        out.push(<p key={k}>{inline((t as Tokens.Paragraph).tokens, k, reveal)}</p>)
        break
      case 'text': // tight list item content
        out.push(<span key={k} className="tight">{inline((t as Tokens.Text).tokens ?? [t], k, reveal)}</span>)
        break
      case 'heading': {
        const ht = t as Tokens.Heading
        const Tag = `h${Math.min(6, ht.depth)}` as 'h1'
        out.push(<Tag key={k}>{inline(ht.tokens, k, reveal)}</Tag>)
        break
      }
      case 'code': {
        const ct = t as Tokens.Code
        out.push(<CodeBlock key={k} code={ct.text} lang={ct.lang} />)
        break
      }
      case 'blockquote':
        out.push(<blockquote key={k}>{blocks((t as Tokens.Blockquote).tokens, k, reveal)}</blockquote>)
        break
      case 'hr':
        out.push(<hr key={k} />)
        break
      case 'list': {
        const lt = t as Tokens.List
        const items = lt.items.map((item, j) => (
          <li key={`${k}.${j}`} className={item.task ? 'task' : undefined}>
            {item.task ? <input type="checkbox" checked={!!item.checked} readOnly disabled aria-label={item.checked ? 'done' : 'not done'} /> : null}
            {blocks(item.tokens, `${k}.${j}`, reveal)}
          </li>
        ))
        const start = typeof lt.start === 'number' && lt.start !== 1 ? lt.start : undefined
        out.push(lt.ordered ? <ol key={k} start={start}>{items}</ol> : <ul key={k}>{items}</ul>)
        break
      }
      case 'table': {
        const tt = t as Tokens.Table
        const align = (a: string | null): CSSProperties | undefined => (a === 'left' || a === 'right' || a === 'center' ? { textAlign: a } : undefined)
        out.push(
          <div key={k} className="table-wrap">
            <table>
              <thead><tr>{tt.header.map((c, j) => <th key={j} style={align(c.align)}>{inline(c.tokens, `${k}.h${j}`)}</th>)}</tr></thead>
              <tbody>{tt.rows.map((row, r) => <tr key={r}>{row.map((c, j) => <td key={j} style={align(c.align)}>{inline(c.tokens, `${k}.${r}.${j}`)}</td>)}</tr>)}</tbody>
            </table>
          </div>,
        )
        break
      }
      case 'html':
        out.push(<p key={k} className="literal">{(t as Tokens.HTML).raw}</p>)
        break
      default:
        if ((t as { raw?: string }).raw?.trim()) out.push(<p key={k}>{(t as { raw: string }).raw}</p>)
    }
  })
  return out
}

const Block = memo(
  function Block({ token, index, reveal }: { token: Token; index: number; reveal?: boolean }) {
    return <>{blocks([token], `b${index}`, reveal)}</>
  },
  (a, b) => a.token.raw === b.token.raw && a.index === b.index && a.reveal === b.reveal,
)

export const Markdown = memo(function Markdown({ text, streaming = false }: { text: string; streaming?: boolean }) {
  // The SSE reader receives individual tokens. Paint at most once per 55 ms (16 ms
  // for a line break), but always flush the exact final answer when the run ends.
  const [shown, setShown] = useState(text)
  const latest = useRef(text)
  const pending = useRef<ReturnType<typeof setTimeout> | null>(null)
  latest.current = text
  useEffect(() => {
    if (!streaming) {
      if (pending.current !== null) clearTimeout(pending.current)
      pending.current = null
      setShown(text)
    } else if (pending.current === null) {
      pending.current = setTimeout(() => {
        pending.current = null
        setShown(latest.current)
      }, text.includes('\n') ? 16 : 55)
    }
  }, [text, streaming])
  useEffect(() => () => { if (pending.current !== null) clearTimeout(pending.current) }, [])
  const tokens = useMemo(() => Lexer.lex(shown, { gfm: true, breaks: false }), [shown])
  return (
    <div className={`md${streaming ? ' is-streaming' : ''}`} data-testid="markdown">
      {tokens.map((t, i) => <Block key={i} token={t} index={i} reveal={streaming} />)}
    </div>
  )
})
