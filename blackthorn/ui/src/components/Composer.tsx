import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { ClipIcon, CloseIcon, SendIcon, StopIcon } from './Icons'

export interface AttachedFile { name: string; content: string }
const MAX_FILES = 5
const MAX_CHARS = 200_000

export function Composer({ busy, stopping, onSend, onStop, onError, focusKey, banner }: {
  busy: boolean
  stopping: boolean
  onSend: (text: string, files: AttachedFile[]) => Promise<boolean>
  onStop: () => void
  onError: (text: string) => void
  focusKey: string
  banner?: React.ReactNode
}) {
  const [text, setText] = useState('')
  const [files, setFiles] = useState<AttachedFile[]>([])
  const area = useRef<HTMLTextAreaElement>(null)
  const picker = useRef<HTMLInputElement>(null)
  const coarse = typeof window !== 'undefined' && window.matchMedia?.('(pointer: coarse)').matches

  useLayoutEffect(() => {
    const el = area.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`
  }, [text])
  useEffect(() => { if (!coarse) area.current?.focus() }, [focusKey, coarse])

  const canSend = (text.trim().length > 0 || files.length > 0) && !busy
  const submit = useCallback(async () => {
    const message = text.trim()
    if ((!message && !files.length) || busy) return
    const snapshot = { text, files }
    setText('')
    setFiles([])
    const ok = await onSend(message || 'Please look at the attached file(s).', snapshot.files)
    if (!ok) { setText(snapshot.text); setFiles(snapshot.files) } // nothing was sent: keep the draft
  }, [text, files, busy, onSend])

  const pick = async (list: FileList | null) => {
    if (!list) return
    const next = [...files]
    for (const f of Array.from(list)) {
      if (next.length >= MAX_FILES) { onError(`You can attach up to ${MAX_FILES} files.`); break }
      try {
        const content = await f.text()
        if (content.includes('\u0000')) { onError(`${f.name} looks like a binary file; only text files can be attached.`); continue }
        if (content.length > MAX_CHARS) onError(`${f.name} is large; only the first ${MAX_CHARS.toLocaleString()} characters are used.`)
        next.push({ name: f.name, content: content.slice(0, MAX_CHARS) })
      } catch {
        onError(`Could not read ${f.name}.`)
      }
    }
    setFiles(next)
    if (picker.current) picker.current.value = ''
  }

  return (
    <div className="composer-wrap">
      {banner}
      <form className="composer" onSubmit={(e) => { e.preventDefault(); void submit() }} data-testid="composer">
        {files.length > 0 && (
          <ul className="chips" data-testid="attachments">
            {files.map((f) => (
              <li key={f.name} className="chip">{f.name}
                <button type="button" className="chip-x" aria-label={`Remove ${f.name}`} onClick={() => setFiles((l) => l.filter((x) => x !== f))}><CloseIcon width={12} height={12} /></button>
              </li>
            ))}
          </ul>
        )}
        <textarea ref={area} rows={1} value={text} placeholder="Message Blackthorn" aria-label="Message" data-testid="composer-input"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing && !coarse) { e.preventDefault(); void submit() }
          }} />
        <div className="composer-row">
          <button type="button" className="icon-btn" onClick={() => picker.current?.click()} aria-label="Attach text files" data-testid="attach"><ClipIcon /></button>
          <input ref={picker} type="file" multiple hidden onChange={(e) => void pick(e.target.files)} data-testid="file-input" />
          <span className="spacer" />
          {busy ? (
            <button type="button" className="send stop" onClick={onStop} disabled={stopping} aria-label={stopping ? 'Stopping' : 'Stop generating'} data-testid="stop"><StopIcon /></button>
          ) : (
            <button type="submit" className="send" disabled={!canSend} aria-label="Send message" data-testid="send"><SendIcon /></button>
          )}
        </div>
      </form>
    </div>
  )
}
