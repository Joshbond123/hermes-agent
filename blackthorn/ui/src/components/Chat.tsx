import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { ChatState } from '../state'
import { ArrowDownIcon } from './Icons'
import { MessageView } from './Message'

export function Chat({ chat, onRegenerate, onRetryLoad }: { chat: ChatState; onRegenerate: () => void; onRetryLoad: () => void }) {
  const scroller = useRef<HTMLDivElement>(null)
  const stick = useRef(true)
  const [away, setAway] = useState(false)

  const toBottom = useCallback((smooth = false) => {
    const el = scroller.current
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: smooth ? 'smooth' : 'auto' })
  }, [])
  const onScroll = () => {
    const el = scroller.current
    if (!el) return
    const near = el.scrollHeight - el.scrollTop - el.clientHeight < 90
    stick.current = near
    setAway(!near)
  }
  // follow the stream unless the reader scrolled up
  useLayoutEffect(() => { if (stick.current) toBottom() })
  // opening another conversation always starts at the latest message
  useEffect(() => { stick.current = true; setAway(false); toBottom() }, [chat.sessionId, toBottom])

  const last = chat.messages[chat.messages.length - 1]
  return (
    <div className="chat" ref={scroller} onScroll={onScroll} data-testid="chat-scroll">
      <div className="thread" role="log" aria-live="polite" aria-relevant="additions text" data-testid="thread">
        {chat.loading && <p className="muted center" data-testid="loading-chat">Loading conversation…</p>}
        {chat.loadError && (
          <div className="notice error center-box" role="alert">
            <span>{chat.loadError}</span> <button type="button" className="link" onClick={onRetryLoad}>Retry</button>
          </div>
        )}
        {!chat.loading && !chat.loadError && chat.messages.length === 0 && (
          <div className="empty" data-testid="empty-state"><h2>How can I help?</h2></div>
        )}
        {chat.messages.map((m) => (
          <MessageView key={m.id} m={m} live={chat.run?.assistantId === m.id} phase={chat.run?.assistantId === m.id ? chat.run.phase : null}
            isLast={m === last} busy={chat.run !== null} onRegenerate={onRegenerate} />
        ))}
      </div>
      {away && chat.run && (
        <button type="button" className="jump" onClick={() => { stick.current = true; toBottom(true); setAway(false) }} data-testid="jump-latest"><ArrowDownIcon /> Latest</button>
      )}
    </div>
  )
}
