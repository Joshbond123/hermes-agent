import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { ChatState } from '../state'
import { ArrowDownIcon } from './Icons'
import { MessageView } from './Message'

/** Follow the stream unless the reader scrolled up; never fight a person who is reading. */
export function Chat({ chat, onRegenerate, onRetryLoad }: { chat: ChatState; onRegenerate: () => void; onRetryLoad: () => void }) {
  const scroller = useRef<HTMLDivElement>(null)
  const stick = useRef(true)
  const [away, setAway] = useState(false)

  const reduced = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
  const toBottom = useCallback((smooth = false) => {
    const el = scroller.current
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: smooth && !reduced ? 'smooth' : 'auto' })
  }, [reduced])
  const onScroll = () => {
    const el = scroller.current
    if (!el) return
    const near = el.scrollHeight - el.scrollTop - el.clientHeight < 120
    stick.current = near
    setAway(!near)
  }
  // Follow the stream. Runs after every paint of new tokens without a layout thrash: the scroll is
  // instant (not smooth) while streaming so text never appears to slide under the reader.
  useLayoutEffect(() => { if (stick.current) toBottom() })
  useEffect(() => { stick.current = true; setAway(false); toBottom() }, [chat.sessionId, toBottom])
  // Leaving the tab and coming back should re-anchor to the live end of the conversation.
  useEffect(() => {
    const onVisible = () => { if (document.visibilityState === 'visible' && stick.current) toBottom() }
    document.addEventListener('visibilitychange', onVisible)
    return () => document.removeEventListener('visibilitychange', onVisible)
  }, [toBottom])

  const last = chat.messages[chat.messages.length - 1]
  const live = chat.run !== null
  return (
    <div className="chat" ref={scroller} onScroll={onScroll} data-testid="chat-scroll">
      <div className="thread" role="log" aria-live="polite" aria-relevant="additions text" data-testid="thread">
        {chat.loading && (
          <div className="skeleton" data-testid="loading-chat" aria-label="Loading conversation">
            <div className="sk-line w40" /><div className="sk-block" /><div className="sk-line w70" /><div className="sk-line w55" />
          </div>
        )}
        {chat.loadError && (
          <div className="notice error center-box" role="alert">
            <span>{chat.loadError}</span> <button type="button" className="link" onClick={onRetryLoad}>Retry</button>
          </div>
        )}
        {!chat.loading && !chat.loadError && chat.messages.length === 0 && (
          <div className="empty" data-testid="empty-state">
            <h2>How can I help?</h2>
            <p className="muted">Ask for something that takes work — a build, a fix, an investigation. The agent plans it, runs it on its computer and keeps going until it is done.</p>
          </div>
        )}
        {chat.messages.map((m) => (
          <MessageView
            key={m.id}
            m={m}
            live={chat.run?.assistantId === m.id}
            phase={chat.run?.assistantId === m.id ? chat.run.phase : null}
            isLast={m === last}
            busy={live}
            onRegenerate={onRegenerate}
          />
        ))}
      </div>
      {away && live && (
        <button type="button" className="jump" onClick={() => { stick.current = true; toBottom(true); setAway(false) }} data-testid="jump-latest">
          <ArrowDownIcon /> Latest
        </button>
      )}
    </div>
  )
}
