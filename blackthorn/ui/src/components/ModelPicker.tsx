import { useEffect, useRef, useState } from 'react'
import type { ModelsView } from '../models'
import { CheckIcon, ChevronIcon } from './Icons'

/**
 * The chat header's model selector.
 *
 * It behaves like the model picker inside an existing conversation: the chat stays, the transcript
 * stays, only the model that answers the *next* message changes. The trigger always names the model
 * in force, so it is never a guess which one is answering.
 */
export function ModelPicker({ models, open, onToggle, onChoose, onClose }: {
  models: ModelsView
  open: boolean
  onToggle: () => void
  onChoose: (id: string) => void
  onClose: () => void
}) {
  const wrap = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<string | null>(null)

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

  const active = models.active
  const label = active?.label ?? (models.loading ? 'Loading models…' : 'Select model')
  const unavailable = models.models.filter((m) => !m.available).length

  return (
    <div className="rel model-picker" ref={wrap}>
      <button
        type="button"
        className={`model-trigger${open ? ' open' : ''}`}
        onClick={onToggle}
        aria-haspopup="listbox"
        aria-expanded={open}
        data-testid="model-button"
        title={active ? `${active.label} · ${active.detail}` : label}
      >
        <span className={`model-dot${active?.available ? ' on' : ''}`} aria-hidden="true" />
        <span className="model-name" data-testid="model-name">{label}</span>
        <ChevronIcon className="chev" />
      </button>

      {open && (
        <div className="model-menu" role="listbox" aria-label="Models" data-testid="model-menu">
          <div className="model-menu-head">Model for this chat</div>
          {models.models.map((m) => {
            const isActive = m.id === models.activeId
            return (
              <button
                key={m.id}
                type="button"
                role="option"
                aria-selected={isActive}
                aria-disabled={!m.available}
                className={`model-item${isActive ? ' active' : ''}${m.available ? '' : ' unavailable'}`}
                onClick={() => onChoose(m.id)}
                onMouseEnter={() => setHover(m.id)}
                onMouseLeave={() => setHover(null)}
                data-testid={`model-option-${m.id}`}
              >
                <span className="model-item-main">
                  <span className="model-item-label">{m.label}</span>
                  <span className="model-item-detail">
                    {m.available ? m.detail : m.reason}
                  </span>
                </span>
                {isActive
                  ? <CheckIcon className="model-check" />
                  : !m.available
                    ? <span className="model-badge">unavailable</span>
                    : hover === m.id ? <span className="model-badge ghost">switch</span> : null}
              </button>
            )
          })}
          <div className="model-menu-foot">
            {models.switching
              ? 'Switching…'
              : unavailable
                ? `${unavailable} model${unavailable > 1 ? 's' : ''} not configured on this server.`
                : 'Switching keeps this conversation and its history.'}
          </div>
          {models.pendingNote && <div className="model-menu-note" data-testid="model-switch-note">{models.pendingNote}</div>}
          {models.error && <div className="model-menu-error" role="alert">{models.error}</div>}
        </div>
      )}
    </div>
  )
}
