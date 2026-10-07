import { useEffect, useRef, type ReactNode } from 'react'
import { CloseIcon } from './Icons'

/** Accessible modal: Esc / backdrop closes, focus moves in and is restored, Tab stays inside. */
export function Dialog({ title, onClose, children, testId, wide = false }: { title: string; onClose: () => void; children: ReactNode; testId?: string; wide?: boolean }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null
    const node = ref.current
    const focusable = () => Array.from(node?.querySelectorAll<HTMLElement>('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])') ?? []).filter((el) => !el.hasAttribute('disabled'))
    ;(node?.querySelector<HTMLElement>('[data-autofocus]') ?? focusable()[0])?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation()
        onClose()
      } else if (e.key === 'Tab') {
        const items = focusable()
        if (!items.length) return
        const first = items[0]
        const last = items[items.length - 1]
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus() }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus() }
      }
    }
    document.addEventListener('keydown', onKey, true)
    return () => {
      document.removeEventListener('keydown', onKey, true)
      previous?.focus?.()
    }
  }, [onClose])
  return (
    <div className="backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }} data-testid="backdrop">
      <div className={`dialog${wide ? ' wide' : ''}`} role="dialog" aria-modal="true" aria-label={title} ref={ref} data-testid={testId}>
        <div className="dialog-head">
          <h2>{title}</h2>
          <button type="button" className="icon-btn" onClick={onClose} aria-label="Close"><CloseIcon /></button>
        </div>
        {children}
      </div>
    </div>
  )
}

export function Confirm({ title, message, confirmLabel, danger = false, busy = false, onConfirm, onCancel }: { title: string; message: string; confirmLabel: string; danger?: boolean; busy?: boolean; onConfirm: () => void; onCancel: () => void }) {
  return (
    <Dialog title={title} onClose={onCancel} testId="confirm-dialog">
      <p className="dialog-text">{message}</p>
      <div className="dialog-actions">
        <button type="button" className="btn" onClick={onCancel} data-autofocus>Cancel</button>
        <button type="button" className={`btn ${danger ? 'danger' : 'primary'}`} onClick={onConfirm} disabled={busy} data-testid="confirm-yes">{busy ? 'Working…' : confirmLabel}</button>
      </div>
    </Dialog>
  )
}
