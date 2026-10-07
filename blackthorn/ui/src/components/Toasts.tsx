import { CloseIcon } from './Icons'

export interface Toast { id: number; kind: 'info' | 'error'; text: string }

export function Toasts({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: number) => void }) {
  return (
    <div className="toasts" aria-live="polite" data-testid="toasts">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`} role={t.kind === 'error' ? 'alert' : 'status'} data-testid="toast">
          <span>{t.text}</span>
          <button type="button" className="icon-btn" aria-label="Dismiss" onClick={() => onDismiss(t.id)}><CloseIcon width={14} height={14} /></button>
        </div>
      ))}
    </div>
  )
}
