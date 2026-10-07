import { useEffect, useRef, type ReactNode } from 'react'

export interface MenuItem { label: string; onSelect: () => void; danger?: boolean; testId?: string }

/** Click-outside / Esc dismissing popover. Render it next to its trigger inside a `position: relative` parent. */
export function Popover({ onClose, children, className = '', testId, align = 'right' }: { onClose: () => void; children: ReactNode; className?: string; testId?: string; align?: 'left' | 'right' }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const down = (e: MouseEvent | TouchEvent) => {
      const target = e.target as Node
      const el = target as HTMLElement
      // clicks inside a modal dialog (e.g. a confirmation opened from this popover) are not "outside" clicks
      if (ref.current && !ref.current.contains(target) && !el.closest?.('[data-popover-trigger]') && !el.closest?.('[role="dialog"]')) onClose()
    }
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('mousedown', down)
    document.addEventListener('touchstart', down)
    document.addEventListener('keydown', key)
    return () => {
      document.removeEventListener('mousedown', down)
      document.removeEventListener('touchstart', down)
      document.removeEventListener('keydown', key)
    }
  }, [onClose])
  return <div ref={ref} className={`popover ${align} ${className}`} data-testid={testId}>{children}</div>
}

export function Menu({ items, onClose, testId }: { items: MenuItem[]; onClose: () => void; testId?: string }) {
  return (
    <Popover onClose={onClose} className="menu" testId={testId}>
      <div role="menu">
        {items.map((item) => (
          <button key={item.label} type="button" role="menuitem" className={`menu-item${item.danger ? ' danger' : ''}`} data-testid={item.testId}
            onClick={() => { onClose(); item.onSelect() }}>{item.label}</button>
        ))}
      </div>
    </Popover>
  )
}
