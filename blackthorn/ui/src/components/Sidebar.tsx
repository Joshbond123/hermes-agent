import { useEffect, useMemo, useRef, useState } from 'react'
import type { Session, VersionInfo } from '../types'
import { groupLabel } from '../util'
import { ChevronIcon, CloseIcon, DotsIcon, PlusIcon, SettingsIcon } from './Icons'
import { Menu } from './Menu'

export interface SidebarProps {
  open: boolean
  narrow: boolean
  onClose: () => void
  sessions: Session[]
  archived: Session[] | null
  loaded: boolean
  error: string | null
  currentId: string | null
  query: string
  onQuery: (q: string) => void
  onSelect: (id: string) => void
  onNew: () => void
  onRename: (id: string, title: string) => Promise<void>
  onPin: (id: string, pinned: boolean) => Promise<void>
  onArchive: (id: string, archived: boolean) => Promise<void>
  onDelete: (session: Session) => void
  onLoadArchived: () => void
  onRetry: () => void
  onSettings: () => void
  version: VersionInfo | null
}

function Item({ s, active, props }: { s: Session; active: boolean; props: SidebarProps }) {
  const [menu, setMenu] = useState(false)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(s.title)
  const input = useRef<HTMLInputElement>(null)
  useEffect(() => { if (editing) { input.current?.focus(); input.current?.select() } }, [editing])
  const commit = async () => {
    setEditing(false)
    const next = draft.trim()
    if (next && next !== s.title) await props.onRename(s.id, next)
    else setDraft(s.title)
  }
  const items = [
    { label: 'Rename', onSelect: () => { setDraft(s.title); setEditing(true) }, testId: 'menu-rename' },
    ...(s.archived ? [] : [{ label: s.pinned ? 'Unpin' : 'Pin', onSelect: () => void props.onPin(s.id, !s.pinned), testId: 'menu-pin' }]),
    { label: s.archived ? 'Unarchive' : 'Archive', onSelect: () => void props.onArchive(s.id, !s.archived), testId: 'menu-archive' },
    { label: 'Delete', onSelect: () => props.onDelete(s), danger: true, testId: 'menu-delete' },
  ]
  return (
    <li className={`s-item${active ? ' active' : ''}${menu ? ' menu-open' : ''}`} data-testid="session-item" data-session-id={s.id}>
      {editing ? (
        <input ref={input} className="s-edit" value={draft} maxLength={120} aria-label="Chat title" data-testid="rename-input"
          onChange={(e) => setDraft(e.target.value)} onBlur={commit}
          onKeyDown={(e) => { if (e.key === 'Enter') void commit(); if (e.key === 'Escape') { setEditing(false); setDraft(s.title) } }} />
      ) : (
        <button type="button" className="s-main" onClick={() => props.onSelect(s.id)} aria-current={active ? 'page' : undefined} title={s.title}>
          {s.pinned ? <span className="pin-dot" aria-label="pinned" /> : null}
          <span className="s-title" data-testid="session-title">{s.title}</span>
        </button>
      )}
      {!editing && (
        <div className="s-actions">
          <button type="button" className="icon-btn s-more" aria-label={`Actions for ${s.title}`} aria-haspopup="menu" aria-expanded={menu} data-popover-trigger data-testid="session-menu" onClick={() => setMenu((v) => !v)}><DotsIcon /></button>
          {menu && <Menu items={items} onClose={() => setMenu(false)} testId="session-menu-popover" />}
        </div>
      )}
    </li>
  )
}

export function Sidebar(props: SidebarProps) {
  const { open, narrow, sessions, archived, currentId } = props
  const [showArchived, setShowArchived] = useState(false)
  const pinned = sessions.filter((s) => s.pinned)
  const groups = useMemo(() => {
    const map = new Map<string, Session[]>()
    for (const s of sessions.filter((x) => !x.pinned)) {
      const label = groupLabel(s.updated_at)
      map.set(label, [...(map.get(label) ?? []), s])
    }
    return ['Today', 'Yesterday', 'Previous 7 days', 'Previous 30 days', 'Older'].filter((l) => map.has(l)).map((l) => [l, map.get(l)!] as const)
  }, [sessions])
  const searching = props.query.trim().length > 0

  return (
    <>
      <aside id="sidebar" className={`sidebar${open ? ' open' : ''}${narrow ? ' narrow' : ''}`} data-testid="sidebar" data-open={open} aria-label="Conversations" aria-hidden={!open} inert={!open}>
        <div className="sidebar-top">
          <button type="button" className="btn new-chat" onClick={props.onNew} data-testid="new-chat"><PlusIcon /> New chat</button>
          {narrow && <button type="button" className="icon-btn" onClick={props.onClose} aria-label="Close sidebar" data-testid="sidebar-close"><CloseIcon /></button>}
        </div>
        <div className="search">
          <input type="search" placeholder="Search chats" value={props.query} onChange={(e) => props.onQuery(e.target.value)} aria-label="Search chats" data-testid="search" />
        </div>
        <nav className="sessions" aria-label="Chat history">
          {!props.loaded && <p className="muted pad">Loading chats…</p>}
          {props.error && (
            <div className="notice error pad" role="alert">
              <span>{props.error}</span>
              <button type="button" className="link" onClick={props.onRetry}>Retry</button>
            </div>
          )}
          {props.loaded && !props.error && !sessions.length && <p className="muted pad" data-testid="no-chats">{searching ? 'No chats match your search.' : 'No conversations yet. Send a message to start one.'}</p>}
          {pinned.length > 0 && (
            <section><h3>Pinned</h3><ul>{pinned.map((s) => <Item key={s.id} s={s} active={s.id === currentId} props={props} />)}</ul></section>
          )}
          {groups.map(([label, list]) => (
            <section key={label}><h3>{label}</h3><ul>{list.map((s) => <Item key={s.id} s={s} active={s.id === currentId} props={props} />)}</ul></section>
          ))}
          {!searching && (
            <section className="archived">
              <button type="button" className="fold" aria-expanded={showArchived} data-testid="archived-toggle"
                onClick={() => { const next = !showArchived; setShowArchived(next); if (next) props.onLoadArchived() }}>
                <ChevronIcon className={showArchived ? 'rot' : ''} /> Archived
              </button>
              {showArchived && (
                <ul data-testid="archived-list">
                  {archived === null && <li className="muted pad">Loading…</li>}
                  {archived?.length === 0 && <li className="muted pad">Nothing archived.</li>}
                  {archived?.map((s) => <Item key={s.id} s={s} active={s.id === currentId} props={props} />)}
                </ul>
              )}
            </section>
          )}
        </nav>
        <div className="sidebar-foot" data-testid="sidebar-foot">
          <div className="sidebar-foot-row">
            <button type="button" className="icon-btn" onClick={props.onSettings} aria-label="Settings" data-testid="settings-open">
              <SettingsIcon />
            </button>
            {props.version && (
              <div className="muted tiny" data-testid="version-foot" title={`UI build ${props.version.ui?.hash ?? 'unknown'}`}>
                Blackthorn v{props.version.version} · {props.version.commit.slice(0, 7)}
              </div>
            )}
          </div>
        </div>
      </aside>
      {narrow && open && <div className="scrim" onClick={props.onClose} data-testid="scrim" />}
    </>
  )
}
