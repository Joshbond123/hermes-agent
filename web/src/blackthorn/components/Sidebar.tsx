import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router";
import type { SessionsApi } from "../hooks/useSessions";
import { groupSessions } from "../groupSessions";
import type { SessionSummary } from "../types";
import { Dialog, Icon, MenuButton, Spinner, type MenuItem } from "./ui";

interface Props {
  api: SessionsApi;
  activeId: string | null;
  onNew: () => void;
  /** Called when a conversation is chosen (lets the shell close the mobile drawer). */
  onChosen: () => void;
  /** The active conversation was deleted or archived. */
  onActiveGone: () => void;
  onClose: () => void;
  showClose: boolean;
}

function RenameInput({ initial, onDone }: { initial: string; onDone: (value: string | null) => void }) {
  const [value, setValue] = useState(initial);
  const ref = useRef<HTMLInputElement>(null);
  const done = useRef(false);
  useEffect(() => {
    ref.current?.focus();
    ref.current?.select();
  }, []);
  const finish = (commit: boolean) => {
    if (done.current) return;
    done.current = true;
    onDone(commit ? value : null);
  };
  return (
    <input
      ref={ref}
      className="bt-rename-input"
      value={value}
      maxLength={120}
      aria-label="Chat title"
      onChange={(e) => setValue(e.target.value)}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          finish(true);
        } else if (e.key === "Escape") {
          e.preventDefault();
          e.stopPropagation();
          finish(false);
        }
      }}
      onBlur={() => finish(true)}
      onClick={(e) => e.preventDefault()}
    />
  );
}

function Row({
  s, active, archivedView, renaming, onRenameDone, onChosen, menu,
}: {
  s: SessionSummary; active: boolean; archivedView: boolean; renaming: boolean;
  onRenameDone: (value: string | null) => void; onChosen: () => void; menu: MenuItem[];
}) {
  return (
    <li className={`bt-row${active ? " bt-row-active" : ""}`}>
      {renaming ? (
        <div className="bt-row-link bt-row-editing">
          <RenameInput initial={s.title} onDone={onRenameDone} />
        </div>
      ) : (
        <Link to={`/chat/${s.id}`} className="bt-row-link" aria-current={active ? "page" : undefined} onClick={onChosen} title={s.title}>
          {s.pinned && !archivedView ? <Icon name="pin" size={13} className="bt-row-pin" /> : null}
          <span className="bt-row-title">{s.title}</span>
        </Link>
      )}
      {!renaming ? (
        <MenuButton items={menu} label={`Actions for ${s.title}`} className="bt-row-menu" />
      ) : null}
    </li>
  );
}

export function Sidebar({ api, activeId, onNew, onChosen, onActiveGone, onClose, showClose }: Props) {
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [toDelete, setToDelete] = useState<SessionSummary | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [showArchived, setShowArchived] = useState(false);
  const groups = useMemo(() => groupSessions(api.sessions), [api.sessions]);
  const searching = api.query.trim().length > 0;

  const menuFor = (s: SessionSummary, archivedView: boolean): MenuItem[] =>
    archivedView
      ? [
          { key: "restore", label: "Restore", icon: "archive", onSelect: () => void api.setArchived(s.id, false) },
          { key: "delete", label: "Delete", icon: "trash", danger: true, separatorBefore: true, onSelect: () => setToDelete(s) },
        ]
      : [
          { key: "rename", label: "Rename", icon: "pencil", onSelect: () => setRenamingId(s.id) },
          { key: "pin", label: s.pinned ? "Unpin" : "Pin to top", icon: "pin", onSelect: () => void api.setPinned(s.id, !s.pinned) },
          {
            key: "archive", label: "Archive", icon: "archive",
            onSelect: async () => {
              if (await api.setArchived(s.id, true) && s.id === activeId) onActiveGone();
              if (showArchived) void api.loadArchived();
            },
          },
          { key: "delete", label: "Delete", icon: "trash", danger: true, separatorBefore: true, onSelect: () => setToDelete(s) },
        ];

  const finishRename = async (s: SessionSummary, value: string | null) => {
    setRenamingId(null);
    const trimmed = (value ?? "").trim();
    if (value === null || !trimmed || trimmed === s.title) return;
    await api.rename(s.id, trimmed);
  };

  const confirmDelete = async () => {
    if (!toDelete) return;
    setDeleting(true);
    const ok = await api.remove(toDelete.id);
    setDeleting(false);
    if (ok) {
      if (toDelete.id === activeId) onActiveGone();
      setToDelete(null);
    }
  };

  const toggleArchived = () => {
    const next = !showArchived;
    setShowArchived(next);
    if (next) void api.loadArchived();
  };

  return (
    <div className="bt-sidebar-inner">
      <div className="bt-sidebar-top">
        <Link to="/chat" className="bt-brand" onClick={onNew} aria-label="Blackthorn — new chat">
          Blackthorn
        </Link>
        {showClose ? (
          <button type="button" className="bt-icon-btn" aria-label="Close sidebar" onClick={onClose}>
            <Icon name="x" />
          </button>
        ) : null}
      </div>

      <Link to="/chat" className="bt-new-chat" onClick={onNew}>
        <Icon name="plus" size={16} />
        <span>New chat</span>
      </Link>

      <label className="bt-search">
        <Icon name="search" size={15} />
        <input
          type="search"
          value={api.query}
          onChange={(e) => api.setQuery(e.target.value)}
          placeholder="Search chats"
          aria-label="Search chats"
        />
        {searching ? (
          <button type="button" className="bt-search-clear" aria-label="Clear search" onClick={() => api.setQuery("")}>
            <Icon name="x" size={13} />
          </button>
        ) : null}
      </label>

      <nav className="bt-history" aria-label="Chat history">
        {api.loading && api.sessions.length === 0 ? (
          <ul className="bt-skeleton-list" aria-busy="true" aria-label="Loading chats">
            {[78, 64, 88, 52, 70, 60].map((w, i) => (
              <li key={i} className="bt-skeleton" style={{ width: `${w}%` }} />
            ))}
          </ul>
        ) : api.error ? (
          <div className="bt-side-note" role="alert">
            <p>{api.error}</p>
            <button type="button" className="bt-btn bt-btn-small" onClick={() => void api.refresh()}>
              Try again
            </button>
          </div>
        ) : api.sessions.length === 0 ? (
          <p className="bt-side-note">{searching ? "No chats match your search." : "Your conversations will appear here."}</p>
        ) : (
          groups.map((g) => (
            <section key={g.key} className="bt-group">
              <h3>{searching ? "Results" : g.label}</h3>
              <ul>
                {g.items.map((s) => (
                  <Row
                    key={s.id} s={s} active={s.id === activeId} archivedView={false}
                    renaming={renamingId === s.id} onRenameDone={(v) => void finishRename(s, v)}
                    onChosen={onChosen} menu={menuFor(s, false)}
                  />
                ))}
              </ul>
            </section>
          ))
        )}

        {!searching ? (
          <section className="bt-group bt-archived">
            <button type="button" className="bt-group-toggle" aria-expanded={showArchived} onClick={toggleArchived}>
              <Icon name="chevronRight" size={14} className={showArchived ? "bt-rot" : ""} />
              <span>Archived</span>
              {showArchived && api.archivedLoaded ? <span className="bt-count">{api.archived.length}</span> : null}
            </button>
            {showArchived ? (
              api.archivedLoaded ? (
                api.archived.length ? (
                  <ul>
                    {api.archived.map((s) => (
                      <Row
                        key={s.id} s={s} active={s.id === activeId} archivedView renaming={false}
                        onRenameDone={() => undefined} onChosen={onChosen} menu={menuFor(s, true)}
                      />
                    ))}
                  </ul>
                ) : (
                  <p className="bt-side-note">No archived chats.</p>
                )
              ) : (
                <p className="bt-side-note"><Spinner /> Loading…</p>
              )
            ) : null}
          </section>
        ) : null}
      </nav>

      <div className="bt-sidebar-foot">
        <Link to="/sessions" className="bt-foot-link">
          <Icon name="external" size={15} />
          <span>Admin dashboard</span>
        </Link>
      </div>

      <Dialog
        open={toDelete !== null}
        onClose={() => !deleting && setToDelete(null)}
        title="Delete this chat?"
        busy={deleting}
        footer={
          <>
            <button type="button" className="bt-btn" onClick={() => setToDelete(null)} disabled={deleting}>Cancel</button>
            <button type="button" className="bt-btn bt-btn-danger" onClick={() => void confirmDelete()} disabled={deleting} data-autofocus>
              {deleting ? <Spinner /> : null} Delete
            </button>
          </>
        }
      >
        <p>
          “{toDelete?.title}” and all of its messages will be permanently removed. This cannot be undone.
        </p>
      </Dialog>
    </div>
  );
}
