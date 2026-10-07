import { useCallback, useEffect, useRef, useState } from "react";
import { useLatest } from "./useLatest";
import { ApiError, btApi } from "../api";
import type { SessionSummary } from "../types";

export interface SessionsApi {
  sessions: SessionSummary[];
  archived: SessionSummary[];
  loading: boolean;
  error: string | null;
  query: string;
  setQuery: (q: string) => void;
  refresh: () => Promise<void>;
  loadArchived: () => Promise<void>;
  archivedLoaded: boolean;
  rename: (id: string, title: string) => Promise<boolean>;
  setPinned: (id: string, pinned: boolean) => Promise<boolean>;
  setArchived: (id: string, archived: boolean) => Promise<boolean>;
  remove: (id: string) => Promise<boolean>;
}

const msg = (err: unknown) => (err instanceof Error ? err.message : "Request failed");

/**
 * History. Every action is persisted by the server (D1) first-class; the UI updates optimistically
 * and rolls back — with a visible error — if the server refuses, so what you see is what is stored.
 */
export function useSessions(onError: (m: string) => void): SessionsApi {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [archived, setArchived] = useState<SessionSummary[]>([]);
  const [archivedLoaded, setArchivedLoaded] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const queryRef = useRef("");
  const seq = useRef(0);
  const err = useLatest(onError);

  const refresh = useCallback(async () => {
    const mine = (seq.current += 1);
    try {
      const res = await btApi.listSessions({ q: queryRef.current || undefined });
      if (mine !== seq.current) return;
      setSessions(res.sessions);
      setError(null);
    } catch (e) {
      if (mine !== seq.current) return;
      setError(e instanceof ApiError && e.status === 503 ? e.message : "Could not load your conversations.");
    } finally {
      if (mine === seq.current) setLoading(false);
    }
  }, []);

  const loadArchived = useCallback(async () => {
    try {
      const res = await btApi.listSessions({ archived: true });
      setArchived(res.sessions);
      setArchivedLoaded(true);
    } catch (e) {
      err.current(`Could not load archived chats: ${msg(e)}`);
    }
  }, [err]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    queryRef.current = query;
    const t = setTimeout(() => void refresh(), query ? 250 : 0);
    return () => clearTimeout(t);
  }, [query, refresh]);

  const patch = useCallback(
    async (id: string, body: Parameters<typeof btApi.patchSession>[1], optimistic: (list: SessionSummary[]) => SessionSummary[]) => {
      const before = { sessions, archived };
      setSessions((l) => optimistic(l));
      try {
        const res = await btApi.patchSession(id, body);
        const stored = res.session;
        // adopt exactly what the server stored
        setSessions((l) => (stored.archived ? l.filter((s) => s.id !== id) : l.map((s) => (s.id === id ? stored : s))));
        if (body.archived !== undefined) {
          setArchived((l) => (stored.archived ? [stored, ...l.filter((s) => s.id !== id)] : l.filter((s) => s.id !== id)));
          if (!stored.archived) void refresh();
        }
        return true;
      } catch (e) {
        setSessions(before.sessions);
        setArchived(before.archived);
        err.current(`That change was not saved: ${msg(e)}`);
        return false;
      }
    },
    [sessions, archived, refresh, err],
  );

  const rename = useCallback(
    (id: string, title: string) => patch(id, { title }, (l) => l.map((s) => (s.id === id ? { ...s, title: title.trim() || s.title } : s))),
    [patch],
  );
  const setPinned = useCallback(
    (id: string, pinned: boolean) => patch(id, { pinned }, (l) => l.map((s) => (s.id === id ? { ...s, pinned } : s))),
    [patch],
  );
  const setArchivedFlag = useCallback(
    (id: string, flag: boolean) =>
      patch(id, { archived: flag }, (l) => (flag ? l.filter((s) => s.id !== id) : l)),
    [patch],
  );
  const remove = useCallback(
    async (id: string) => {
      const before = { sessions, archived };
      setSessions((l) => l.filter((s) => s.id !== id));
      setArchived((l) => l.filter((s) => s.id !== id));
      try {
        await btApi.deleteSession(id);
        return true;
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) return true; // already gone
        setSessions(before.sessions);
        setArchived(before.archived);
        err.current(`The chat was not deleted: ${msg(e)}`);
        return false;
      }
    },
    [sessions, archived, err],
  );

  return { sessions, archived, loading, error, query, setQuery, refresh, loadArchived, archivedLoaded,
           rename, setPinned, setArchived: setArchivedFlag, remove };
}
