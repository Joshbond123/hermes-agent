import { useCallback, useEffect, useReducer, useRef } from "react";
import { useLatest } from "./useLatest";
import { ApiError, btApi, runTurn, type StartBody } from "../api";
import { chatReducer, fromApiMessage, initialChatState, isBusy, type ChatState } from "../store";
import type { StreamEvent } from "../types";

interface Options {
  /** Session id from the URL (/chat/:id) or null for a new chat. */
  routeSessionId: string | null;
  /** The server told us the id of a conversation we just started — put it in the URL. */
  onSessionAssigned: (id: string) => void;
  /** History changed (new chat, finished answer): re-read the list. */
  onHistoryChanged: () => void;
  onError: (message: string) => void;
}

export interface ChatApi {
  state: ChatState;
  busy: boolean;
  send: (text: string, opts?: { thinking?: boolean; attachments?: Array<{ path: string; name: string }> }) => Promise<void>;
  stop: () => Promise<void>;
  regenerate: (opts?: { thinking?: boolean }) => Promise<void>;
  reload: () => void;
}

let counter = 0;
const nextKey = (prefix: string) => `${prefix}${Date.now().toString(36)}${(counter += 1)}`;

export function useChat({ routeSessionId, onSessionAssigned, onHistoryChanged, onError }: Options): ChatApi {
  const [state, dispatch] = useReducer(chatReducer, initialChatState);
  const stateRef = useLatest(state);
  const abortRef = useRef<AbortController | null>(null);
  const queueRef = useRef<StreamEvent[]>([]);
  const rafRef = useRef<number | null>(null);
  const cb = useLatest({ onSessionAssigned, onHistoryChanged, onError });
  const loadToken = useRef(0);

  // --- event batching: one dispatch per animation frame (same events, same order, fewer renders)
  const flush = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    const events = queueRef.current;
    if (!events.length) return;
    queueRef.current = [];
    dispatch({ type: "events", events });
    let announced = false;
    for (const ev of events) {
      if (!announced && ev.type === "turn" && typeof ev.data.session_id === "string" && !stateRef.current.activeId) {
        announced = true;
        cb.current.onSessionAssigned(ev.data.session_id);
      }
      if (ev.type === "turn" && ev.data.state === "ready" && ev.data.created) cb.current.onHistoryChanged();
      if (ev.type === "done") cb.current.onHistoryChanged();
    }
  }, [cb, stateRef]);

  const enqueue = useCallback(
    (events: StreamEvent[]) => {
      queueRef.current.push(...events);
      if (events.some((e) => e.type === "done" || e.type === "error")) return flush();
      if (rafRef.current === null) rafRef.current = requestAnimationFrame(flush);
    },
    [flush],
  );

  const drive = useCallback(
    async (opts: { start?: StartBody; attach?: { turnId: string; after: number } }) => {
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;
      const result = await runTurn({
        ...opts,
        signal: ac.signal,
        onEvents: enqueue,
        onReconnecting: (attempt) => dispatch({ type: "reconnecting", attempt }),
      });
      flush();
      if (abortRef.current !== ac) return; // superseded (the user opened another chat)
      abortRef.current = null;
      if (!result.done) {
        if (result.aborted) dispatch({ type: "streamEnded", error: null });
        else {
          const detail = result.error instanceof ApiError ? result.error.message : result.error?.message;
          dispatch({
            type: "streamEnded",
            error: { code: "connection_lost", retryable: true, action: "retry",
                     message: detail ? `Connection problem: ${detail}` : "The connection was lost before the answer finished." },
          });
        }
      }
    },
    [enqueue, flush],
  );

  // --- open a stored conversation, re-attaching to a turn that is still running on the server
  const loadAbort = useRef<AbortController | null>(null);
  const loadConversation = useCallback(
    (target: string | null) => {
      abortRef.current?.abort(); // detach only; a running turn keeps going on the server
      abortRef.current = null;
      loadAbort.current?.abort();
      queueRef.current = [];
      dispatch({ type: "open", id: target });
      if (!target) return;
      const token = (loadToken.current += 1);
      const ac = new AbortController();
      loadAbort.current = ac;
      (async () => {
        try {
          const [detail, msgs] = await Promise.all([btApi.getSession(target, ac.signal), btApi.messages(target, ac.signal)]);
          if (token !== loadToken.current) return;
          dispatch({ type: "loaded", id: target, title: detail.session.title, messages: msgs.messages.map(fromApiMessage) });
          if (detail.active_turn) {
            dispatch({ type: "attach", assistantKey: nextKey("a"), dropMessageId: detail.active_turn.assistant_message_id,
                       turnId: detail.active_turn.turn_id });
            void drive({ attach: { turnId: detail.active_turn.turn_id, after: 0 } });
          }
        } catch (err) {
          if (ac.signal.aborted || token !== loadToken.current) return;
          dispatch({ type: "loadFailed", id: target,
                     error: err instanceof ApiError && err.status === 404 ? "This conversation no longer exists."
                            : err instanceof Error ? err.message : "Could not load the conversation." });
        }
      })();
    },
    [drive],
  );

  useEffect(() => {
    if (routeSessionId === stateRef.current.activeId) return; // e.g. the URL just caught up with a new chat
    loadConversation(routeSessionId);
  }, [routeSessionId, loadConversation, stateRef]);

  useEffect(
    () => () => {
      abortRef.current?.abort();
      loadAbort.current?.abort();
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    },
    [],
  );

  const send = useCallback<ChatApi["send"]>(
    async (text, opts) => {
      const message = text.trim();
      if (!message || isBusy(stateRef.current)) return;
      dispatch({ type: "send", userKey: nextKey("u"), assistantKey: nextKey("a"), text: message,
                 attachments: opts?.attachments?.map((a) => a.name) });
      await drive({
        start: {
          session_id: stateRef.current.activeId, message, thinking: Boolean(opts?.thinking),
          attachments: opts?.attachments?.map((a) => ({ path: a.path, name: a.name })),
        },
      });
    },
    [drive, stateRef],
  );

  const regenerate = useCallback<ChatApi["regenerate"]>(
    async (opts) => {
      const id = stateRef.current.activeId;
      if (!id || isBusy(stateRef.current)) return;
      dispatch({ type: "regenerate", assistantKey: nextKey("a") });
      await drive({ start: { session_id: id, regenerate: true, thinking: Boolean(opts?.thinking) } });
    },
    [drive, stateRef],
  );

  const stop = useCallback(async () => {
    const turnId = stateRef.current.turn.turnId;
    if (isBusy(stateRef.current) === false) return;
    dispatch({ type: "stopping" });
    if (!turnId) {
      abortRef.current?.abort(); // the request was never accepted: nothing exists on the server to cancel
      return;
    }
    try {
      await btApi.cancelTurn(turnId);
    } catch (err) {
      cb.current.onError(err instanceof Error ? `Could not stop the answer: ${err.message}` : "Could not stop the answer.");
    }
    // The server confirms with a `done(cancelled)` event. If that never arrives, stop listening.
    const guard = abortRef.current;
    setTimeout(() => {
      if (guard && abortRef.current === guard && stateRef.current.turn.status === "stopping") guard.abort();
    }, 4000);
  }, [cb, stateRef]);

  const reload = useCallback(() => {
    const id = stateRef.current.activeId;
    if (id) loadConversation(id);
  }, [loadConversation, stateRef]);

  return { state, busy: isBusy(state), send, stop, regenerate, reload };
}
