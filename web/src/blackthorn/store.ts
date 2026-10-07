/**
 * Conversation state (pure reducer — no I/O, fully unit-tested).
 *
 * Streamed events are applied to the *live assistant message* only; nothing is ever
 * synthesised on the client. A status line, a tool row or a "thinking" indicator exists
 * only because the server sent the corresponding real event.
 */
import type {
  ApiMessage,
  ErrorInfo,
  FinishReason,
  Part,
  StreamEvent,
  ToolPart,
  ToolStatus,
  UiMessage,
} from "./types";

export type TurnStatus = "idle" | "connecting" | "streaming" | "reconnecting" | "stopping";

export interface TurnState {
  status: TurnStatus;
  turnId: string | null;
  lastSeq: number;
  reasoning: { active: boolean; since: number | null };
  model: string | null;
  reconnectAttempt: number;
}

export interface ChatState {
  activeId: string | null;
  title: string;
  messages: UiMessage[];
  loading: boolean;
  loadError: string | null;
  turn: TurnState;
}

export const IDLE_TURN: TurnState = {
  status: "idle",
  turnId: null,
  lastSeq: 0,
  reasoning: { active: false, since: null },
  model: null,
  reconnectAttempt: 0,
};

export const initialChatState: ChatState = {
  activeId: null,
  title: "",
  messages: [],
  loading: false,
  loadError: null,
  turn: IDLE_TURN,
};

export type ChatAction =
  | { type: "open"; id: string | null }
  | { type: "loaded"; id: string; title: string; messages: UiMessage[] }
  | { type: "loadFailed"; id: string; error: string }
  | { type: "send"; userKey: string; assistantKey: string; text: string; attachments?: string[] }
  | { type: "attach"; assistantKey: string; dropMessageId: number | null; turnId: string }
  | { type: "regenerate"; assistantKey: string }
  | { type: "events"; events: StreamEvent[]; now?: number }
  | { type: "reconnecting"; attempt: number }
  | { type: "stopping" }
  | { type: "streamEnded"; error?: ErrorInfo | null }
  | { type: "title"; title: string }
  | { type: "sessionAssigned"; id: string };

// --------------------------------------------------------------------------- //
// stored message -> UI message
// --------------------------------------------------------------------------- //
function normalizePart(raw: Record<string, unknown>): Part | null {
  if (raw.t === "text" && typeof raw.text === "string") return { t: "text", text: raw.text };
  if (raw.t === "tool") {
    const status = String(raw.status || "ok") as ToolStatus;
    return {
      t: "tool",
      id: String(raw.id ?? ""),
      name: String(raw.name ?? "tool"),
      label: String(raw.label ?? raw.name ?? "Tool"),
      status: status === "running" ? "cancelled" : status, // a stored "running" can never still be running
      args: (raw.args && typeof raw.args === "object" ? raw.args : {}) as Record<string, unknown>,
      output: typeof raw.output === "string" ? raw.output : undefined,
      durationMs: typeof raw.duration_ms === "number" ? raw.duration_ms : undefined,
      exitCode: typeof raw.exit_code === "number" ? raw.exit_code : null,
    };
  }
  return null;
}

export function statusForFinish(reason: FinishReason | undefined): UiMessage["status"] {
  switch (reason) {
    case "cancelled":
    case "interrupted":
    case "running":
      return "stopped";
    case "error":
    case "timeout":
      return "error";
    default:
      return "complete";
  }
}

export function fromApiMessage(m: ApiMessage): UiMessage {
  const parts: Part[] = [];
  if (m.parts && m.parts.length) {
    for (const raw of m.parts) {
      const p = normalizePart(raw);
      if (p) parts.push(p);
    }
  } else if (m.content) {
    parts.push({ t: "text", text: m.content });
  }
  return {
    key: `m${m.id}`,
    id: m.id,
    role: m.role,
    parts,
    status: m.role === "user" ? "complete" : statusForFinish(m.finish_reason),
    finishReason: m.finish_reason ?? null,
    error: m.error ?? null,
    thinkingMs: m.thinking_ms ?? undefined,
    durationMs: m.duration_ms ?? undefined,
    completionTokens: m.usage?.completion_tokens,
    attachments: m.attachments ?? undefined,
  };
}

// --------------------------------------------------------------------------- //
// stream events -> live message
// --------------------------------------------------------------------------- //
function lastStreamingIndex(messages: UiMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === "assistant" && messages[i].status === "streaming") return i;
  }
  return -1;
}

function appendText(parts: Part[], text: string): Part[] {
  const last = parts[parts.length - 1];
  if (last && last.t === "text") {
    return [...parts.slice(0, -1), { t: "text", text: last.text + text }];
  }
  return [...parts, { t: "text", text }];
}

function updateTool(parts: Part[], id: string, patch: Partial<ToolPart>): Part[] {
  return parts.map((p) => (p.t === "tool" && p.id === id ? { ...p, ...patch } : p));
}

export function applyEvents(state: ChatState, events: StreamEvent[], now = Date.now()): ChatState {
  let { messages, turn } = state;
  let activeId = state.activeId;
  const title = state.title;
  for (const ev of events) {
    if (ev.seq <= turn.lastSeq) continue; // duplicate (replayed after a reconnect)
    turn = { ...turn, lastSeq: ev.seq };
    const idx = lastStreamingIndex(messages);
    const msg = idx >= 0 ? messages[idx] : null;
    const set = (patch: Partial<UiMessage>) => {
      if (idx < 0) return;
      messages = messages.map((m, i) => (i === idx ? { ...m, ...patch } : m));
    };
    const d = ev.data;
    switch (ev.type) {
      case "turn": {
        if (typeof d.turn_id === "string") turn = { ...turn, turnId: d.turn_id };
        if (typeof d.model === "string" && d.model) turn = { ...turn, model: d.model };
        if (typeof d.session_id === "string" && d.session_id && !activeId) activeId = d.session_id;
        if (typeof d.assistant_message_id === "number") set({ id: d.assistant_message_id });
        if (turn.status === "connecting" || turn.status === "reconnecting") turn = { ...turn, status: "streaming" };
        break;
      }
      case "delta": {
        if (msg && typeof d.text === "string") set({ parts: appendText(msg.parts, d.text) });
        break;
      }
      case "reasoning": {
        if (d.active) {
          turn = { ...turn, reasoning: { active: true, since: now } };
        } else {
          turn = { ...turn, reasoning: { active: false, since: null } };
          if (msg && typeof d.duration_ms === "number") set({ thinkingMs: (msg.thinkingMs ?? 0) + d.duration_ms });
        }
        break;
      }
      case "tool_start": {
        if (msg) {
          const part: ToolPart = {
            t: "tool",
            id: String(d.id),
            name: String(d.name),
            label: String(d.label ?? d.name),
            status: "running",
            args: (d.args && typeof d.args === "object" ? d.args : {}) as Record<string, unknown>,
          };
          set({ parts: [...msg.parts, part] });
        }
        break;
      }
      case "tool_result": {
        if (msg) {
          set({
            parts: updateTool(msg.parts, String(d.id), {
              status: d.ok ? "ok" : "error",
              output: typeof d.output === "string" ? d.output : undefined,
              durationMs: typeof d.duration_ms === "number" ? d.duration_ms : undefined,
              exitCode: typeof d.exit_code === "number" ? d.exit_code : null,
              truncated: Boolean(d.truncated),
            }),
          });
        }
        break;
      }
      case "notice": {
        if (msg && typeof d.text === "string") set({ notices: [...(msg.notices ?? []), d.text] });
        break;
      }
      case "error": {
        set({ error: { code: String(d.code ?? "error"), message: String(d.message ?? "Something went wrong"),
                       retryable: Boolean(d.retryable), action: d.action ?? null } });
        break;
      }
      case "done": {
        const reason = (d.finish_reason ?? "stop") as FinishReason;
        const hasError = Boolean(d.error) || (msg?.error ?? null) !== null;
        // close any tool row that never reported back
        const parts = (msg?.parts ?? []).map((p) =>
          p.t === "tool" && p.status === "running" ? { ...p, status: "cancelled" as ToolStatus } : p,
        );
        set({
          parts,
          status: hasError && reason !== "cancelled" ? "error" : statusForFinish(reason),
          finishReason: reason,
          error: (d.error as ErrorInfo | null) ?? msg?.error ?? null,
          id: typeof d.message_id === "number" ? d.message_id : msg?.id,
          durationMs: typeof d.duration_ms === "number" ? d.duration_ms : undefined,
          thinkingMs: typeof d.thinking_ms === "number" ? d.thinking_ms : msg?.thinkingMs,
          completionTokens: d.usage?.completion_tokens || undefined,
        });
        if (typeof d.session_id === "string" && d.session_id && !activeId) activeId = d.session_id;
        turn = { ...turn, status: "idle", turnId: null, reasoning: { active: false, since: null }, reconnectAttempt: 0 };
        break;
      }
      default:
        break;
    }
  }
  return { ...state, messages, turn, activeId, title };
}

// --------------------------------------------------------------------------- //
export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "open":
      return { ...initialChatState, activeId: action.id, loading: Boolean(action.id) };
    case "loaded":
      if (action.id !== state.activeId) return state;
      return { ...state, messages: action.messages, title: action.title, loading: false, loadError: null };
    case "loadFailed":
      if (action.id !== state.activeId) return state;
      return { ...state, loading: false, loadError: action.error };
    case "send": {
      const user: UiMessage = {
        key: action.userKey, role: "user", status: "complete",
        parts: [{ t: "text", text: action.text }], attachments: action.attachments,
      };
      const assistant: UiMessage = { key: action.assistantKey, role: "assistant", status: "streaming", parts: [] };
      return {
        ...state,
        messages: [...state.messages, user, assistant],
        turn: { ...IDLE_TURN, status: "connecting" },
      };
    }
    case "attach": {
      const kept = state.messages.filter((m) => !(action.dropMessageId !== null && m.id === action.dropMessageId));
      const assistant: UiMessage = {
        key: action.assistantKey, id: action.dropMessageId, role: "assistant", status: "streaming", parts: [],
      };
      return { ...state, messages: [...kept, assistant], turn: { ...IDLE_TURN, status: "connecting", turnId: action.turnId } };
    }
    case "regenerate": {
      const msgs = [...state.messages];
      if (msgs.length && msgs[msgs.length - 1].role === "assistant") msgs.pop();
      const assistant: UiMessage = { key: action.assistantKey, role: "assistant", status: "streaming", parts: [] };
      return { ...state, messages: [...msgs, assistant], turn: { ...IDLE_TURN, status: "connecting" } };
    }
    case "events":
      return applyEvents(state, action.events, action.now);
    case "reconnecting":
      return { ...state, turn: { ...state.turn, status: "reconnecting", reconnectAttempt: action.attempt } };
    case "stopping":
      return state.turn.status === "idle" ? state : { ...state, turn: { ...state.turn, status: "stopping" } };
    case "streamEnded": {
      // The stream is over but no `done` event arrived (aborted or lost). Close the live message honestly.
      const idx = lastStreamingIndex(state.messages);
      if (idx < 0) return { ...state, turn: IDLE_TURN };
      const msg = state.messages[idx];
      const stopped = state.turn.status === "stopping";
      const parts = msg.parts.map((p) =>
        p.t === "tool" && p.status === "running" ? { ...p, status: "cancelled" as ToolStatus } : p,
      );
      const messages = state.messages.map((m, i) =>
        i === idx
          ? { ...m, parts, status: stopped ? ("stopped" as const) : ("error" as const),
              finishReason: stopped ? ("cancelled" as FinishReason) : ("error" as FinishReason),
              error: stopped ? null : action.error ?? { code: "connection_lost",
                message: "The connection was lost before the answer finished.", retryable: true, action: "retry" } }
          : m,
      );
      return { ...state, messages, turn: IDLE_TURN };
    }
    case "title":
      return { ...state, title: action.title };
    case "sessionAssigned":
      return state.activeId ? state : { ...state, activeId: action.id };
    default:
      return state;
  }
}

export const isBusy = (s: ChatState): boolean => s.turn.status !== "idle";

export function messageText(m: UiMessage): string {
  return m.parts
    .filter((p): p is Extract<Part, { t: "text" }> => p.t === "text")
    .map((p) => p.text)
    .join("\n\n");
}

export function exportMarkdown(title: string, messages: UiMessage[]): string {
  const lines = [`# ${title || "Blackthorn chat"}`, ""];
  for (const m of messages) {
    lines.push(`## ${m.role === "user" ? "You" : "Blackthorn"}`, "", messageText(m) || "_(no text)_", "");
    for (const p of m.parts) {
      if (p.t === "tool") lines.push(`> Tool: ${p.label} — ${p.status}`, "");
    }
  }
  return lines.join("\n");
}
