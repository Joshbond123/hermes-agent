import { describe, expect, it } from "vitest";
import {
  chatReducer,
  exportMarkdown,
  fromApiMessage,
  initialChatState,
  isBusy,
  messageText,
  type ChatState,
} from "./store";
import type { ApiMessage, StreamEvent } from "./types";

let seq = 0;
const ev = (type: string, data: Record<string, unknown> = {}): StreamEvent => ({ seq: ++seq, type, data });
const start = (): ChatState => {
  seq = 0;
  return chatReducer(initialChatState, { type: "send", userKey: "u1", assistantKey: "a1", text: "hi" });
};
const last = (s: ChatState) => s.messages[s.messages.length - 1];

describe("send + stream", () => {
  it("adds the user message and a streaming assistant placeholder", () => {
    const s = start();
    expect(s.messages.map((m) => [m.role, m.status])).toEqual([["user", "complete"], ["assistant", "streaming"]]);
    expect(s.turn.status).toBe("connecting");
    expect(isBusy(s)).toBe(true);
  });

  it("builds one continuous message: text, tool, text — in event order", () => {
    let s = start();
    s = chatReducer(s, { type: "events", events: [
      ev("turn", { state: "accepted", turn_id: "turn-1", session_id: "studio-9" }),
      ev("turn", { state: "ready", assistant_message_id: 42, model: "qwen" }),
      ev("delta", { text: "Checking " }), ev("delta", { text: "disk. " }),
      ev("tool_start", { id: "c1", name: "terminal", label: "Terminal", args: { command: "df -h" } }),
      ev("tool_result", { id: "c1", ok: true, duration_ms: 120, output: "19G free", exit_code: 0 }),
      ev("delta", { text: "You have 19G." }),
    ] });
    expect(s.activeId).toBe("studio-9");
    expect(s.turn).toMatchObject({ status: "streaming", turnId: "turn-1", model: "qwen" });
    const m = last(s);
    expect(m.id).toBe(42);
    expect(m.parts.map((p) => p.t)).toEqual(["text", "tool", "text"]);
    expect(m.parts[0]).toEqual({ t: "text", text: "Checking disk. " });
    expect(m.parts[1]).toMatchObject({ name: "terminal", status: "ok", output: "19G free", durationMs: 120, exitCode: 0 });
    s = chatReducer(s, { type: "events", events: [ev("done", { finish_reason: "stop", message_id: 42, duration_ms: 900, usage: { completion_tokens: 12 } })] });
    expect(last(s)).toMatchObject({ status: "complete", finishReason: "stop", durationMs: 900, completionTokens: 12 });
    expect(s.turn.status).toBe("idle");
    expect(messageText(last(s))).toBe("Checking disk. \n\nYou have 19G.");
  });

  it("ignores events it has already seen (replay after a reconnect) — no duplicated tokens", () => {
    let s = start();
    const events = [ev("turn", { turn_id: "t" }), ev("delta", { text: "A" }), ev("delta", { text: "B" })];
    s = chatReducer(s, { type: "events", events });
    s = chatReducer(s, { type: "events", events }); // the whole batch again
    s = chatReducer(s, { type: "events", events: [...events, ev("delta", { text: "C" })] });
    expect(messageText(last(s))).toBe("ABC");
    expect(s.turn.lastSeq).toBe(4);
  });

  it("tracks real thinking state and accumulates its duration", () => {
    let s = start();
    s = chatReducer(s, { type: "events", now: 1000, events: [ev("reasoning", { active: true })] });
    expect(s.turn.reasoning).toEqual({ active: true, since: 1000 });
    s = chatReducer(s, { type: "events", events: [ev("reasoning", { active: false, duration_ms: 4000 }), ev("reasoning", { active: true }),
      ev("reasoning", { active: false, duration_ms: 1500 })] });
    expect(s.turn.reasoning.active).toBe(false);
    expect(last(s).thinkingMs).toBe(5500);
  });

  it("marks a failed turn as error, keeping the partial text and the reason", () => {
    let s = start();
    s = chatReducer(s, { type: "events", events: [ev("delta", { text: "Partial" }),
      ev("error", { code: "gpu_unreachable", message: "tunnel down", retryable: true, action: "check_gpu" }),
      ev("done", { finish_reason: "error", error: { code: "gpu_unreachable", message: "tunnel down" } })] });
    expect(last(s)).toMatchObject({ status: "error", finishReason: "error" });
    expect(last(s).error).toMatchObject({ code: "gpu_unreachable", message: "tunnel down" });
    expect(messageText(last(s))).toBe("Partial");
  });

  it("marks a user-stopped turn as stopped (not an error)", () => {
    let s = start();
    s = chatReducer(s, { type: "events", events: [ev("delta", { text: "Half"}), ev("done", { finish_reason: "cancelled" })] });
    expect(last(s)).toMatchObject({ status: "stopped", finishReason: "cancelled", error: null });
  });

  it("collects server notices on the message", () => {
    let s = start();
    s = chatReducer(s, { type: "events", events: [ev("notice", { text: "Retrying once." })] });
    expect(last(s).notices).toEqual(["Retrying once."]);
  });

  it("closes tool rows that never reported back", () => {
    let s = start();
    s = chatReducer(s, { type: "events", events: [ev("tool_start", { id: "c", name: "terminal", args: {} }),
      ev("done", { finish_reason: "cancelled" })] });
    expect((last(s).parts[0] as { status: string }).status).toBe("cancelled");
  });
});

describe("streams that end without `done`", () => {
  it("after Stop: stopped, running tools cancelled", () => {
    let s = start();
    s = chatReducer(s, { type: "events", events: [ev("delta", { text: "x" }), ev("tool_start", { id: "c", name: "t", args: {} })] });
    s = chatReducer(s, { type: "stopping" });
    s = chatReducer(s, { type: "streamEnded" });
    expect(last(s)).toMatchObject({ status: "stopped", finishReason: "cancelled", error: null });
    expect((last(s).parts[1] as { status: string }).status).toBe("cancelled");
    expect(s.turn.status).toBe("idle");
  });

  it("when the connection was lost: an honest, retryable error", () => {
    let s = start();
    s = chatReducer(s, { type: "events", events: [ev("delta", { text: "partial" })] });
    s = chatReducer(s, { type: "streamEnded", error: null });
    expect(last(s).status).toBe("error");
    expect(last(s).error).toMatchObject({ code: "connection_lost", retryable: true });
    expect(messageText(last(s))).toBe("partial");
  });

  it("reconnecting is a state, not an error", () => {
    let s = start();
    s = chatReducer(s, { type: "events", events: [ev("turn", { turn_id: "t" })] });
    s = chatReducer(s, { type: "reconnecting", attempt: 2 });
    expect(s.turn).toMatchObject({ status: "reconnecting", reconnectAttempt: 2 });
    s = chatReducer(s, { type: "events", events: [ev("delta", { text: "ok" })] });
    expect(last(s).status).toBe("streaming");
  });
});

describe("opening, attaching, regenerating", () => {
  const api = (over: Partial<ApiMessage>): ApiMessage => ({ id: 1, role: "assistant", content: "x", ...over });

  it("open resets; a late load for another chat is ignored", () => {
    let s = chatReducer(initialChatState, { type: "open", id: "a" });
    expect(s.loading).toBe(true);
    s = chatReducer(s, { type: "loaded", id: "b", title: "B", messages: [] });
    expect(s.loading).toBe(true);
    s = chatReducer(s, { type: "loaded", id: "a", title: "A", messages: [fromApiMessage(api({}))] });
    expect(s).toMatchObject({ loading: false, title: "A" });
    expect(s.messages).toHaveLength(1);
    s = chatReducer(s, { type: "open", id: null });
    expect(s).toMatchObject({ activeId: null, messages: [], loading: false });
  });

  it("attach drops the stored partial answer so the replay cannot duplicate it", () => {
    let s = chatReducer(initialChatState, { type: "open", id: "a" });
    s = chatReducer(s, { type: "loaded", id: "a", title: "A", messages: [
      fromApiMessage(api({ id: 1, role: "user", content: "q" })),
      fromApiMessage(api({ id: 2, content: "partial words so far", finish_reason: "running" })) ] });
    s = chatReducer(s, { type: "attach", assistantKey: "live", dropMessageId: 2, turnId: "turn-7" });
    expect(s.messages.map((m) => m.key)).toEqual(["m1", "live"]);
    expect(last(s)).toMatchObject({ status: "streaming", parts: [] });
    expect(s.turn).toMatchObject({ status: "connecting", turnId: "turn-7", lastSeq: 0 });
    s = chatReducer(s, { type: "events", events: [ev("delta", { text: "partial words so far" })] });
    expect(messageText(last(s))).toBe("partial words so far");
  });

  it("regenerate replaces only the last assistant message", () => {
    let s = start();
    s = chatReducer(s, { type: "events", events: [ev("delta", { text: "old" }), ev("done", {})] });
    s = chatReducer(s, { type: "regenerate", assistantKey: "a2" });
    expect(s.messages.map((m) => [m.role, m.key])).toEqual([["user", "u1"], ["assistant", "a2"]]);
  });
});

describe("fromApiMessage", () => {
  it("restores parts (text + tool activity) and statuses", () => {
    const m = fromApiMessage({
      id: 5, role: "assistant", content: "Done.", finish_reason: "stop", thinking_ms: 3000, duration_ms: 8000,
      usage: { completion_tokens: 20 },
      parts: [{ t: "text", text: "Running." },
        { t: "tool", id: "c1", name: "terminal", label: "Terminal", status: "ok", args: { command: "ls" }, output: "a b", duration_ms: 50, exit_code: 0 },
        { t: "text", text: "Done." }],
    });
    expect(m).toMatchObject({ key: "m5", status: "complete", thinkingMs: 3000, durationMs: 8000, completionTokens: 20 });
    expect(m.parts.map((p) => p.t)).toEqual(["text", "tool", "text"]);
    expect(m.parts[1]).toMatchObject({ status: "ok", durationMs: 50, exitCode: 0 });
  });

  it("falls back to content, and maps finish reasons", () => {
    expect(fromApiMessage({ id: 1, role: "assistant", content: "plain" }).parts).toEqual([{ t: "text", text: "plain" }]);
    const by = (r: ApiMessage["finish_reason"]) => fromApiMessage({ id: 1, role: "assistant", content: "x", finish_reason: r }).status;
    expect([by("stop"), by("length"), by("incomplete"), by("cancelled"), by("interrupted"), by("error"), by("timeout")])
      .toEqual(["complete", "complete", "complete", "stopped", "stopped", "error", "error"]);
  });

  it("a stored 'running' tool can never still be running", () => {
    const m = fromApiMessage({ id: 1, role: "assistant", content: "x", parts: [{ t: "tool", id: "c", name: "t", status: "running", args: {} }] });
    expect((m.parts[0] as { status: string }).status).toBe("cancelled");
  });

  it("user messages keep attachments", () => {
    expect(fromApiMessage({ id: 2, role: "user", content: "see file", attachments: ["a.txt"] }).attachments).toEqual(["a.txt"]);
  });
});

describe("export", () => {
  it("renders a readable markdown transcript", () => {
    let s = start();
    s = chatReducer(s, { type: "events", events: [ev("delta", { text: "Hello!" }), ev("done", {})] });
    const md = exportMarkdown("My chat", s.messages);
    expect(md).toContain("# My chat");
    expect(md).toContain("## You\n\nhi");
    expect(md).toContain("## Blackthorn\n\nHello!");
  });
});
