import { describe, expect, it } from "vitest";
import { runTurn } from "./api";
import type { StreamEvent } from "./types";

const enc = new TextEncoder();
const frame = (seq: number, type: string, data: object) => `id: ${seq}\nevent: ${type}\ndata: ${JSON.stringify(data)}\n\n`;

/** A Response whose body yields the given text chunks, then ends (or errors). */
function sseResponse(chunks: string[], opts: { status?: number; failAfter?: boolean } = {}): Response {
  const body = new ReadableStream<Uint8Array>({
    async start(controller) {
      for (const c of chunks) {
        controller.enqueue(enc.encode(c));
        await new Promise((r) => setTimeout(r, 1));
      }
      if (opts.failAfter) controller.error(new TypeError("network error"));
      else controller.close();
    },
  });
  return new Response(body, { status: opts.status ?? 200, headers: { "content-type": "text/event-stream" } });
}

const collect = () => {
  const events: StreamEvent[] = [];
  return { events, onEvents: (b: StreamEvent[]) => events.push(...b) };
};
const fast = { backoffMs: [1, 1, 1, 1] };

describe("runTurn", () => {
  it("streams a turn to completion and reports the turn id", async () => {
    const { events, onEvents } = collect();
    const calls: string[] = [];
    const res = await runTurn({
      start: { message: "hi" }, signal: new AbortController().signal, onEvents, ...fast,
      fetchImpl: (async (url: string) => {
        calls.push(url);
        return sseResponse([
          "retry: 2000\n: pad\n\n" + frame(1, "turn", { turn_id: "turn-1", state: "accepted" }),
          frame(2, "delta", { text: "Hel" }) + frame(3, "delta", { text: "lo" }).slice(0, 20),
          frame(3, "delta", { text: "lo" }).slice(20) + frame(4, "done", { finish_reason: "stop" }),
        ]);
      }) as unknown as typeof fetch,
    });
    expect(res).toMatchObject({ done: true, turnId: "turn-1", lastSeq: 4, error: null, aborted: false });
    expect(events.map((e) => e.type)).toEqual(["turn", "delta", "delta", "done"]);
    expect(calls).toHaveLength(1);
  });

  it("re-attaches after a dropped connection and neither loses nor duplicates events", async () => {
    const { events, onEvents } = collect();
    const reconnects: number[] = [];
    const urls: string[] = [];
    let n = 0;
    const res = await runTurn({
      start: { message: "hi" }, signal: new AbortController().signal, onEvents, ...fast,
      onReconnecting: (a) => reconnects.push(a),
      fetchImpl: (async (url: string) => {
        urls.push(url);
        n += 1;
        if (n === 1) {
          return sseResponse([frame(1, "turn", { turn_id: "turn-9" }), frame(2, "delta", { text: "A" }), frame(3, "delta", { text: "B" })]);
        }
        if (n === 2) { // server replays from seq 3 on (re-sends seq 3 too: must be ignored) then goes on
          return sseResponse([frame(3, "delta", { text: "B" }), frame(4, "delta", { text: "C" }), frame(5, "done", { finish_reason: "stop" })]);
        }
        throw new Error("unexpected");
      }) as unknown as typeof fetch,
    });
    expect(res).toMatchObject({ done: true, lastSeq: 5, error: null });
    expect(events.map((e) => e.seq)).toEqual([1, 2, 3, 4, 5]);
    expect(events.filter((e) => e.type === "delta").map((e) => e.data.text).join("")).toBe("ABC");
    expect(urls[1]).toContain("/api/studio/turns/turn-9/stream?after=3");
    expect(reconnects).toEqual([1]);
  });

  it("re-attaches after a network *error* too", async () => {
    const { events, onEvents } = collect();
    let n = 0;
    const res = await runTurn({
      start: { message: "hi" }, signal: new AbortController().signal, onEvents, ...fast,
      fetchImpl: (async () => {
        n += 1;
        if (n === 1) return sseResponse([frame(1, "turn", { turn_id: "t" }), frame(2, "delta", { text: "x" })], { failAfter: true });
        return sseResponse([frame(3, "done", {})]);
      }) as unknown as typeof fetch,
    });
    expect(res.done).toBe(true);
    expect(events.map((e) => e.seq)).toEqual([1, 2, 3]);
  });

  it("gives up after the reconnect budget with an error", async () => {
    const { onEvents } = collect();
    let n = 0;
    const res = await runTurn({
      start: { message: "hi" }, signal: new AbortController().signal, onEvents, backoffMs: [1, 1], maxReconnects: 2,
      fetchImpl: (async () => {
        n += 1;
        return n === 1 ? sseResponse([frame(1, "turn", { turn_id: "t" })]) : sseResponse([]);
      }) as unknown as typeof fetch,
    });
    expect(res.done).toBe(false);
    expect(res.error).toBeInstanceOf(Error);
    expect(n).toBe(3);
  });

  it("does not retry the start request when the connection drops before the turn was accepted", async () => {
    const { onEvents } = collect();
    let n = 0;
    const res = await runTurn({
      start: { message: "hi" }, signal: new AbortController().signal, onEvents, ...fast,
      fetchImpl: (async () => { n += 1; return sseResponse([], { failAfter: true }); }) as unknown as typeof fetch,
    });
    expect(n).toBe(1);
    expect(res.done).toBe(false);
    expect(res.turnId).toBeNull();
    expect(res.error?.message).toBeTruthy();
  });

  it("client errors (4xx) are returned without retrying; the detail is surfaced", async () => {
    const { onEvents } = collect();
    let n = 0;
    const res = await runTurn({
      start: { message: "" }, signal: new AbortController().signal, onEvents, ...fast,
      fetchImpl: (async () => { n += 1; return new Response(JSON.stringify({ detail: "message is empty" }), { status: 400 }); }) as unknown as typeof fetch,
    });
    expect(n).toBe(1);
    expect(res.error?.message).toBe("message is empty");
  });

  it("attaching to a turn that no longer exists is reported (404), not retried forever", async () => {
    const { onEvents } = collect();
    let n = 0;
    const res = await runTurn({
      attach: { turnId: "turn-gone", after: 0 }, signal: new AbortController().signal, onEvents, ...fast,
      fetchImpl: (async () => { n += 1; return new Response(JSON.stringify({ detail: "turn not found" }), { status: 404 }); }) as unknown as typeof fetch,
    });
    expect(n).toBe(1);
    expect((res.error as Error).message).toContain("turn not found");
  });

  it("abort stops immediately and is reported as aborted", async () => {
    const ac = new AbortController();
    const { events, onEvents } = collect();
    const res = await runTurn({
      start: { message: "hi" }, signal: ac.signal, onEvents, ...fast,
      fetchImpl: (async (_u: string, init: RequestInit) => {
        const body = new ReadableStream<Uint8Array>({
          start(c) {
            c.enqueue(enc.encode(frame(1, "turn", { turn_id: "t" }) + frame(2, "delta", { text: "x" })));
            init.signal?.addEventListener("abort", () => c.error(new DOMException("aborted", "AbortError")));
          },
        });
        queueMicrotask(() => setTimeout(() => ac.abort(), 5));
        return new Response(body, { status: 200 });
      }) as unknown as typeof fetch,
    });
    expect(res.aborted).toBe(true);
    expect(res.done).toBe(false);
    expect(events.length).toBeGreaterThan(0);
  });

  it("skips malformed events but keeps going", async () => {
    const { events, onEvents } = collect();
    const res = await runTurn({
      start: { message: "hi" }, signal: new AbortController().signal, onEvents, ...fast,
      fetchImpl: (async () => sseResponse([
        frame(1, "turn", { turn_id: "t" }), "id: 2\nevent: delta\ndata: {not json\n\n", frame(3, "delta", { text: "ok" }), frame(4, "done", {}),
      ])) as unknown as typeof fetch,
    });
    expect(res.done).toBe(true);
    expect(events.map((e) => e.seq)).toEqual([1, 3, 4]);
  });
});
