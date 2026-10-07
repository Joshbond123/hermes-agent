import { describe, expect, it } from "vitest";
import { SseParser } from "./sse";

const STREAM =
  "retry: 2000\n: " + " ".repeat(300) + "\n\n" +
  'id: 1\nevent: turn\ndata: {"state":"accepted"}\n\n' +
  ": ping\n\n" +
  'id: 2\nevent: delta\ndata: {"text":"héllo wörld ✓"}\n\n' +
  "id: 3\nevent: multi\ndata: line one\ndata: line two\n\n" +
  'id: nope\nevent: weird\ndata: {"x":1}\n\n' +
  'id: 4\r\nevent: done\r\ndata: {"finish_reason":"stop"}\r\n\r\n';

function parseAll(chunks: string[]) {
  const p = new SseParser();
  return chunks.flatMap((c) => p.push(c));
}

describe("SseParser", () => {
  const whole = parseAll([STREAM]);

  it("extracts frames, ignoring comments/padding, and keeps ids and multi-line data", () => {
    expect(whole.map((f) => f.event)).toEqual(["turn", "delta", "multi", "weird", "done"]);
    expect(whole[0]).toEqual({ id: 1, event: "turn", data: '{"state":"accepted"}' });
    expect(JSON.parse(whole[1].data).text).toBe("héllo wörld ✓");
    expect(whole[2].data).toBe("line one\nline two");
    expect(whole[3].id).toBeUndefined(); // malformed id ignored
    expect(whole[4]).toEqual({ id: 4, event: "done", data: '{"finish_reason":"stop"}' });
  });

  it("gives identical frames whatever the chunk boundaries are (every split point)", () => {
    for (let i = 0; i <= STREAM.length; i += 1) {
      expect(parseAll([STREAM.slice(0, i), STREAM.slice(i)])).toEqual(whole);
    }
  });

  it("works one character at a time", () => {
    expect(parseAll(STREAM.split(""))).toEqual(whole);
  });

  it("does not emit a half-received frame", () => {
    const p = new SseParser();
    expect(p.push('id: 1\nevent: delta\ndata: {"te')).toEqual([]);
    expect(p.push('xt":"a"}\n')).toEqual([]);
    expect(p.push("\n")).toHaveLength(1);
  });

  it("treats a blank line without data as a no-op", () => {
    expect(parseAll(["\n\n\n"])).toEqual([]);
  });
});
