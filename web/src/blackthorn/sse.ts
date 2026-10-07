/**
 * Incremental Server-Sent-Events parser.
 *
 * Fed arbitrary text chunks (network chunk boundaries are meaningless), it returns the
 * complete frames seen so far and keeps the unfinished tail. Handles CRLF, comments
 * (`: ping`), multi-line `data:`, and ignores malformed ids.
 */
export interface SseFrame {
  id?: number;
  event: string;
  data: string;
}

export class SseParser {
  private tail = "";
  private id: number | undefined;
  private event = "message";
  private data: string[] = [];

  push(chunk: string): SseFrame[] {
    this.tail += chunk;
    const frames: SseFrame[] = [];
    let idx: number;
    while ((idx = this.nextLineBreak()) !== -1) {
      let line = this.tail.slice(0, idx);
      const skip = this.tail[idx] === "\r" && this.tail[idx + 1] === "\n" ? 2 : 1;
      this.tail = this.tail.slice(idx + skip);
      if (line.endsWith("\r")) line = line.slice(0, -1);
      const frame = this.line(line);
      if (frame) frames.push(frame);
    }
    return frames;
  }

  /** A lone trailing "\r" might be the first half of "\r\n", so wait for more input. */
  private nextLineBreak(): number {
    const n = this.tail.indexOf("\n");
    const r = this.tail.indexOf("\r");
    if (r !== -1 && (n === -1 || r < n)) {
      if (r === this.tail.length - 1) return -1;
      return r;
    }
    return n;
  }

  private line(line: string): SseFrame | null {
    if (line === "") {
      if (this.data.length === 0) {
        this.event = "message";
        this.id = undefined;
        return null;
      }
      const frame: SseFrame = { event: this.event, data: this.data.join("\n") };
      if (this.id !== undefined) frame.id = this.id;
      this.data = [];
      this.event = "message";
      this.id = undefined;
      return frame;
    }
    if (line.startsWith(":")) return null;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "data") this.data.push(value);
    else if (field === "event") this.event = value || "message";
    else if (field === "id" && /^\d+$/.test(value)) this.id = Number(value);
    return null;
  }
}
