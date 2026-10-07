import { describe, expect, it } from "vitest";
import { groupSessions } from "./groupSessions";
import type { SessionSummary } from "./types";

const mk = (id: string, updated: number, pinned = false): SessionSummary => ({
  id, title: id, pinned, archived: false, updated_at: updated, created_at: updated, message_count: 1, model: "m",
});

describe("groupSessions", () => {
  // 2026-10-07 14:00 UTC, user in UTC
  const now = Date.UTC(2026, 9, 7, 14, 0, 0) / 1000;
  const h = 3600;
  it("puts pinned chats first regardless of age and buckets the rest by day", () => {
    const groups = groupSessions([
      mk("old", now - 40 * 86400), mk("today", now - 2 * h), mk("yday", now - 20 * h), mk("week", now - 4 * 86400),
      mk("pin", now - 90 * 86400, true), mk("today2", now - 1 * h),
    ], now, 0);
    expect(groups.map((g) => g.key)).toEqual(["pinned", "today", "yesterday", "week", "earlier"]);
    expect(groups[1].items.map((s) => s.id)).toEqual(["today2", "today"]); // most recent first
    expect(groups[0].items[0].id).toBe("pin");
  });
  it("omits empty groups and respects the local timezone for 'today'", () => {
    expect(groupSessions([mk("a", now - h)], now, 0).map((g) => g.key)).toEqual(["today"]);
    // 00:30 local (UTC+1) is 23:30 UTC the previous day: still "today" for that user
    const t = Date.UTC(2026, 9, 6, 23, 30, 0) / 1000;
    expect(groupSessions([mk("a", t)], Date.UTC(2026, 9, 7, 0, 40, 0) / 1000, 60).map((g) => g.key)).toEqual(["today"]);
    expect(groupSessions([], now, 0)).toEqual([]);
  });
});
