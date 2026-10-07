import type { SessionSummary } from "./types";

export interface SessionGroup {
  key: string;
  label: string;
  items: SessionSummary[];
}

const DAY = 86400;

/** Pinned first, then by recency buckets. `now` is epoch seconds (injectable for tests). */
export function groupSessions(sessions: SessionSummary[], now = Date.now() / 1000, tzOffsetMinutes = -new Date().getTimezoneOffset()): SessionGroup[] {
  const localMidnight = (() => {
    const local = now + tzOffsetMinutes * 60;
    return Math.floor(local / DAY) * DAY - tzOffsetMinutes * 60;
  })();
  const buckets: SessionGroup[] = [
    { key: "pinned", label: "Pinned", items: [] },
    { key: "today", label: "Today", items: [] },
    { key: "yesterday", label: "Yesterday", items: [] },
    { key: "week", label: "Previous 7 days", items: [] },
    { key: "earlier", label: "Earlier", items: [] },
  ];
  const sorted = [...sessions].sort((a, b) => (b.updated_at ?? 0) - (a.updated_at ?? 0));
  for (const s of sorted) {
    if (s.pinned) buckets[0].items.push(s);
    else {
      const t = s.updated_at ?? 0;
      if (t >= localMidnight) buckets[1].items.push(s);
      else if (t >= localMidnight - DAY) buckets[2].items.push(s);
      else if (t >= localMidnight - 7 * DAY) buckets[3].items.push(s);
      else buckets[4].items.push(s);
    }
  }
  return buckets.filter((b) => b.items.length > 0);
}
