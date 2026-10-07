import { useCallback, useEffect, useRef, useState, type RefObject } from "react";

/**
 * Keep a growing transcript pinned to the bottom — unless the user scrolled up to read.
 * Returns whether the "jump to latest" affordance should be shown and a function to jump.
 */
export function useAutoScroll(ref: RefObject<HTMLElement | null>, watch: unknown[], force: unknown) {
  const stick = useRef(true);
  const [away, setAway] = useState(false);

  const jump = useCallback(
    (smooth = false) => {
      const el = ref.current;
      if (!el) return;
      stick.current = true;
      setAway(false);
      el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" });
    },
    [ref],
  );

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onScroll = () => {
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 72;
      stick.current = atBottom;
      setAway(!atBottom);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [ref]);

  // content changed: follow it only if we were already at the bottom
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (stick.current) el.scrollTop = el.scrollHeight;
    else setAway(el.scrollHeight - el.scrollTop - el.clientHeight >= 72);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, watch);

  // the user sent a message / opened a chat: always go to the end
  useEffect(() => {
    jump(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [force]);

  return { away, jump };
}
