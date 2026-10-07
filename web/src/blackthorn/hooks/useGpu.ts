import { useCallback, useEffect, useRef, useState } from "react";
import { useLatest } from "./useLatest";
import { btApi } from "../api";
import type { GpuSnapshot } from "../types";

export interface GpuApi {
  gpu: GpuSnapshot | null;
  /** The status endpoint itself is unreachable (distinct from the GPU being off). */
  unreachable: boolean;
  acting: boolean;
  refresh: () => Promise<void>;
  turnOn: () => Promise<void>;
  turnOff: () => Promise<void>;
  restart: () => Promise<void>;
  verify: () => Promise<void>;
  setAutoOff: (minutes: number) => Promise<void>;
  /** Poll faster while the panel is open. */
  setWatching: (on: boolean) => void;
}

/**
 * Real GPU state from /api/kaggle-gpu/status (an in-memory snapshot, answers instantly).
 * Polls every 2 s while booting/stopping or watched, otherwise every 8 s; pauses when the tab
 * is hidden and refreshes the moment it becomes visible again — so after a page reload or a
 * long absence the header always shows the *current* truth.
 */
export function useGpu(onError: (m: string) => void): GpuApi {
  const [gpu, setGpu] = useState<GpuSnapshot | null>(null);
  const [unreachable, setUnreachable] = useState(false);
  const [acting, setActing] = useState(false);
  const watching = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const alive = useRef(true);
  const stateRef = useRef<GpuSnapshot | null>(null);
  const onErr = useLatest(onError);
  const pollRef = useRef<() => Promise<void>>(async () => {});

  const schedule = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    if (!alive.current) return;
    const s = stateRef.current?.state;
    const fast = watching.current || s === "starting" || s === "stopping" || stateRef.current?.transitioning;
    timer.current = setTimeout(() => void pollRef.current(), fast ? 2000 : 8000);
  }, []);

  const poll = useCallback(async () => {
    if (typeof document !== "undefined" && document.hidden) return schedule();
    try {
      const snap = await btApi.gpuStatus();
      if (!alive.current) return;
      stateRef.current = snap;
      setGpu(snap);
      setUnreachable(false);
    } catch {
      if (alive.current) setUnreachable(true);
    }
    schedule();
  }, [schedule]);
  useEffect(() => {
    pollRef.current = poll;
  }, [poll]);

  useEffect(() => {
    alive.current = true;
    void poll();
    const onVisible = () => {
      if (!document.hidden) void poll();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      alive.current = false;
      if (timer.current) clearTimeout(timer.current);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [poll]);

  const act = useCallback(
    async (fn: () => Promise<unknown>, failure: string) => {
      setActing(true);
      try {
        await fn();
      } catch (e) {
        onErr.current(`${failure}: ${e instanceof Error ? e.message : "request failed"}`);
      } finally {
        setActing(false);
        void poll();
      }
    },
    [poll, onErr],
  );

  return {
    gpu,
    unreachable,
    acting,
    refresh: poll,
    turnOn: () => act(btApi.gpuTurnOn, "Could not start the GPU"),
    turnOff: () => act(btApi.gpuTurnOff, "Could not stop the GPU"),
    restart: () => act(btApi.gpuRestart, "Could not restart the GPU"),
    verify: () => act(btApi.gpuVerify, "The model test failed"),
    setAutoOff: (minutes) => act(() => btApi.gpuAutoOff(minutes), "Could not change auto-off"),
    setWatching: (on) => {
      watching.current = on;
      schedule();
      if (on) void poll();
    },
  };
}
