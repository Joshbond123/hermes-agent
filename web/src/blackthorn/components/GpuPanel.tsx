import { useEffect, useRef, useState } from "react";
import { btApi } from "../api";
import type { GpuApi } from "../hooks/useGpu";
import type { GpuSnapshot } from "../types";
import { formatDuration } from "../format";
import { Icon, Spinner } from "./ui";

const HEADLINE: Record<string, string> = {
  unknown: "Checking…", off: "GPU is off", starting: "GPU is starting", ready: "GPU is ready", stopping: "GPU is stopping", error: "GPU error",
};

function gb(n: number | undefined): string {
  return n === undefined ? "" : (n / 1e9).toFixed(1);
}

function Progress({ gpu }: { gpu: GpuSnapshot }) {
  const p = gpu.progress;
  if (!p || p.mode === "none") return null;
  if (p.mode === "determinate" && p.fraction !== undefined) {
    const pct = Math.round(p.fraction * 100);
    return (
      <div className="bt-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct} aria-label={gpu.stage_label}>
        <div className="bt-progress-bar" style={{ width: `${pct}%` }} />
        <div className="bt-progress-text">{p.unit === "bytes" ? `${gb(p.done)} of ${gb(p.total)} GB · ${pct}%` : `${pct}%`}</div>
      </div>
    );
  }
  return (
    <div className="bt-progress bt-progress-indeterminate" role="progressbar" aria-label={`${gpu.stage_label ?? "Working"} (duration unknown)`}>
      <div className="bt-progress-bar" />
    </div>
  );
}

function Logs() {
  const [lines, setLines] = useState<string[] | null>(null);
  const [note, setNote] = useState("");
  const box = useRef<HTMLPreElement>(null);
  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const res = await btApi.gpuLogs(160);
        if (!alive) return;
        setLines(res.lines);
        setNote(res.note ?? "");
      } catch (e) {
        if (alive) setNote(e instanceof Error ? e.message : "Logs unavailable");
      }
    };
    void load();
    const t = setInterval(load, 4000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);
  useEffect(() => {
    if (box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [lines]);
  return (
    <div className="bt-gpu-logs">
      <pre ref={box} tabIndex={0}>{lines === null ? "Loading…" : lines.length ? lines.join("\n") : note || "No log lines yet."}</pre>
      {lines && lines.length && note ? <p className="bt-muted">{note}</p> : null}
    </div>
  );
}

export function GpuPanel({ api, onClose }: { api: GpuApi; onClose: () => void }) {
  const { gpu, acting } = api;
  // "confirm stop" belongs to one state: it disappears by itself when the GPU state changes
  const [confirmedFor, setConfirmedFor] = useState<string | null>(null);
  const confirmStop = confirmedFor === gpu?.state;
  const setConfirmStop = (on: boolean) => setConfirmedFor(on ? gpu?.state ?? null : null);
  const [showLogs, setShowLogs] = useState(false);
  useEffect(() => {
    api.setWatching(true);
    return () => api.setWatching(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!gpu) {
    return (
      <div className="bt-gpu">
        <div className="bt-gpu-head"><Spinner /> <strong>{api.unreachable ? "Can’t reach the server" : "Checking the GPU…"}</strong></div>
        <p className="bt-muted">{api.unreachable ? "The GPU status could not be loaded. Retrying automatically." : ""}</p>
      </div>
    );
  }

  const state = gpu.state;
  const quota = gpu.quota;
  const auto = gpu.auto_off;
  const li = gpu.last_inference;

  return (
    <div className="bt-gpu">
      <div className="bt-gpu-head">
        <span className={`bt-dot bt-dot-gpu-${state}`} aria-hidden="true" />
        <strong>{HEADLINE[state] ?? state}</strong>
        <span className="bt-spacer" />
        <button type="button" className="bt-icon-btn" aria-label="Refresh GPU status" onClick={() => void api.refresh()}><Icon name="refresh" size={16} /></button>
        <button type="button" className="bt-icon-btn bt-gpu-close" aria-label="Close GPU panel" onClick={onClose}><Icon name="x" size={16} /></button>
      </div>

      <p className="bt-gpu-message">{gpu.message}</p>

      {state === "starting" ? (
        <div className="bt-gpu-section">
          <div className="bt-gpu-stage">
            <Spinner size={12} />
            <span>{gpu.stage_label || gpu.stage}</span>
          </div>
          <Progress gpu={gpu} />
          <div className="bt-gpu-times">
            {gpu.stage_elapsed_s !== undefined ? <span>This step: {formatDuration(gpu.stage_elapsed_s)}</span> : null}
            {gpu.elapsed_s ? <span>Total: {formatDuration(gpu.elapsed_s)}</span> : null}
          </div>
          {gpu.stalled ? (
            <div className="bt-callout bt-callout-warn" role="alert">
              No update from the GPU for {formatDuration(gpu.silent_for_s ?? 0)}. It may be stuck — you can restart it.
            </div>
          ) : null}
        </div>
      ) : null}

      {state === "error" && gpu.error ? (
        <div className="bt-callout bt-callout-error" role="alert">{gpu.error.message}</div>
      ) : null}

      {state === "ready" || li ? (
        <div className="bt-gpu-section">
          <div className="bt-kv"><span>Model</span><span>{gpu.model}</span></div>
          {gpu.gpu ? <div className="bt-kv"><span>Hardware</span><span>{gpu.gpu}</span></div> : null}
          {state === "ready" ? (
            <div className="bt-kv">
              <span>In GPU memory</span>
              <span>{gpu.model_loaded === true ? "Yes" : gpu.model_loaded === false ? "Loads on the next message" : "Unknown"}</span>
            </div>
          ) : null}
          {li ? (
            <div className="bt-kv">
              <span>Last test</span>
              <span className={li.ok ? "bt-ok" : "bt-warn"}>
                {li.ok ? `Answered in ${((li.latency_ms ?? 0) / 1000).toFixed(1)}s` : `Failed${li.error ? ` — ${li.error}` : ""}`}
              </span>
            </div>
          ) : null}
          {state === "ready" ? (
            <button type="button" className="bt-btn bt-btn-small" onClick={() => void api.verify()} disabled={acting}>
              {acting ? <Spinner /> : null} Test the model now
            </button>
          ) : null}
        </div>
      ) : null}

      {quota ? (
        <div className="bt-gpu-section">
          <div className="bt-kv"><span>Kaggle GPU quota (weekly)</span><span>{quota.used_hours}h of {quota.total_hours}h used</span></div>
          <div className="bt-progress bt-progress-quota" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(quota.used_pct)} aria-label="Weekly GPU quota used">
            <div className="bt-progress-bar" style={{ width: `${Math.min(100, quota.used_pct)}%` }} />
          </div>
          <p className="bt-muted">{quota.remaining_hours}h left{quota.refresh_time ? ` · resets ${new Date(quota.refresh_time).toLocaleDateString(undefined, { month: "short", day: "numeric" })}` : ""}</p>
        </div>
      ) : null}

      {auto ? (
        <div className="bt-gpu-section">
          <div className="bt-kv">
            <span>Stop automatically when idle</span>
            <span>{auto.enabled ? `after ${auto.minutes} min` : "never"}</span>
          </div>
          <div className="bt-seg" role="group" aria-label="Auto-off after inactivity">
            {auto.choices.map((m) => (
              <button
                key={m} type="button" className={`bt-seg-btn${auto.minutes === m ? " is-on" : ""}`}
                aria-pressed={auto.minutes === m} disabled={acting} onClick={() => void api.setAutoOff(m)}
              >
                {m === 0 ? "Never" : `${m}m`}
              </button>
            ))}
          </div>
          {state === "ready" && auto.enabled && auto.remaining_s !== null ? (
            <p className="bt-muted">{gpu.busy ? "Working now — the timer is paused." : `Stops in ${formatDuration(auto.remaining_s)} without activity.`}</p>
          ) : null}
        </div>
      ) : null}

      <div className="bt-gpu-actions">
        {state === "off" || state === "unknown" ? (
          <button type="button" className="bt-btn bt-btn-primary" onClick={() => void api.turnOn()} disabled={acting}>
            {acting ? <Spinner /> : null} Start GPU
          </button>
        ) : null}
        {state === "starting" ? (
          <>
            <button type="button" className={`bt-btn${gpu.stalled ? " bt-btn-primary" : ""}`} onClick={() => void api.restart()} disabled={acting}>
              Restart GPU
            </button>
            <button type="button" className="bt-btn" onClick={() => void api.turnOff()} disabled={acting}>Cancel start</button>
          </>
        ) : null}
        {state === "ready" ? (
          confirmStop ? (
            <>
              <button type="button" className="bt-btn bt-btn-danger" onClick={() => void api.turnOff()} disabled={acting}>Confirm stop</button>
              <button type="button" className="bt-btn" onClick={() => setConfirmStop(false)}>Keep running</button>
            </>
          ) : (
            <>
              <button type="button" className="bt-btn" onClick={() => setConfirmStop(true)} disabled={acting}>Stop GPU</button>
              <button type="button" className="bt-btn" onClick={() => void api.restart()} disabled={acting}>Restart</button>
            </>
          )
        ) : null}
        {state === "error" ? (
          <>
            <button type="button" className="bt-btn bt-btn-primary" onClick={() => void api.restart()} disabled={acting}>Restart GPU</button>
            <button type="button" className="bt-btn" onClick={() => void api.turnOff()} disabled={acting}>Turn off</button>
          </>
        ) : null}
        {state === "stopping" ? <span className="bt-muted"><Spinner size={12} /> Stopping — one moment…</span> : null}
      </div>

      {state !== "off" ? (
        <div className="bt-gpu-logs-wrap">
          <button type="button" className="bt-link-btn" aria-expanded={showLogs} onClick={() => setShowLogs(!showLogs)}>
            {showLogs ? "Hide" : "Show"} server log
          </button>
          {showLogs ? <Logs /> : null}
        </div>
      ) : null}
    </div>
  );
}
