import { memo, useEffect, useState } from "react";
import { copyTextToClipboard } from "@/lib/clipboard";
import { Markdown } from "../markdown/Markdown";
import { messageText, type TurnState } from "../store";
import type { ErrorInfo, GpuSnapshot, ToolPart, UiMessage } from "../types";
import { formatDuration, primaryArg } from "../format";
import { Icon, Spinner } from "./ui";

// --------------------------------------------------------------------------- //
// tool activity — rendered only for tool events the server really emitted
// --------------------------------------------------------------------------- //
function ToolRow({ part }: { part: ToolPart }) {
  const [open, setOpen] = useState(false);
  const summary = primaryArg(part.args).replace(/\s+/g, " ");
  const state =
    part.status === "running" ? "running"
    : part.status === "ok" ? "ok"
    : part.status === "cancelled" ? "stopped"
    : part.exitCode ? `exit ${part.exitCode}` : "failed";
  const args = Object.entries(part.args);
  return (
    <div className={`bt-tool bt-tool-${part.status}`}>
      <button type="button" className="bt-tool-head" aria-expanded={open} onClick={() => setOpen(!open)}>
        {part.status === "running" ? <Spinner size={12} /> : <span className={`bt-dot bt-dot-${part.status}`} aria-hidden="true" />}
        <span className="bt-tool-label">{part.label}</span>
        {summary ? <span className="bt-tool-arg">{summary}</span> : null}
        <span className="bt-tool-state">
          {state}
          {part.durationMs !== undefined && part.status !== "running" ? ` · ${(part.durationMs / 1000).toFixed(1)}s` : ""}
        </span>
        <Icon name="chevron" size={14} className={open ? "bt-rot-up" : ""} />
      </button>
      {open ? (
        <div className="bt-tool-body">
          {args.length ? (
            <dl className="bt-tool-args">
              {args.map(([k, v]) => (
                <div key={k}>
                  <dt>{k}</dt>
                  <dd><pre>{typeof v === "string" ? v : JSON.stringify(v)}</pre></dd>
                </div>
              ))}
            </dl>
          ) : null}
          {part.output !== undefined ? (
            <>
              <div className="bt-tool-out-label">Output{part.truncated ? " (truncated)" : ""}</div>
              <pre className="bt-tool-out">{part.output || "(no output)"}</pre>
            </>
          ) : part.status === "running" ? (
            <p className="bt-muted">Running…</p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

// --------------------------------------------------------------------------- //
// real-state indicators
// --------------------------------------------------------------------------- //
function useElapsed(since: number | null): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (since === null) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [since]);
  return since === null ? 0 : Math.max(0, (now - since) / 1000);
}

function StatusLine({ text, since }: { text: string; since: number | null }) {
  const elapsed = useElapsed(since);
  return (
    <div className="bt-status-line" role="status">
      <Spinner size={12} />
      <span>{text}</span>
      {since !== null && elapsed >= 2 ? <span className="bt-muted">{formatDuration(elapsed)}</span> : null}
    </div>
  );
}

function ErrorBlock({
  error, canRetry, onRetry, onStartGpu, onOpenGpu, gpu,
}: {
  error: ErrorInfo; canRetry: boolean; onRetry: () => void; onStartGpu: () => void; onOpenGpu: () => void;
  gpu: GpuSnapshot | null;
}) {
  const gpuStarting = gpu?.state === "starting";
  return (
    <div className="bt-error" role="alert">
      <Icon name="alert" size={16} />
      <div className="bt-error-text">
        <p>{error.message}</p>
        <div className="bt-error-actions">
          {error.action === "start_gpu" && !gpuStarting ? <button type="button" className="bt-btn bt-btn-small" onClick={onStartGpu}>Start GPU</button> : null}
          {error.action === "start_gpu" && gpuStarting ? <button type="button" className="bt-btn bt-btn-small" onClick={onOpenGpu}>GPU is starting — view</button> : null}
          {error.action === "check_gpu" ? <button type="button" className="bt-btn bt-btn-small" onClick={onOpenGpu}>Check GPU</button> : null}
          {canRetry && (error.retryable || error.action === "retry" || error.action === "start_gpu" || error.action === "check_gpu") ? (
            <button type="button" className="bt-btn bt-btn-small" onClick={onRetry}>Retry</button>
          ) : null}
        </div>
      </div>
    </div>
  );
}

const FINISH_NOTES: Record<string, string> = {
  cancelled: "Stopped",
  interrupted: "Interrupted — the server restarted while this answer was being written",
  length: "Cut off — the reply reached the length limit",
  incomplete: "Ended early — the tool budget for one reply was reached",
};

// --------------------------------------------------------------------------- //
export interface MessageViewProps {
  message: UiMessage;
  isLast: boolean;
  turn: TurnState | null; // non-null only for the message that is streaming right now
  gpu: GpuSnapshot | null;
  onRegenerate: () => void;
  onStartGpu: () => void;
  onOpenGpu: () => void;
}

function MessageViewImpl({ message, isLast, turn, gpu, onRegenerate, onStartGpu, onOpenGpu }: MessageViewProps) {
  const [copied, setCopied] = useState(false);
  const streaming = message.status === "streaming";

  if (message.role === "user") {
    return (
      <article className="bt-msg bt-msg-user" aria-label="Your message">
        <div className="bt-msg-label">You</div>
        <div className="bt-user-text">{messageText(message)}</div>
        {message.attachments?.length ? (
          <ul className="bt-attached">
            {message.attachments.map((a, i) => <li key={`${a}${i}`}>{a}</li>)}
          </ul>
        ) : null}
      </article>
    );
  }

  const hasContent = message.parts.length > 0;
  const lastPart = message.parts[message.parts.length - 1];
  const trailingText = lastPart && lastPart.t === "text";
  let status: { text: string; since: number | null } | null = null;
  if (streaming && turn) {
    if (turn.status === "reconnecting") status = { text: `Connection lost — reconnecting (attempt ${turn.reconnectAttempt})…`, since: null };
    else if (turn.status === "stopping") status = { text: "Stopping…", since: null };
    else if (turn.reasoning.active) status = { text: "Thinking…", since: turn.reasoning.since };
    else if (turn.status === "connecting") status = { text: "Connecting…", since: null };
    else if (!hasContent) status = { text: "Waiting for the model…", since: null };
    else if (!trailingText && lastPart?.t === "tool" && lastPart.status !== "running") status = { text: "Working…", since: null };
  }

  const finishNote = message.finishReason ? FINISH_NOTES[message.finishReason] : undefined;
  const text = messageText(message);
  const onCopy = async () => {
    if (await copyTextToClipboard(text)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }
  };

  return (
    <article className={`bt-msg bt-msg-assistant${streaming ? " is-streaming" : ""}`} aria-label="Blackthorn" aria-busy={streaming}>
      <div className="bt-msg-label">Blackthorn</div>
      <div className="bt-parts">
        {message.parts.map((p, i) =>
          p.t === "text" ? (
            p.text.trim() || (streaming && i === message.parts.length - 1) ? (
              <Markdown key={i} text={p.text} streaming={streaming && i === message.parts.length - 1} />
            ) : null
          ) : (
            <ToolRow key={p.id || i} part={p} />
          ),
        )}
      </div>
      {status ? <StatusLine text={status.text} since={status.since} /> : null}
      {message.notices?.map((n, i) => (
        <p key={i} className="bt-notice">{n}</p>
      ))}
      {message.error ? (
        <ErrorBlock error={message.error} canRetry={isLast && !streaming} onRetry={onRegenerate} onStartGpu={onStartGpu} onOpenGpu={onOpenGpu} gpu={gpu} />
      ) : null}
      {!streaming ? (
        <footer className="bt-msg-foot">
          {text ? (
            <button type="button" className="bt-foot-btn" onClick={() => void onCopy()}>
              {copied ? "Copied" : "Copy"}
            </button>
          ) : null}
          {isLast && !message.error ? (
            <button type="button" className="bt-foot-btn" onClick={onRegenerate}>
              Regenerate
            </button>
          ) : null}
          <span className="bt-foot-meta">
            {finishNote ? <span className="bt-finish-note">{finishNote}</span> : null}
            {message.thinkingMs ? <span>Thought for {formatDuration(message.thinkingMs / 1000)}</span> : null}
            {message.durationMs ? <span>{formatDuration(message.durationMs / 1000)}</span> : null}
            {message.completionTokens ? <span>{message.completionTokens} tokens</span> : null}
          </span>
        </footer>
      ) : null}
    </article>
  );
}

export const MessageView = memo(MessageViewImpl, (a, b) =>
  a.message === b.message && a.isLast === b.isLast && a.turn === b.turn && a.gpu?.state === b.gpu?.state,
);

// --------------------------------------------------------------------------- //
export function Welcome({
  gpu, unreachable, onPick, onStartGpu, onOpenGpu,
}: {
  gpu: GpuSnapshot | null; unreachable: boolean; onPick: (text: string) => void; onStartGpu: () => void; onOpenGpu: () => void;
}) {
  const state = gpu?.state;
  return (
    <div className="bt-welcome">
      <h1>Blackthorn</h1>
      <p className="bt-welcome-sub">
        Ask anything. When a question needs fresh information or work on the machine, the model decides to search the
        web or run code — and you see exactly what it did.
      </p>
      {unreachable ? (
        <p className="bt-welcome-gpu bt-warn">Can’t reach the server to check the GPU right now.</p>
      ) : state === "off" || state === "unknown" || !state ? (
        <p className="bt-welcome-gpu">
          The GPU is off. <button type="button" className="bt-link-btn" onClick={onStartGpu}>Start the GPU</button> to chat
          {state === "off" ? " (about 5–8 minutes)" : ""}.
        </p>
      ) : state === "starting" ? (
        <p className="bt-welcome-gpu">
          <Spinner size={12} /> {gpu?.message || "The GPU is starting…"}{" "}
          <button type="button" className="bt-link-btn" onClick={onOpenGpu}>Details</button>
        </p>
      ) : state === "error" ? (
        <p className="bt-welcome-gpu bt-warn">
          {gpu?.error?.message || "The GPU hit an error."}{" "}
          <button type="button" className="bt-link-btn" onClick={onOpenGpu}>Open GPU panel</button>
        </p>
      ) : state === "stopping" ? (
        <p className="bt-welcome-gpu"><Spinner size={12} /> The GPU is stopping…</p>
      ) : (
        <p className="bt-welcome-gpu bt-ok">The GPU is ready.</p>
      )}
      <div className="bt-suggestions" role="list">
        {[
          "Explain how TCP congestion control works, with a short example.",
          "Write a Python function that merges overlapping intervals, and test it.",
          "What are the latest developments in open-weight language models?",
          "Check how much disk space and GPU memory this machine has.",
        ].map((t) => (
          <button key={t} type="button" role="listitem" className="bt-suggestion" onClick={() => onPick(t)}>
            {t}
          </button>
        ))}
      </div>
    </div>
  );
}
